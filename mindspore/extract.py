import argparse
import mlconfig
import mindspore as ms
import random
import numpy as np
import time
import util
import os
from tqdm import tqdm
from exp_mgmt import ExperimentManager
from mindspore import context

# Import models and datasets to register with mlconfig
import models
import datasets
import detection

# Set MindSpore context
if ms.context.get_context("device_target") == "CPU":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
else:
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    # MindSpore handles GPU optimization internally


parser = argparse.ArgumentParser(description='CognitiveDistillation')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='rn18', type=str)
parser.add_argument('--exp_path', default='experiments', type=str)
parser.add_argument('--exp_config', default='configs', type=str)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--method', type=str, default="CD")

# Distilation Parameters
parser.add_argument('--p', default=1, type=int)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--num_steps', default=100, type=int)
parser.add_argument('--step_size', default=0.1, type=float)
parser.add_argument('--mask_channel', default=1, type=int)
parser.add_argument('--norm_only', action='store_true', default=False)


def main(exp):
    # Setup model and exp to run
    logger = exp.logger
    config = exp.config
    
    # Initialize model
    model = config.model()
    ckpt = 'model_state_dict'
    model = exp.load_state(model, ckpt)
    
    if args.data_parallel:
        # Skip DataParallel for now as per instructions
        raise NotImplementedError("Skip")
        
    # Set model to eval mode
    model.set_train(False)
    
    # MindSpore doesn't need explicit requires_grad setting in eval mode
    # The model is already in eval mode which disables gradient computation
    
    # Update the train transform option
    # Set train transform to ToTensor() only
    config.set_immutable(False)
    if hasattr(config.dataset, 'train_tf_op'):
        if config.dataset.train_tf_op not in ['None', 'GTSRB']:
            config.dataset.train_tf_op = 'NoAug'
            print('Set to no augmentation')
    data = config.dataset(exp)
    loader = data.get_loader(train_shuffle=False)
    train_loader, test_loader, bd_test_loader = loader

    # Initialize detection method
    if 'CD' in args.method:
        detector = detection.CognitiveDistillation(p=args.p, gamma=args.gamma, beta=args.beta,
                                                   num_steps=args.num_steps, lr=args.step_size,
                                                   mask_channel=args.mask_channel, norm_only=args.norm_only)
        if args.method == 'CD':
            hyper_params = (args.p, args.mask_channel, args.gamma, args.beta, args.num_steps, args.step_size)
            file_extension = 'p={:d}_c={:d}_gamma={:6f}_beta={:6f}_steps={:d}_step_size={:3f}.npy'.format(*hyper_params)
            train_filename = 'cd_train_mask_' + file_extension
            test_filename = 'cd_bd_test_mask_' + file_extension
        elif args.method == 'CD_FE':
            hyper_params = (args.p, args.mask_channel, args.gamma, args.beta, args.num_steps, args.step_size)
            file_extension = 'p={:d}_c={:d}_gamma={:6f}_beta={:6f}_steps={:d}_step_size={:3f}.npy'.format(*hyper_params)
            train_filename = 'cd_fe_train_mask_' + file_extension
            test_filename = 'cd_fe_bd_test_mask_' + file_extension
            model.get_features = True
            detector.get_features = True
    elif args.method == 'STRIP':
        train_filename = 'train_STRIP_entropy.npy'
        test_filename = 'bd_test_STRIP_entropy.npy'
        
        # Get data for STRIP
        if hasattr(data.train_set.torch_dataset, 'data'):
            strip_data = data.train_set.torch_dataset.data
            # Convert to MindSpore tensor with proper shape
            strip_data = ms.Tensor(strip_data, dtype=ms.float32)
            # Permute from NHWC to NCHW
            strip_data = ms.ops.transpose(strip_data, (0, 3, 1, 2))
            strip_data = strip_data / 255.0
        else:
            # ImageNet for ISSBA
            idx = np.random.choice(range(len(data.train_set)), size=5000)
            imgs = []
            for i in idx:
                img, target = data.train_set[i]
                # img is already numpy array in CHW format from TorchDatasetWrapper
                imgs.append(ms.Tensor(img, dtype=ms.float32))
            imgs = ms.ops.stack(imgs)
            strip_data = imgs
        detector = detection.strip.STRIP_Detection(strip_data)
    elif args.method == 'Feature':
        train_filename = 'train_features.npy'
        test_filename = 'bd_test_features.npy'
        detector = detection.get_features.Feature_Detection()
    elif args.method == 'FCT':
        train_filename = 'train_fct.npy'
        test_filename = 'bd_test_fct.npy'
        detector = detection.fct.FCT_Detection(model, train_loader)
    else:
        raise TypeError('Unknown method')  # Using TypeError per mindspore-exemptions.md #1

    # Run detections on training set
    results = []
    for data_batch in tqdm(train_loader.create_tuple_iterator()):
        images, labels = data_batch
        # Convert to MindSpore tensors
        images = ms.Tensor(images, dtype=ms.float32)
        labels = ms.Tensor(labels, dtype=ms.int32)
        
        batch_rs = detector(model, images, labels)
        # Convert to numpy for storage
        results.append(batch_rs.asnumpy())
    
    results = np.concatenate(results, axis=0)
    print('results shape', results.shape)
    
    # Save results to file
    filename = os.path.join(exp.exp_path, train_filename)
    np.save(filename, results)
    print(filename + ' saved!')

    # Run detections on backdoor test set
    results = []
    for data_batch in tqdm(bd_test_loader.create_tuple_iterator()):
        images, labels = data_batch
        # Convert to MindSpore tensors
        images = ms.Tensor(images, dtype=ms.float32)
        labels = ms.Tensor(labels, dtype=ms.int32)
        
        batch_rs = detector(model, images, labels)
        # Convert to numpy for storage
        results.append(batch_rs.asnumpy())

    results = np.concatenate(results, axis=0)
    print('results shape', results.shape)
    
    # Save results to file
    filename = os.path.join(exp.exp_path, test_filename)
    np.save(filename, results)
    print(filename + ' saved!')
    return


if __name__ == '__main__':
    args = parser.parse_args()
    ms.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename,
                                   eval_mode=True)
    logger = experiment.logger
    logger.info("MindSpore Version: %s" % (ms.__version__))

    # Log device information
    device_target = context.get_context("device_target")
    logger.info("Device Target: %s" % device_target)
    
    if device_target == "GPU":
        # MindSpore doesn't provide direct GPU name access like PyTorch
        logger.info("Running on GPU")

    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    for key in experiment.config:
        logger.info("%s: %s" % (key, experiment.config[key]))

    start = time.time()
    main(experiment)
    end = time.time()

    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)