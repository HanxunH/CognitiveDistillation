import argparse
import mlconfig
import torch
import random
import numpy as np
import datasets
import time
import util
import models
import detection
import os
from tqdm import tqdm
from exp_mgmt import ExperimentManager
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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
    model = config.model().to(device)
    ckpt = 'model_state_dict'
    model = exp.load_state(model, ckpt)
    if args.data_parallel:
        model = torch.nn.DataParallel(model).to(device)
        logger.info("Using torch.nn.DataParallel")
    # model.get_features = True
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Update the train transform option
    # Set train transoform to ToTensor() only
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
            file_extension = 'p={:d}_c={:d}_gamma={:6f}_beta={:6f}_steps={:d}_step_size={:3f}.pt'.format(*hyper_params)
            train_filename = 'cd_train_mask_' + file_extension
            test_filename = 'cd_bd_test_mask_' + file_extension
        elif args.method == 'CD_FE':
            hyper_params = (args.p, args.mask_channel, args.gamma, args.beta, args.num_steps, args.step_size)
            file_extension = 'p={:d}_c={:d}_gamma={:6f}_beta={:6f}_steps={:d}_step_size={:3f}.pt'.format(*hyper_params)
            train_filename = 'cd_fe_train_mask_' + file_extension
            test_filename = 'cd_fe_bd_test_mask_' + file_extension
            model.get_features = True
            detector.get_features = True
    elif args.method == 'STRIP':
        train_filename = 'train_STRIP_entropy.pt'
        test_filename = 'bd_test_STRIP_entropy.pt'
        if hasattr(data.train_set, 'data'):
            strip_data = data.train_set.data
            strip_data = torch.tensor(strip_data).permute(0, 3, 1, 2)
            strip_data = strip_data / 255.0
        else:
            # ImageNet for ISBBA
            idx = np.random.choice(range(len(data.train_set)), size=5000)
            imgs = []
            for i in idx:
                img, target = data.train_set[i]
                imgs.append(img)
            imgs = torch.stack(imgs)
            strip_data = imgs
        detector = detection.strip.STRIP_Detection(strip_data)
    elif args.method == 'Feature':
        train_filename = 'train_features.pt'
        test_filename = 'bd_test_features.pt'
        detector = detection.get_features.Feature_Detection()
    elif args.method == 'FCT':
        train_filename = 'train_fct.pt'
        test_filename = 'bd_test_fct.pt'
        detector = detection.fct.FCT_Detection(model, train_loader)
    else:
        raise('Unknown method')

    # Run detections on training set
    results = []
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        batch_rs = detector(model, images, labels)
        results.append(batch_rs.detach().cpu())
    results = torch.cat(results, dim=0)
    print('results shape', results.shape)
    # save resutlts to file
    filename = os.path.join(exp.exp_path, train_filename)
    torch.save(results, filename)
    print(filename + ' saved!')

    # Run detections on backdoor test set
    results = []
    for images, labels in tqdm(bd_test_loader):
        images, labels = images.to(device), labels.to(device)
        batch_rs = detector(model, images, labels)
        results.append(batch_rs.detach().cpu())

    results = torch.cat(results, dim=0)
    print('results shape', results.shape)
    # save resutlts to file
    filename = os.path.join(exp.exp_path, test_filename)
    torch.save(results, filename)
    print(filename + ' saved!')
    return


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename,
                                   eval_mode=True)
    logger = experiment.logger
    logger.info("PyTorch Version: %s" % (torch.__version__))

    if torch.cuda.is_available():
        device_list = [torch.cuda.get_device_name(i)
                       for i in range(0, torch.cuda.device_count())]
        logger.info("GPU List: %s" % (device_list))

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
