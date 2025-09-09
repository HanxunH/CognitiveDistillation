import argparse
import paddle
import random
import numpy as np
import time
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='CognitiveDistillation')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options (simplified from original)
parser.add_argument('--dataset', type=str, default='cifar_badnet', help='dataset name')
parser.add_argument('--model', type=str, default='resnet18', help='model name')
parser.add_argument('--method', type=str, default="CD")
parser.add_argument('--poison_rate', type=float, default=0.01, help='poison rate')
parser.add_argument('--output_dir', type=str, default='./output', help='output directory')

# Distilation Parameters
parser.add_argument('--p', default=1, type=int)
parser.add_argument('--gamma', default=0.01, type=float)
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--num_steps', default=100, type=int)
parser.add_argument('--step_size', default=0.1, type=float)
parser.add_argument('--mask_channel', default=1, type=int)
parser.add_argument('--norm_only', action='store_true', default=False)


def get_dataset(args):
    """Load dataset based on name"""
    data_root = '/root/CognitiveDistillation/paddlepaddle/data'
    
    if args.dataset == 'cifar_badnet':
        from datasets.cifar_badnet import BadNetCIFAR10
        train_set = BadNetCIFAR10(root=data_root, train=True, download=False, 
                                 poison_rate=args.poison_rate)
        test_set = BadNetCIFAR10(root=data_root, train=False, download=False, 
                                poison_rate=args.poison_rate)
    elif args.dataset == 'cifar_blend':
        from datasets.cifar_blend import BlendCIFAR10
        train_set = BlendCIFAR10(root=data_root, train=True, download=False, 
                                poison_rate=args.poison_rate)
        test_set = BlendCIFAR10(root=data_root, train=False, download=False, 
                               poison_rate=args.poison_rate)
    elif args.dataset == 'cifar_trojan':
        from datasets.cifar_trojan import TrojanCIFAR10
        train_set = TrojanCIFAR10(root=data_root, train=True, download=False, 
                                 poison_rate=args.poison_rate)
        test_set = TrojanCIFAR10(root=data_root, train=False, download=False, 
                                poison_rate=args.poison_rate)
    elif args.dataset == 'cifar_cl':
        from datasets.cifar_cl import CLCIFAR10
        train_set = CLCIFAR10(root=data_root, train=True, download=False, 
                             poison_rate=args.poison_rate)
        test_set = CLCIFAR10(root=data_root, train=False, download=False, 
                            poison_rate=args.poison_rate)
    elif args.dataset == 'cifar_dynamic':
        from datasets.cifar_dynamic import DynamicCIFAR10
        train_set = DynamicCIFAR10(root=data_root, train=True, download=False, 
                                  poison_rate=args.poison_rate)
        test_set = DynamicCIFAR10(root=data_root, train=False, download=False, 
                                 poison_rate=args.poison_rate)
    elif args.dataset == 'cifar_wanet':
        from datasets.cifar_wanet import WaNetCIFAR10
        train_set = WaNetCIFAR10(root=data_root, train=True, download=False, 
                                poison_rate=args.poison_rate)
        test_set = WaNetCIFAR10(root=data_root, train=False, download=False, 
                               poison_rate=args.poison_rate)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    return train_set, test_set


def get_model(args):
    """Load model based on name"""
    if args.model == 'resnet18':
        from models.resnet import ResNet18
        model = ResNet18(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Load pretrained weights if available
    model_path = os.path.join(args.output_dir, f'{args.dataset}_{args.model}.pdparams')
    if os.path.exists(model_path):
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: No pretrained model found at {model_path}, using random initialization")
    
    model.eval()
    for param in model.parameters():
        param.stop_gradient = True
    
    return model


def main():
    args = parser.parse_args()
    
    # Set random seed
    paddle.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    train_set, test_set = get_dataset(args)
    
    # Create data loaders
    train_loader = paddle.io.DataLoader(
        train_set, 
        batch_size=128, 
        shuffle=False,
        num_workers=4
    )
    
    # For backdoor test set, we use the test_set which already contains poisoned samples
    bd_test_loader = paddle.io.DataLoader(
        test_set,
        batch_size=128,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    model = get_model(args)
    
    # Initialize detection method
    if 'CD' in args.method:
        from detection.cognitive_distillation import CognitiveDistillation
        detector = CognitiveDistillation(p=args.p, gamma=args.gamma, beta=args.beta,
                                       num_steps=args.num_steps, lr=args.step_size,
                                       mask_channel=args.mask_channel, norm_only=args.norm_only)
        if args.method == 'CD':
            hyper_params = (args.p, args.mask_channel, args.gamma, args.beta, args.num_steps, args.step_size)
            file_extension = 'p={:d}_c={:d}_gamma={:6f}_beta={:6f}_steps={:d}_step_size={:3f}.pt'.format(*hyper_params)
            train_filename = 'cd_train_mask_' + file_extension
            test_filename = 'cd_bd_test_mask_' + file_extension
        elif args.method == 'CD_FE':
            # CD_FE not implemented in simplified version
            raise NotImplementedError("CD_FE method not implemented in simplified version")
    elif args.method == 'STRIP':
        from detection.strip import STRIP_Detection
        train_filename = 'train_STRIP_entropy.pt'
        test_filename = 'bd_test_STRIP_entropy.pt'
        
        # Prepare STRIP data
        if hasattr(train_set, 'data'):
            strip_data = train_set.data
            strip_data = paddle.to_tensor(strip_data).transpose([0, 3, 1, 2])
            strip_data = strip_data / 255.0
        else:
            # For other datasets
            idx = np.random.choice(range(len(train_set)), size=5000)
            imgs = []
            for i in idx:
                img, target = train_set[i]
                imgs.append(img)
            imgs = paddle.stack(imgs)
            strip_data = imgs
        
        detector = STRIP_Detection(strip_data)
    elif args.method == 'Feature':
        from detection.get_features import Feature_Detection
        train_filename = 'train_features.pt'
        test_filename = 'bd_test_features.pt'
        detector = Feature_Detection()
    elif args.method == 'AC':
        # Load features for AC analysis
        from analysis.activation_clustering import ACAnalysis
        train_features_path = os.path.join(args.output_dir, 'train_features.pt')
        if not os.path.exists(train_features_path):
            raise ValueError(f"Features not found at {train_features_path}. Run Feature method first.")
        
        train_filename = 'train_AC_scores.pt'
        test_filename = 'bd_test_AC_scores.pt'
        detector = None  # AC uses pre-extracted features
    elif args.method == 'SS':
        # Load features for SS analysis
        from analysis.spectral_signatures import SSAnalysis
        train_features_path = os.path.join(args.output_dir, 'train_features.pt')
        if not os.path.exists(train_features_path):
            raise ValueError(f"Features not found at {train_features_path}. Run Feature method first.")
        
        train_filename = 'train_SS_scores.pt'
        test_filename = 'bd_test_SS_scores.pt'
        detector = None  # SS uses pre-extracted features
    elif args.method == 'FCT':
        from detection.fct import FCT_Detection
        train_filename = 'train_fct_consistency.pt'
        test_filename = 'bd_test_fct_consistency.pt'
        detector = FCT_Detection(model, train_loader)
    elif args.method == 'Frequency':
        from analysis.frequency import FrequencyAnalysis
        train_filename = 'train_frequency_scores.pt'
        test_filename = 'bd_test_frequency_scores.pt'
        detector = FrequencyAnalysis(input_size=32)
    else:
        raise ValueError('Unknown method')
    
    # Handle AC and SS methods separately
    if args.method in ['AC', 'SS']:
        # Load pre-extracted features
        train_features = paddle.load(os.path.join(args.output_dir, 'train_features.pt'))
        test_features = paddle.load(os.path.join(args.output_dir, 'bd_test_features.pt'))
        
        # Flatten features to 2D as done in original PyTorch code
        train_features = train_features.flatten(start_axis=1)
        test_features = test_features.flatten(start_axis=1)
        
        # Get labels from datasets
        train_labels = paddle.to_tensor([train_set[i][1] for i in range(len(train_set))])
        test_labels = paddle.to_tensor([test_set[i][1] for i in range(len(test_set))])
        
        # Get class indices
        num_classes = 10  # CIFAR-10
        train_cls_idx = [paddle.where(train_labels == i)[0].squeeze().numpy() for i in range(num_classes)]
        test_cls_idx = [paddle.where(test_labels == i)[0].squeeze().numpy() for i in range(num_classes)]
        
        if args.method == 'AC':
            analyzer = ACAnalysis()
            analyzer.train(train_features, train_labels.numpy(), train_cls_idx)
            train_scores = analyzer.analysis(train_features, train_labels.numpy(), train_cls_idx)
            test_scores = analyzer.analysis(test_features, test_labels.numpy(), test_cls_idx)
        else:  # SS
            analyzer = SSAnalysis()
            analyzer.train(train_features, train_labels.numpy(), train_cls_idx)
            train_scores = analyzer.analysis(train_features, train_labels.numpy(), train_cls_idx)
            test_scores = analyzer.analysis(test_features, test_labels.numpy(), test_cls_idx)
        
        # Save results
        paddle.save(paddle.to_tensor(train_scores), os.path.join(args.output_dir, train_filename))
        print(f"{train_filename} saved!")
        paddle.save(paddle.to_tensor(test_scores), os.path.join(args.output_dir, test_filename))
        print(f"{test_filename} saved!")
    elif args.method == 'Frequency':
        # Frequency analysis works on the dataset directly
        print(f"Running Frequency analysis on training set...")
        detector.train(train_set)
        train_scores = detector.analysis(train_set)
        
        print(f"Running Frequency analysis on test set...")
        test_scores = detector.analysis(test_set)
        
        # Save results
        paddle.save(paddle.to_tensor(train_scores), os.path.join(args.output_dir, train_filename))
        print(f"{train_filename} saved!")
        paddle.save(paddle.to_tensor(test_scores), os.path.join(args.output_dir, test_filename))
        print(f"{test_filename} saved!")
    else:
        # Run detections on training set
        print(f"Running {args.method} detection on training set...")
        results = []
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            # DataLoader already returns tensors
            batch_rs = detector(model, images, labels)
            results.append(batch_rs.detach().cpu())
        
        results = paddle.concat(results, axis=0)
        print('results shape', results.shape)
        
        # Save results to file
        filename = os.path.join(args.output_dir, train_filename)
        paddle.save(results, filename)
        print(filename + ' saved!')
        
        # Run detections on backdoor test set
        print(f"Running {args.method} detection on backdoor test set...")
        results = []
        for batch_idx, (images, labels) in enumerate(tqdm(bd_test_loader)):
            # DataLoader already returns tensors
            batch_rs = detector(model, images, labels)
            results.append(batch_rs.detach().cpu())
        
        results = paddle.concat(results, axis=0)
        print('results shape', results.shape)
        
        # Save results to file
        filename = os.path.join(args.output_dir, test_filename)
        paddle.save(results, filename)
        print(filename + ' saved!')
    
    return


if __name__ == '__main__':
    args = parser.parse_args()
    paddle.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    start = time.time()
    main()
    end = time.time()
    
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    print(payload)