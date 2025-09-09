"""
Simplified MindSpore implementation of extract.py for Phase 2 testing.

This module provides basic extraction functionality for:
- CD (Cognitive Distillation) detection
- STRIP detection
- Support for CIFAR datasets (badnet, blend, trojan)
- Basic model loading and inference
"""

import argparse
import os
import sys
# Add mindspore_impl to path to ensure we import the correct modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import mindspore as ms
from mindspore import Tensor, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import ops
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import MindSpore implementations
from mindspore_impl import models
from mindspore_impl import detection
from mindspore_impl.datasets import DatasetGenerator

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)
if ms.get_context("device_target") == "CPU":
    device = "cpu"
else:
    device = "gpu"
    ms.set_context(device_target="GPU")


parser = argparse.ArgumentParser(description='Simple Extract for MindSpore')
# Dataset and model options
parser.add_argument('--dataset', type=str, default='cifar_badnet', 
                    help='Dataset name (cifar_badnet, cifar_blend, cifar_trojan)')
parser.add_argument('--model', type=str, default='resnet18',
                    help='Model architecture')
parser.add_argument('--data_path', type=str, default='./data',
                    help='Path to dataset')

# Detection method
parser.add_argument('--method', type=str, default='CD',
                    help='Detection method (CD, STRIP)')

# Model checkpoint
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to model checkpoint')
parser.add_argument('--num_classes', type=int, default=10,
                    help='Number of classes')

# Batch size
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for evaluation')

# CD parameters
parser.add_argument('--p', default=1, type=int,
                    help='CD parameter p')
parser.add_argument('--gamma', default=0.01, type=float,
                    help='CD gamma parameter')
parser.add_argument('--beta', default=1.0, type=float,
                    help='CD beta parameter')
parser.add_argument('--num_steps', default=100, type=int,
                    help='Number of optimization steps')
parser.add_argument('--step_size', default=0.1, type=float,
                    help='Step size for optimization')
parser.add_argument('--mask_channel', default=1, type=int,
                    help='Number of mask channels')
parser.add_argument('--norm_only', action='store_true', default=False,
                    help='Use norm only')

# Random seed
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')

# Output directory
parser.add_argument('--output_dir', type=str, default='./output',
                    help='Output directory for results')


def get_dataset_config(dataset_name):
    """Map dataset names to configuration."""
    dataset_map = {
        'cifar_badnet': ('BadNetCIFAR10', 'BadNetCIFAR10'),
        'cifar_blend': ('BlendCIFAR10', 'BlendCIFAR10'),
        'cifar_trojan': ('TrojanCIFAR10', 'TrojanCIFAR10'),
        'cifar_cl': ('CLCIFAR10', 'CLCIFAR10'),
        'cifar_dynamic': ('DynamicCIFAR10', 'DynamicCIFAR10'),
        'cifar_wanet': ('WaNetCIFAR10', 'WaNetCIFAR10'),
        'cifar_dfst': ('DFSTCIFAR10', 'DFSTCIFAR10'),
        'cifar10': ('CIFAR10', 'CIFAR10'),
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_map[dataset_name]


def load_model(model_name, num_classes, checkpoint_path=None):
    """Load model architecture and optionally load checkpoint."""
    # Get model architecture
    if model_name == 'resnet18':
        model = models.ResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        model = models.ResNet34(num_classes=num_classes)
    elif model_name == 'resnet50':
        model = models.ResNet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(model, param_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Using randomly initialized model")
    
    # Set to evaluation mode
    model.set_train(False)
    
    return model


def main():
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    ms.set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Running extraction with method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    
    # Get dataset configuration
    train_dataset_type, test_dataset_type = get_dataset_config(args.dataset)
    
    # Create dataset configuration
    dataset_kwargs = {
        'train_bs': args.batch_size,
        'eval_bs': args.batch_size,
        'train_d_type': train_dataset_type,
        'test_d_type': test_dataset_type,
        'train_path': args.data_path,
        'test_path': args.data_path,
        'train_tf_op': 'NoAug',  # No augmentation for extraction
        'test_tf_op': 'NoAug',
        'poison_test_d_type': test_dataset_type,  # Backdoor test set
        'poison_rate': 0.01,  # Default poison rate
        'target_label': 0,     # Default target label
    }
    
    # Create dataset
    data_generator = DatasetGenerator(exp=None, **dataset_kwargs)
    train_loader, test_loader, bd_test_loader = data_generator.get_loader(train_shuffle=False)
    
    # Load model
    model = load_model(args.model, args.num_classes, args.checkpoint)
    
    # Initialize detection method
    if args.method == 'CD':
        detector = detection.CognitiveDistillation(
            p=args.p,
            gamma=args.gamma,
            beta=args.beta,
            num_steps=args.num_steps,
            lr=args.step_size,
            mask_channel=args.mask_channel,
            norm_only=args.norm_only
        )
        
        # Create filenames for saving results
        hyper_params = (args.p, args.mask_channel, args.gamma, args.beta, 
                       args.num_steps, args.step_size)
        file_extension = 'p={:d}_c={:d}_gamma={:6f}_beta={:6f}_steps={:d}_step_size={:3f}.npy'.format(*hyper_params)
        train_filename = f'cd_train_mask_{file_extension}'
        test_filename = f'cd_bd_test_mask_{file_extension}'
        
    elif args.method == 'STRIP':
        # Prepare STRIP data (sample from training set)
        print("Preparing STRIP detection data...")
        
        # Get a sample of training data for STRIP
        strip_data_list = []
        sample_count = 0
        max_samples = 5000  # Use 5000 samples for STRIP
        
        for batch in train_loader.create_tuple_iterator():
            images, labels = batch
            strip_data_list.append(images)
            sample_count += images.shape[0]
            if sample_count >= max_samples:
                break
        
        # Concatenate and trim to exact size
        strip_data = ops.concat(strip_data_list, axis=0)
        strip_data = strip_data[:max_samples]
        
        # Initialize STRIP detector
        detector = detection.STRIP_Detection(strip_data)
        
        train_filename = 'train_STRIP_entropy.npy'
        test_filename = 'bd_test_STRIP_entropy.npy'
        
    elif args.method == 'Feature':
        # Feature extraction for AC/SS detection
        from detection.get_features import Feature_Detection
        detector = Feature_Detection()
        
        # Extract features from training set
        print("Extracting training features...")
        train_features = []
        train_labels = []
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(train_loader)}")
            features = detector(model, images, labels)
            train_features.append(features.asnumpy())
            train_labels.append(labels.asnumpy())
        
        train_features = np.concatenate(train_features, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
        # Extract features from backdoor test set
        print("\nExtracting backdoor test features...")
        bd_test_features = []
        bd_test_labels = []
        for batch_idx, (images, labels) in enumerate(bd_test_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(bd_test_loader)}")
            features = detector(model, images, labels)
            bd_test_features.append(features.asnumpy())
            bd_test_labels.append(labels.asnumpy())
        
        bd_test_features = np.concatenate(bd_test_features, axis=0)
        bd_test_labels = np.concatenate(bd_test_labels, axis=0)
        
        # Save features for AC/SS detection
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'train_features.npy'), train_features)
        np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(output_dir, 'bd_test_features.npy'), bd_test_features)
        np.save(os.path.join(output_dir, 'bd_test_labels.npy'), bd_test_labels)
        
        print(f"\nFeature extraction complete!")
        print(f"Train features shape: {train_features.shape}")
        print(f"Train labels shape: {train_labels.shape}")
        print(f"BD test features shape: {bd_test_features.shape}")
        print(f"BD test labels shape: {bd_test_labels.shape}")
        print(f"Features saved to {output_dir}")
        return  # Exit after feature extraction
    
    elif args.method == 'AC':
        # AC detection using pre-extracted features
        from mindspore_impl.analysis.activation_clustering import ACAnalysis
        
        # Load pre-extracted features
        output_dir = args.output_dir
        train_features = np.load(os.path.join(output_dir, 'train_features.npy'))
        train_labels = np.load(os.path.join(output_dir, 'train_labels.npy'))
        bd_test_features = np.load(os.path.join(output_dir, 'bd_test_features.npy'))
        bd_test_labels = np.load(os.path.join(output_dir, 'bd_test_labels.npy'))
        
        # Create class indices
        num_classes = args.num_classes
        train_cls_idx = [np.where(train_labels == i)[0] for i in range(num_classes)]
        test_cls_idx = [np.where(bd_test_labels == i)[0] for i in range(num_classes)]
        
        # Run AC analysis
        print("Running AC analysis...")
        analyzer = ACAnalysis()
        analyzer.train(train_features, train_labels, train_cls_idx)
        scores = analyzer.analysis(bd_test_features, bd_test_labels, test_cls_idx)
        
        # Save results
        np.save(os.path.join(output_dir, 'ac_scores_cifar_badnet.npy'), scores)
        predictions = analyzer.predict(bd_test_features, bd_test_labels, test_cls_idx)
        np.save(os.path.join(output_dir, 'ac_predictions_cifar_badnet.npy'), predictions)
        
        print(f"AC scores shape: {scores.shape}")
        print(f"AC predictions shape: {predictions.shape}")
        print(f"Results saved to {output_dir}")
        return  # Exit after AC detection
        
    elif args.method == 'SS':
        # SS detection using pre-extracted features
        from mindspore_impl.analysis.spectral_signatures import SSAnalysis
        
        # Load pre-extracted features
        output_dir = args.output_dir
        train_features = np.load(os.path.join(output_dir, 'train_features.npy'))
        train_labels = np.load(os.path.join(output_dir, 'train_labels.npy'))
        bd_test_features = np.load(os.path.join(output_dir, 'bd_test_features.npy'))
        bd_test_labels = np.load(os.path.join(output_dir, 'bd_test_labels.npy'))
        
        # Create class indices
        num_classes = args.num_classes
        train_cls_idx = [np.where(train_labels == i)[0] for i in range(num_classes)]
        test_cls_idx = [np.where(bd_test_labels == i)[0] for i in range(num_classes)]
        
        # Run SS analysis
        print("Running SS analysis...")
        analyzer = SSAnalysis()
        analyzer.train(train_features, train_labels, train_cls_idx)
        scores = analyzer.analysis(bd_test_features, bd_test_labels, test_cls_idx)
        
        # Save results
        np.save(os.path.join(output_dir, 'ss_scores_cifar_badnet.npy'), scores)
        predictions = analyzer.predict(bd_test_features, bd_test_labels, test_cls_idx)
        np.save(os.path.join(output_dir, 'ss_predictions_cifar_badnet.npy'), predictions)
        
        print(f"SS scores shape: {scores.shape}")
        print(f"SS predictions shape: {predictions.shape}")
        print(f"Results saved to {output_dir}")
        return  # Exit after SS detection
        
    elif args.method == 'FCT':
        # FCT detection requires model and train_loader
        from detection.fct import FCT_Detection
        detector = FCT_Detection(model, train_loader)
        train_filename = 'train_FCT_scores.npy'
        test_filename = 'bd_test_FCT_scores.npy'
        
    elif args.method == 'Frequency':
        # Frequency detection
        from mindspore_impl.analysis.frequency import FrequencyAnalysis
        
        # Initialize analyzer (uses pre-trained PyTorch model)
        analyzer = FrequencyAnalysis()
        
        # Process training set
        print("Running Frequency analysis on training set...")
        train_results = []
        for batch in tqdm(train_loader.create_tuple_iterator(), desc="Training set"):
            images, labels = batch
            # Convert to numpy for FrequencyAnalysis (which uses PyTorch internally)
            images_np = images.asnumpy()
            scores = analyzer.analysis(images_np)
            train_results.append(scores)
        
        train_results = np.concatenate(train_results, axis=0)
        
        # Process backdoor test set
        print("Running Frequency analysis on backdoor test set...")
        bd_test_results = []
        for batch in tqdm(bd_test_loader.create_tuple_iterator(), desc="BD test set"):
            images, labels = batch
            # Convert to numpy for FrequencyAnalysis
            images_np = images.asnumpy()
            scores = analyzer.analysis(images_np)
            bd_test_results.append(scores)
        
        bd_test_results = np.concatenate(bd_test_results, axis=0)
        
        # Save results
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        train_filename = 'train_Frequency_scores.npy'
        test_filename = 'bd_test_Frequency_scores.npy'
        
        np.save(os.path.join(output_dir, train_filename), train_results)
        np.save(os.path.join(output_dir, test_filename), bd_test_results)
        
        print(f"\nFrequency analysis complete!")
        print(f"Train results shape: {train_results.shape}")
        print(f"BD test results shape: {bd_test_results.shape}")
        print(f"Results saved to {output_dir}")
        
        # Also get predictions for compatibility
        train_predictions = analyzer.predict(train_results)
        bd_test_predictions = analyzer.predict(bd_test_results)
        
        np.save(os.path.join(output_dir, 'train_Frequency_predictions.npy'), train_predictions)
        np.save(os.path.join(output_dir, 'bd_test_Frequency_predictions.npy'), bd_test_predictions)
        
        return  # Exit after Frequency analysis
        
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Run detection on training set
    print("Running detection on training set...")
    train_results = []
    
    for batch in tqdm(train_loader.create_tuple_iterator(), desc="Training set"):
        images, labels = batch
        # Convert to MindSpore tensors if needed
        if not isinstance(images, Tensor):
            images = Tensor(images, dtype=ms.float32)
        if not isinstance(labels, Tensor):
            labels = Tensor(labels, dtype=ms.int32)
        
        # Run detector
        batch_results = detector(model, images, labels)
        
        # Convert to numpy for saving
        if isinstance(batch_results, Tensor):
            batch_results = batch_results.asnumpy()
        
        train_results.append(batch_results)
    
    # Concatenate results
    train_results = np.concatenate(train_results, axis=0)
    print(f"Train results shape: {train_results.shape}")
    
    # Save training results
    train_path = os.path.join(args.output_dir, train_filename)
    np.save(train_path, train_results)
    print(f"Saved training results to {train_path}")
    
    # Run detection on backdoor test set
    if bd_test_loader is not None:
        print("Running detection on backdoor test set...")
        test_results = []
        
        for batch in tqdm(bd_test_loader.create_tuple_iterator(), desc="Backdoor test set"):
            images, labels = batch
            # Convert to MindSpore tensors if needed
            if not isinstance(images, Tensor):
                images = Tensor(images, dtype=ms.float32)
            if not isinstance(labels, Tensor):
                labels = Tensor(labels, dtype=ms.int32)
            
            # Run detector
            batch_results = detector(model, images, labels)
            
            # Convert to numpy for saving
            if isinstance(batch_results, Tensor):
                batch_results = batch_results.asnumpy()
            
            test_results.append(batch_results)
        
        # Concatenate results
        test_results = np.concatenate(test_results, axis=0)
        print(f"Test results shape: {test_results.shape}")
        
        # Save test results
        test_path = os.path.join(args.output_dir, test_filename)
        np.save(test_path, test_results)
        print(f"Saved test results to {test_path}")
    
    print("Extraction completed successfully!")


if __name__ == '__main__':
    main()