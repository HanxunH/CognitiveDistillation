#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorflow_impl import datasets, models, detection
from tensorflow_impl.analysis import cognitive_distillation as cd_analysis
from tensorflow_impl.analysis import activation_clustering, spectral_signatures
from tensorflow_impl.datasets.dataset import TorchDatasetWrapper

parser = argparse.ArgumentParser(description='Simple Extract for Testing')
parser.add_argument('--dataset', type=str, default='cifar_badnet', 
                    help='Dataset name (cifar_badnet, cifar_blend, cifar_trojan, etc.)')
parser.add_argument('--model', type=str, default='resnet18',
                    help='Model architecture')
parser.add_argument('--method', type=str, default='CD',
                    help='Detection method (CD, STRIP, FCT, AC, SS, Feature)')
parser.add_argument('--poison_rate', type=float, default=0.01,
                    help='Poison rate for backdoor datasets')
parser.add_argument('--target_label', type=int, default=0,
                    help='Target label for backdoor attack')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')
# CD specific parameters
parser.add_argument('--gamma', type=float, default=0.01,
                    help='CD gamma parameter')
parser.add_argument('--beta', type=float, default=10.0,
                    help='CD beta parameter')
parser.add_argument('--num_steps', type=int, default=100,
                    help='CD optimization steps')
parser.add_argument('--lr', type=float, default=0.1,
                    help='CD learning rate')
parser.add_argument('--p', type=int, default=1,
                    help='CD p parameter')

def load_dataset(args):
    """Load dataset based on arguments"""
    print(f"Loading dataset: {args.dataset}")
    
    # Map common dataset names to registry names
    dataset_name_map = {
        'cifar_badnet': 'BadNetCIFAR10',
        'cifar_blend': 'BlendCIFAR10',
        'cifar_trojan': 'TrojanCIFAR10',
        'cifar_cl': 'CLCIFAR10',
        'cifar_dynamic': 'DynamicCIFAR10',
        'cifar_wanet': 'WaNetCIFAR10',
        'cifar_dfst': 'DFSTCIFAR10',
        'cifar_smooth': 'SmoothCIFAR10',
        'cifar_sig': 'SIGCIFAR10',
        'cifar_fc': 'FCCIFAR10',
        'cifar_nashville': 'NashvilleCIFAR10',
        'cifar10': 'CIFAR10',
        'cifar100': 'CIFAR100',
        'svhn': 'SVHN',
        'mnist': 'MNIST',
        'gtsrb': 'GTSRB',
    }
    
    # Get the registry name
    registry_name = dataset_name_map.get(args.dataset, args.dataset)
    
    # Get dataset from registry
    if registry_name in datasets.utils.dataset_options:
        dataset_fn = datasets.utils.dataset_options[registry_name]
        
        # Build kwargs for dataset creation
        kwargs = {}
        if 'BadNet' in registry_name or 'Blend' in registry_name or 'Trojan' in registry_name or 'CL' in registry_name or 'Dynamic' in registry_name or 'WaNet' in registry_name:
            kwargs['poison_rate'] = args.poison_rate
            kwargs['target_label'] = args.target_label
        
        # Create train and test datasets with transforms
        transform = datasets.utils.transform_options['NoAug']
        train_transform = transform['train_transform']
        test_transform = transform['test_transform']
        
        # Compose transforms
        from torchvision import transforms
        if train_transform:
            train_transform = transforms.Compose(train_transform)
        if test_transform:
            test_transform = transforms.Compose(test_transform)
        
        train_set = dataset_fn('data', train_transform, False, kwargs)
        test_set = dataset_fn('data', test_transform, True, kwargs)
        
        # Wrap with TorchDatasetWrapper for TensorFlow compatibility
        train_set = TorchDatasetWrapper(train_set)
        test_set = TorchDatasetWrapper(test_set)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset} (registry name: {registry_name})")
    
    return train_set, test_set

def load_model(args):
    """Load model based on arguments"""
    print(f"Loading model: {args.model}")
    
    if args.model == 'resnet18':
        model = models.resnet.ResNet18(num_classes=10)
    elif args.model == 'resnet34':
        model = models.resnet.ResNet34(num_classes=10)
    elif args.model == 'resnet50':
        model = models.resnet.ResNet50(num_classes=10)
    elif args.model == 'mobilenetv2':
        model = models.mobilenetv2.MobileNetV2(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Initialize with random weights for testing
    # In real usage, you would load pretrained weights here
    return model

def run_detection(model, dataset, detector, batch_size=128):
    """Run detection on a dataset"""
    results = []
    num_samples = len(dataset)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Running detection on {num_samples} samples...")
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        
        # Get batch of data
        batch_images = []
        batch_labels = []
        for j in range(start_idx, end_idx):
            img, label = dataset[j]
            batch_images.append(img)
            batch_labels.append(label)
        
        # Stack into batch tensors
        images = tf.stack(batch_images)
        labels = tf.constant(batch_labels)
        
        # Run detector - CD expects (model, images, preprocessor, labels)
        # Other detectors expect (model, images, labels)
        if isinstance(detector, detection.CognitiveDistillation):
            batch_results = detector(model, images, None, labels)
        else:
            batch_results = detector(model, images, labels)
        
        # Convert to numpy if needed
        if isinstance(batch_results, tf.Tensor):
            batch_results = batch_results.numpy()
        
        results.append(batch_results)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {end_idx}/{num_samples} samples")
    
    # Concatenate all results
    results = np.concatenate(results, axis=0)
    return results

def main():
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    print("=" * 60)
    print("Simple Extract - TensorFlow Implementation")
    print("=" * 60)
    
    # Load dataset
    train_set, test_set = load_dataset(args)
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Load model
    model = load_model(args)
    print(f"Model loaded: {args.model}")
    
    # Initialize detector based on method
    if args.method == 'CD':
        print(f"\nInitializing Cognitive Distillation detector")
        print(f"  Parameters: gamma={args.gamma}, beta={args.beta}, num_steps={args.num_steps}")
        detector = detection.CognitiveDistillation(
            lr=args.lr, p=args.p, gamma=args.gamma, 
            beta=args.beta, num_steps=args.num_steps
        )
        
        # Run detection on test set
        start_time = time.time()
        results = run_detection(model, test_set, detector, args.batch_size)
        elapsed = time.time() - start_time
        
        print(f"\nResults shape: {results.shape}")
        print(f"Results range: [{results.min():.4f}, {results.max():.4f}]")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        # Save results
        output_file = f'cd_results_{args.dataset}.npy'
        np.save(output_file, results)
        print(f"Results saved to {output_file}")
        
    elif args.method == 'STRIP':
        print(f"\nInitializing STRIP detector")
        
        # Get background data from training set
        strip_data = []
        num_background = min(5000, len(train_set))
        for i in range(num_background):
            img, _ = train_set[i]
            strip_data.append(img)
        strip_data = tf.stack(strip_data)
        
        detector = detection.STRIP_Detection(strip_data)
        
        # Run detection on test set
        start_time = time.time()
        results = run_detection(model, test_set, detector, args.batch_size)
        elapsed = time.time() - start_time
        
        print(f"\nResults shape: {results.shape}")
        print(f"Entropy range: [{results.min():.4f}, {results.max():.4f}]")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        # Save results
        output_file = f'strip_results_{args.dataset}.npy'
        np.save(output_file, results)
        print(f"Results saved to {output_file}")
        
    elif args.method == 'CD_Analysis':
        # Test the CognitiveDistillationAnalysis class
        print(f"\nTesting CognitiveDistillationAnalysis")
        
        # First run CD to get masks
        detector = detection.CognitiveDistillation(
            lr=args.lr, p=args.p, gamma=args.gamma, 
            beta=args.beta, num_steps=args.num_steps
        )
        
        # Get masks for a subset of data
        num_samples = min(100, len(test_set))
        masks = []
        for i in range(num_samples):
            img, label = test_set[i]
            images = tf.expand_dims(img, 0)
            labels = tf.constant([label])
            mask = detector(model, images, None, labels)
            masks.append(mask[0])
        
        masks = tf.stack(masks)
        print(f"Generated {len(masks)} masks")
        
        # Initialize analyzer
        analyzer = cd_analysis.CognitiveDistillationAnalysis()
        
        # Train on first 80%
        train_masks = masks[:80]
        analyzer.train(train_masks)
        print(f"Trained analyzer on {len(train_masks)} masks")
        print(f"  Mean: {analyzer.mean:.4f}, Std: {analyzer.std:.4f}")
        
        # Analyze remaining 20%
        test_masks = masks[80:]
        scores = analyzer.analysis(test_masks)
        print(f"Analysis scores shape: {scores.shape}")
        print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Test predict method
        predictions = analyzer.predict(test_masks, t=1)
        print(f"Predictions: {predictions}")
        
    elif args.method == 'Feature':
        print(f"\nExtracting features using Feature Detection")
        
        detector = detection.Feature_Detection()
        
        # Extract features from train set for AC/SS analysis
        print("Extracting training features...")
        train_features = []
        train_labels = []
        num_train = min(5000, len(train_set))
        for i in range(0, num_train, args.batch_size):
            batch_images = []
            batch_labels = []
            for j in range(i, min(i + args.batch_size, num_train)):
                img, label = train_set[j]
                batch_images.append(img)
                batch_labels.append(label)
            
            images = tf.stack(batch_images)
            labels = tf.constant(batch_labels)
            features = detector(model, images, labels)
            train_features.append(features)
            train_labels.extend(batch_labels)
            
            if (i + args.batch_size) % 500 == 0:
                print(f"  Processed {min(i + args.batch_size, num_train)}/{num_train} samples")
        
        train_features = tf.concat(train_features, axis=0)
        train_labels = tf.constant(train_labels)
        
        # Extract features from test set  
        print("Extracting test features...")
        test_features = []
        test_labels = []
        num_test = min(1000, len(test_set))
        for i in range(0, num_test, args.batch_size):
            batch_images = []
            batch_labels = []
            for j in range(i, min(i + args.batch_size, num_test)):
                img, label = test_set[j]
                batch_images.append(img)
                batch_labels.append(label)
            
            images = tf.stack(batch_images)
            labels = tf.constant(batch_labels)
            features = detector(model, images, labels)
            test_features.append(features)
            test_labels.extend(batch_labels)
        
        test_features = tf.concat(test_features, axis=0)
        test_labels = tf.constant(test_labels)
        
        print(f"\nTrain features shape: {train_features.shape}")
        print(f"Test features shape: {test_features.shape}")
        
        # Save features for AC/SS analysis
        np.save('train_features.npy', train_features.numpy())
        np.save('train_labels.npy', train_labels.numpy())
        np.save('bd_test_features.npy', test_features.numpy())
        np.save('bd_test_labels.npy', test_labels.numpy())
        print("Features saved to train_features.npy and bd_test_features.npy")
        
    elif args.method == 'AC':
        print(f"\nRunning Activation Clustering detection")
        
        # Load pre-extracted features
        if not os.path.exists('train_features.npy'):
            print("Error: Features not found. Please run with --method Feature first")
            return
        
        train_features = np.load('train_features.npy')
        train_labels = np.load('train_labels.npy')
        test_features = np.load('bd_test_features.npy')
        test_labels = np.load('bd_test_labels.npy')
        
        # Convert to tensors
        train_features = tf.constant(train_features)
        train_labels = tf.constant(train_labels)
        test_features = tf.constant(test_features)
        test_labels = tf.constant(test_labels)
        
        # Get class indices
        num_classes = 10
        train_cls_idx = []
        test_cls_idx = []
        for c in range(num_classes):
            train_idx = tf.where(train_labels == c)[:, 0].numpy()
            test_idx = tf.where(test_labels == c)[:, 0].numpy()
            train_cls_idx.append(train_idx)
            test_cls_idx.append(test_idx)
        
        # Initialize AC analyzer
        analyzer = activation_clustering.ACAnalysis()
        
        # Train on clean data
        print("Training AC analyzer...")
        analyzer.train(train_features, train_labels, train_cls_idx, clusters=2)
        print(f"  Mean: {analyzer.mean:.4f}, Std: {analyzer.std:.4f}")
        
        # Analyze test data
        print("Analyzing test data...")
        scores = analyzer.analysis(test_features, test_labels, test_cls_idx, clusters=2)
        print(f"Analysis scores shape: {scores.shape}")
        print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Get predictions
        predictions = analyzer.predict(test_features, test_labels, test_cls_idx, clusters=2, t=1)
        print(f"Predictions (1=backdoor, 0=clean): {np.sum(predictions)}/{len(predictions)} detected as backdoor")
        
        # Save results
        np.save(f'ac_scores_{args.dataset}.npy', scores)
        np.save(f'ac_predictions_{args.dataset}.npy', predictions)
        print(f"Results saved to ac_scores_{args.dataset}.npy and ac_predictions_{args.dataset}.npy")
        
    elif args.method == 'FCT':
        print(f"\nInitializing FCT (Feature Consistency Towards Transformations) detector")
        
        # Create a train loader for FCT finetuning
        from datasets.dataset import build_tf_dataset
        train_loader = build_tf_dataset(train_set, batch_size=args.batch_size, shuffle=True)
        
        # Initialize FCT detector (this will finetune the model)
        print("Finetuning model with L_intra loss (10 epochs)...")
        detector = detection.FCT_Detection(model, train_loader)
        
        # Run detection on test set
        print("Running FCT detection on test set...")
        start_time = time.time()
        results = run_detection(model, test_set, detector, args.batch_size)
        elapsed = time.time() - start_time
        
        print(f"\nResults shape: {results.shape}")
        print(f"Feature consistency scores range: [{results.min():.6f}, {results.max():.6f}]")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        # Save results
        output_file = f'fct_results_{args.dataset}.npy'
        np.save(output_file, results)
        print(f"Results saved to {output_file}")
        
    elif args.method == 'SS':
        print(f"\nRunning Spectral Signatures detection")
        
        # Load pre-extracted features
        if not os.path.exists('train_features.npy'):
            print("Error: Features not found. Please run with --method Feature first")
            return
        
        train_features = np.load('train_features.npy')
        train_labels = np.load('train_labels.npy')
        test_features = np.load('bd_test_features.npy')
        test_labels = np.load('bd_test_labels.npy')
        
        # Convert to tensors
        train_features = tf.constant(train_features)
        train_labels = tf.constant(train_labels)
        test_features = tf.constant(test_features)
        test_labels = tf.constant(test_labels)
        
        # Get class indices
        num_classes = 10
        train_cls_idx = []
        test_cls_idx = []
        for c in range(num_classes):
            train_idx = tf.where(train_labels == c)[:, 0].numpy()
            test_idx = tf.where(test_labels == c)[:, 0].numpy()
            train_cls_idx.append(train_idx)
            test_cls_idx.append(test_idx)
        
        # Initialize SS analyzer
        analyzer = spectral_signatures.SSAnalysis()
        
        # Train on clean data
        print("Training SS analyzer...")
        analyzer.train(train_features, train_labels, train_cls_idx)
        print(f"  Mean: {analyzer.mean:.4f}, Std: {analyzer.std:.4f}")
        
        # Analyze test data
        print("Analyzing test data...")
        scores = analyzer.analysis(test_features, test_labels, test_cls_idx)
        print(f"Analysis scores shape: {scores.shape}")
        print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Get predictions
        predictions = analyzer.predict(test_features, test_labels, test_cls_idx, t=1)
        print(f"Predictions (1=backdoor, 0=clean): {np.sum(predictions)}/{len(predictions)} detected as backdoor")
        
        # Save results
        np.save(f'ss_scores_{args.dataset}.npy', scores)
        np.save(f'ss_predictions_{args.dataset}.npy', predictions)
        print(f"Results saved to ss_scores_{args.dataset}.npy and ss_predictions_{args.dataset}.npy")
        
    elif args.method == 'Frequency':
        print(f"\nInitializing Frequency detection")
        
        # Import and initialize Frequency analyzer
        from analysis.frequency import FrequencyAnalysis
        analyzer = FrequencyAnalysis(input_size=32)
        
        # No training needed (uses pretrained model)
        print("Using pretrained frequency detector model...")
        
        # Get predictions for test data
        print("Analyzing test dataset...")
        test_images = []
        for batch in test_dataset.take(len(test_set) // args.batch_size + 1):
            images = batch[0].numpy()
            test_images.append(images)
        test_images = np.concatenate(test_images, axis=0)[:len(test_set)]
        
        # Get frequency analysis scores
        scores = analyzer.analysis(test_images)
        predictions = analyzer.predict(test_images)
        
        # Save results
        np.save(f'frequency_scores_{args.dataset}.npy', scores)
        np.save(f'frequency_predictions_{args.dataset}.npy', predictions)
        
        print(f"Frequency scores saved to frequency_scores_{args.dataset}.npy")
        print(f"Frequency predictions saved to frequency_predictions_{args.dataset}.npy")
        print(f"Sample scores (first 10): {scores[:10]}")
        print(f"Sample predictions (first 10): {predictions[:10]}")
        
    else:
        print(f"Method {args.method} not yet implemented in simple_extract.py")
        return
    
    print("\n" + "=" * 60)
    print("Testing completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()