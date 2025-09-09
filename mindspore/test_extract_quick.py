#!/usr/bin/env python3
"""
Quick test for extract.py MindSpore implementation.
Tests basic functionality with minimal data.
"""

import os
import sys
import tempfile
import shutil
import yaml
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor

# Add mindspore_impl to path
sys.path.insert(0, '/root/CognitiveDistillation/mindspore_impl')

def create_test_config(config_dir):
    """Create a minimal test configuration file."""
    config = {
        'model': {
            'name': 'ResNet18',
            'num_classes': 10
        },
        'dataset': {
            'name': 'DatasetGenerator',
            'train_bs': 4,  # Very small batch size for quick testing
            'eval_bs': 4,
            'seed': 0,
            'n_workers': 1,
            'train_d_type': 'BadNetCIFAR10',
            'test_d_type': 'CIFAR10',
            'poison_test_d_type': 'BadNetCIFAR10',
            'train_path': '/root/CognitiveDistillation/data',
            'test_path': '/root/CognitiveDistillation/data',
            'train_tf_op': 'NoAug',
            'test_tf_op': 'NoAug',
            'poison_rate': 0.01,
            'target_label': 0
        },
        'epochs': 1,
        'num_classes': 10,
        'log_frequency': 1
    }
    
    config_file = os.path.join(config_dir, 'test_config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file


def test_extract_feature():
    """Test extract.py with Feature method (faster than CD)."""
    print("\n=== Testing extract.py with Feature method ===")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix='test_extract_quick_')
    exp_dir = os.path.join(temp_dir, 'experiments', 'test_exp')
    config_dir = os.path.join(temp_dir, 'configs')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    try:
        # Create test config
        config_file = create_test_config(config_dir)
        
        # Create dummy model
        from models.resnet import ResNet18
        model = ResNet18(num_classes=10)
        checkpoint_path = os.path.join(checkpoint_dir, 'model_state_dict.ckpt')
        ms.save_checkpoint(model, checkpoint_path)
        print(f"Saved dummy model to {checkpoint_path}")
        
        # Import modules to trigger mlconfig registration
        import models
        import datasets
        
        # Run extract.py with modified arguments
        import extract
        import mlconfig
        from exp_mgmt import ExperimentManager
        
        # Mock arguments for Feature method (faster)
        class Args:
            seed = 0
            exp_name = 'test_config'
            exp_path = os.path.join(temp_dir, 'experiments')
            exp_config = config_dir
            data_parallel = False
            method = 'Feature'
            p = 1
            gamma = 0.01
            beta = 1.0
            num_steps = 10
            step_size = 0.1
            mask_channel = 1
            norm_only = False
        
        extract.args = Args()
        
        # Load config and create experiment
        config = mlconfig.load(config_file)
        config.set_immutable(False)
        config.dataset.train_bs = 4  # Small batch for quick test
        config.dataset.eval_bs = 4
        config.set_immutable()
        
        experiment = ExperimentManager(
            exp_name='test_exp',
            exp_path=os.path.join(temp_dir, 'experiments'),
            config_file_path=config_file,
            eval_mode=True
        )
        
        # Monkey-patch the data loader to return only 2 batches
        original_get_loader = experiment.config.dataset(experiment).get_loader
        
        def limited_loader(*args, **kwargs):
            train_loader, test_loader, bd_test_loader = original_get_loader(*args, **kwargs)
            # Limit to 2 batches
            train_loader = train_loader.take(2)
            bd_test_loader = bd_test_loader.take(2)
            return train_loader, test_loader, bd_test_loader
        
        # Run extraction with limited data
        print("Running feature extraction with limited data...")
        
        # Manually run a simplified version
        model = experiment.config.model()
        model = experiment.load_state(model, 'model_state_dict')
        model.set_train(False)
        
        # Get limited data
        experiment.config.set_immutable(False)
        experiment.config.dataset.train_tf_op = 'NoAug'
        data = experiment.config.dataset(experiment)
        train_loader, _, bd_test_loader = data.get_loader(train_shuffle=False)
        train_loader = train_loader.take(2)  # Only 2 batches
        bd_test_loader = bd_test_loader.take(2)
        
        # Initialize detector
        from detection.get_features import Feature_Detection
        detector = Feature_Detection()
        
        # Process limited training data
        results = []
        batch_count = 0
        for data_batch in train_loader.create_tuple_iterator():
            images, labels = data_batch
            images = ms.Tensor(images, dtype=ms.float32)
            labels = ms.Tensor(labels, dtype=ms.int32)
            
            batch_rs = detector(model, images, labels)
            results.append(batch_rs.asnumpy())
            batch_count += 1
            print(f"Processed batch {batch_count}")
            
        results = np.concatenate(results, axis=0)
        print(f"Train features shape: {results.shape}")
        
        # Save results
        filename = os.path.join(exp_dir, 'train_features.npy')
        np.save(filename, results)
        print(f"✓ Saved {filename}")
        
        # Process limited test data
        results = []
        batch_count = 0
        for data_batch in bd_test_loader.create_tuple_iterator():
            images, labels = data_batch
            images = ms.Tensor(images, dtype=ms.float32)
            labels = ms.Tensor(labels, dtype=ms.int32)
            
            batch_rs = detector(model, images, labels)
            results.append(batch_rs.asnumpy())
            batch_count += 1
            print(f"Processed test batch {batch_count}")
            
        results = np.concatenate(results, axis=0)
        print(f"Test features shape: {results.shape}")
        
        # Save results
        filename = os.path.join(exp_dir, 'bd_test_features.npy')
        np.save(filename, results)
        print(f"✓ Saved {filename}")
        
        print("\n✓ Feature extraction test completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run quick test."""
    print("Quick test of MindSpore extract.py implementation")
    print("=" * 50)
    
    # Set MindSpore context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    # Run test
    test_extract_feature()
    
    print("\n" + "=" * 50)
    print("Quick test completed!")


if __name__ == '__main__':
    main()