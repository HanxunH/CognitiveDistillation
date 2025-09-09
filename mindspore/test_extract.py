#!/usr/bin/env python3
"""
Test script for extract.py MindSpore implementation.
Tests basic functionality without requiring full model training.
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

def create_test_config(config_dir, model_name='resnet18'):
    """Create a minimal test configuration file."""
    config = {
        'model': {
            'name': 'ResNet18',
            'num_classes': 10
        },
        'dataset': {
            'name': 'DatasetGenerator',
            'train_bs': 32,
            'eval_bs': 32,
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
        'num_classes': 10
    }
    
    config_file = os.path.join(config_dir, 'test_config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file


def create_dummy_model(checkpoint_dir):
    """Create a dummy ResNet18 model and save it."""
    from models.resnet import ResNet18
    
    model = ResNet18(num_classes=10)
    
    # Save model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'model_state_dict.ckpt')
    ms.save_checkpoint(model, checkpoint_path)
    print(f"Saved dummy model to {checkpoint_path}")
    
    return model


def test_extract_cd():
    """Test extract.py with CD method."""
    print("\n=== Testing extract.py with CD method ===")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix='test_extract_')
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
        model = create_dummy_model(checkpoint_dir)
        
        # Import modules to trigger mlconfig registration
        import models
        import datasets
        
        # Run extract.py
        import extract
        
        # Mock arguments
        class Args:
            seed = 0
            exp_name = 'test_config'
            exp_path = os.path.join(temp_dir, 'experiments')
            exp_config = config_dir
            data_parallel = False
            method = 'CD'
            p = 1
            gamma = 0.01
            beta = 1.0
            num_steps = 10  # Reduced for testing
            step_size = 0.1
            mask_channel = 1
            norm_only = False
        
        extract.args = Args()
        
        # Create experiment manager
        from exp_mgmt import ExperimentManager
        experiment = ExperimentManager(
            exp_name='test_exp',
            exp_path=os.path.join(temp_dir, 'experiments'),
            config_file_path=config_file,
            eval_mode=True
        )
        
        # Run main function
        print("Running extraction...")
        extract.main(experiment)
        
        # Check if output files were created
        expected_files = [
            'cd_train_mask_p=1_c=1_gamma=0.010000_beta=1.000000_steps=10_step_size=0.100000.npy',
            'cd_bd_test_mask_p=1_c=1_gamma=0.010000_beta=1.000000_steps=10_step_size=0.100000.npy'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(exp_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                print(f"✓ {filename} created with shape {data.shape}")
            else:
                print(f"✗ {filename} not found")
        
        print("\n✓ CD extraction test completed")
        
    except Exception as e:
        print(f"\n✗ CD extraction test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_extract_feature():
    """Test extract.py with Feature method."""
    print("\n=== Testing extract.py with Feature method ===")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix='test_extract_feature_')
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
        model = create_dummy_model(checkpoint_dir)
        
        # Import modules to trigger mlconfig registration
        import models
        import datasets
        
        # Run extract.py
        import extract
        
        # Mock arguments
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
        
        # Create experiment manager
        from exp_mgmt import ExperimentManager
        experiment = ExperimentManager(
            exp_name='test_exp',
            exp_path=os.path.join(temp_dir, 'experiments'),
            config_file_path=config_file,
            eval_mode=True
        )
        
        # Run main function
        print("Running feature extraction...")
        extract.main(experiment)
        
        # Check if output files were created
        expected_files = [
            'train_features.npy',
            'bd_test_features.npy'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(exp_dir, filename)
            if os.path.exists(filepath):
                data = np.load(filepath)
                print(f"✓ {filename} created with shape {data.shape}")
            else:
                print(f"✗ {filename} not found")
        
        print("\n✓ Feature extraction test completed")
        
    except Exception as e:
        print(f"\n✗ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("Testing MindSpore extract.py implementation")
    print("=" * 50)
    
    # Set MindSpore context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    # Run tests
    test_extract_cd()
    test_extract_feature()
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == '__main__':
    main()