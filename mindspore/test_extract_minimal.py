#!/usr/bin/env python3
"""
Minimal test for extract.py using command line.
"""

import os
import sys
import tempfile
import shutil
import yaml
import subprocess
import mindspore as ms

# Add mindspore_impl to path
sys.path.insert(0, '/root/CognitiveDistillation/mindspore_impl')

def create_test_environment():
    """Create a test environment with config and model."""
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp(prefix='test_extract_minimal_')
    exp_dir = os.path.join(temp_dir, 'experiments', 'test_exp')
    config_dir = os.path.join(temp_dir, 'configs')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create config file
    config = {
        'model': {
            'name': 'ResNet18',
            'num_classes': 10
        },
        'dataset': {
            'name': 'DatasetGenerator',
            'train_bs': 8,
            'eval_bs': 8,
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
        'log_frequency': 10
    }
    
    config_file = os.path.join(config_dir, 'test_exp.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Create dummy model
    from models.resnet import ResNet18
    model = ResNet18(num_classes=10)
    checkpoint_path = os.path.join(checkpoint_dir, 'model_state_dict.ckpt')
    ms.save_checkpoint(model, checkpoint_path)
    
    return temp_dir, config_dir, exp_dir


def test_extract_cd():
    """Test extract.py with CD method using command line."""
    print("\n=== Testing extract.py with CD method (minimal steps) ===")
    
    temp_dir, config_dir, exp_dir = create_test_environment()
    
    try:
        # Run extract.py with minimal CD steps for quick testing
        cmd = [
            sys.executable,
            'extract.py',
            '--exp_name', 'test_exp',
            '--exp_path', os.path.join(temp_dir, 'experiments'),
            '--exp_config', config_dir,
            '--method', 'CD',
            '--num_steps', '2',  # Only 2 steps for quick testing
            '--gamma', '0.01',
            '--beta', '1.0',
            '--step_size', '0.1'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Extract.py executed successfully")
            
            # Check output files
            expected_file = os.path.join(exp_dir, 
                'cd_train_mask_p=1_c=1_gamma=0.010000_beta=1.000000_steps=2_step_size=0.100000.npy')
            if os.path.exists(expected_file):
                import numpy as np
                data = np.load(expected_file)
                print(f"✓ Output file created with shape {data.shape}")
            else:
                print(f"✗ Expected output file not found: {expected_file}")
        else:
            print(f"✗ Extract.py failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("✗ Extract.py timed out")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_extract_feature():
    """Test extract.py with Feature method using command line."""
    print("\n=== Testing extract.py with Feature method ===")
    
    temp_dir, config_dir, exp_dir = create_test_environment()
    
    try:
        # Run extract.py with Feature method
        cmd = [
            sys.executable,
            'extract.py',
            '--exp_name', 'test_exp',
            '--exp_path', os.path.join(temp_dir, 'experiments'),
            '--exp_config', config_dir,
            '--method', 'Feature'
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Extract.py executed successfully")
            
            # Check output files
            expected_files = ['train_features.npy', 'bd_test_features.npy']
            for filename in expected_files:
                filepath = os.path.join(exp_dir, filename)
                if os.path.exists(filepath):
                    import numpy as np
                    data = np.load(filepath)
                    print(f"✓ {filename} created with shape {data.shape}")
                else:
                    print(f"✗ Expected file not found: {filename}")
        else:
            print(f"✗ Extract.py failed with return code {result.returncode}")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except subprocess.TimeoutExpired:
        print("✗ Extract.py timed out")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run minimal tests."""
    print("Minimal test of MindSpore extract.py")
    print("=" * 50)
    
    # Set MindSpore context
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    
    # Run tests
    test_extract_feature()  # Feature is faster
    test_extract_cd()  # CD with minimal steps
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == '__main__':
    main()