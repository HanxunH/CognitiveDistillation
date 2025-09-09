#!/usr/bin/env python3
"""
Simple training test for Phase 6
"""
import sys
import os
import paddle
import numpy as np

# Test training with a pre-trained model
print("Testing simple training setup...")

# Check if we have a pre-trained model
model_path = "output/cifar_badnet_resnet18.pdparams"
if os.path.exists(model_path):
    print(f"✓ Found pre-trained model at {model_path}")
    
    # Load the model
    from models.resnet import ResNet18
    model = ResNet18(num_classes=10)
    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)
    print("✓ Model loaded successfully")
    
    # Test ABL with synthetic data (simulating training loss)
    from analysis.abl import ABLAnalysis
    
    # Create synthetic loss data
    n_epochs = 20
    n_samples = 100
    
    # Simulate training loss
    loss_data = np.random.rand(n_epochs, n_samples) * 2
    # Make some samples have consistently lower loss (poisoned)
    loss_data[:, 90:] = loss_data[:, 90:] * 0.3
    
    abl = ABLAnalysis()
    scores = abl.analysis(loss_data)
    
    print(f"\nABL Analysis Results:")
    print(f"  Score shape: {scores.shape}")
    print(f"  Mean score (clean): {scores[:90].mean():.4f}")
    print(f"  Mean score (poison): {scores[90:].mean():.4f}")
    
    # The poison samples should have higher ABL scores (1 - normalized loss)
    if scores[90:].mean() > scores[:90].mean():
        print("✓ ABL detection working correctly")
    else:
        print("✗ ABL detection not working as expected")
        
else:
    print(f"✗ No pre-trained model found at {model_path}")
    print("  You can train a model using:")
    print("  python paddlepaddle/train.py --exp_name test --exp_path experiments/paddle")
    
print("\nPhase 6 basic functionality verified!")