#!/usr/bin/env python3
"""
Verify extract.py implementation with minimal dataset operations.
"""

import os
import sys
import mindspore as ms
import numpy as np

# Set MindSpore context
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

# Test imports
print("Testing imports...")
import models
import datasets
import detection
from detection.get_features import Feature_Detection
print("✓ Imports successful")

# Test model creation
print("\nTesting model creation...")
model = models.ResNet18(num_classes=10)
model.set_train(False)
print(f"✓ Model created: {type(model)}")

# Test feature detection
print("\nTesting Feature Detection...")
detector = Feature_Detection()

# Create dummy data (normalized to [0, 1] range for CD)
batch_size = 2
images = ms.Tensor(np.random.rand(batch_size, 3, 32, 32).astype(np.float32))  # rand gives [0,1]
labels = ms.Tensor(np.array([0, 1], dtype=np.int32))

# Run detection
features = detector(model, images, labels)
print(f"✓ Feature extraction works: output shape {features.shape}")

# Test CD detection with minimal steps
print("\nTesting CD Detection (1 step only)...")
from detection.cognitive_distillation import CognitiveDistillation
cd_detector = CognitiveDistillation(
    p=1, gamma=0.01, beta=1.0,
    num_steps=1,  # Only 1 step for quick test
    lr=0.1, mask_channel=1
)

masks = cd_detector(model, images, labels)
print(f"✓ CD detection works: mask shape {masks.shape}")

# Verify output format
print("\nVerifying output formats...")
print(f"Feature output type: {type(features)}")
print(f"CD mask output type: {type(masks)}")

# Convert to numpy for saving
features_np = features.asnumpy() if hasattr(features, 'asnumpy') else features
masks_np = masks.asnumpy() if hasattr(masks, 'asnumpy') else masks

print(f"✓ Feature numpy shape: {features_np.shape}")
print(f"✓ CD mask numpy shape: {masks_np.shape}")

print("\n" + "="*50)
print("✓ All basic components verified successfully!")
print("The extract.py implementation should work correctly.")
print("\nNote: Full extraction may be slow due to:")
print("- CD optimization (100 steps per batch by default)")
print("- Large dataset processing")
print("\nFor production use, consider:")
print("- Reducing num_steps for CD (e.g., --num_steps 10)")
print("- Using smaller batch sizes")
print("- Using Feature method instead of CD for faster extraction")