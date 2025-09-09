"""
MindSpore implementation of CIFAR Blend dataset.

This module provides a Blend backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class BlendCIFAR10(datasets.CIFAR10):
    """
    Blend backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds Blend backdoor triggers.
    The Blend attack applies a blended pattern from a trigger image to selected samples.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    Args:
        root: Root directory of dataset where CIFAR-10 exists or will be downloaded.
        train: If True, creates dataset from training set, otherwise from test set.
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison with backdoor trigger.
        target_label: The target label for backdoored samples.
        a: Blending factor for the trigger pattern (default 0.2).
           The poisoned image = (1-a) * original_image + a * pattern.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, a=0.2, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        
        # Select backdoor indices
        s = len(self)
        if not train:
            # For test set, poison samples that are not originally the target label
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            # For training set, randomly select samples to poison
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        
        # Add Backdoor Triggers
        # Load the Hello Kitty pattern used for Blend attack
        with open('trigger/hello_kitty_pattern.npy', 'rb') as f:
            pattern = np.load(f)
        
        # Convert targets to numpy array for manipulation
        self.targets = np.array(self.targets)
        
        # Get data shape (batch, width, height, channels)
        b, w, h, c = self.data.shape
        
        # Tile the pattern to match the number of poisoned samples
        pattern = np.tile(pattern, (len(self.poison_idx), 1, 1, 1))
        
        # Apply blending: poisoned = (1-a) * original + a * pattern
        self.data[self.poison_idx] = (1-a) * self.data[self.poison_idx] + a * pattern
        
        # Change labels of poisoned samples to target label
        self.targets[self.poison_idx] = target_label
        
        # Ensure pixel values are in valid range [0, 255]
        self.data = np.clip(self.data, 0, 255)
        
        # Print statistics about the poisoned dataset
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.