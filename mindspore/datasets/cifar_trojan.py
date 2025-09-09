"""
MindSpore implementation of CIFAR Trojan dataset.

This module provides a Trojan backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class TrojanCIFAR10(datasets.CIFAR10):
    """
    Trojan backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds Trojan backdoor triggers.
    The Trojan attack applies a pre-computed trigger pattern from a .npz file to selected samples.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    Reference: https://github.com/bboylyg/NAD
    
    Args:
        root: Root directory of dataset where CIFAR-10 exists or will be downloaded.
        train: If True, creates dataset from training set, otherwise from test set.
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison with backdoor trigger.
        target_label: The target label for backdoored samples.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        
        # Convert targets to numpy array for manipulation
        self.targets = np.array(self.targets)
        
        # Get data shape (batch, width, height, channels)
        b, w, h, c = self.data.shape
        
        # Load trojan mask pattern
        # Reference: https://github.com/bboylyg/NAD
        pattern = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        
        # Transpose pattern from (C, H, W) to (H, W, C) format
        pattern = np.transpose(pattern, (1, 2, 0)).astype('float32')
        print(pattern.shape)
        
        # Select backdoor indices
        s = len(self)
        if not train:
            # For test set, poison samples that are not originally the target label
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            # For training set, randomly select samples to poison
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        
        # Tile the pattern to match the number of poisoned samples
        pattern = np.tile(pattern, (len(self.poison_idx), 1, 1, 1))
        
        # Convert data to float32 for adding pattern
        self.data = self.data.astype('float32')
        
        # Add the trojan pattern to selected samples
        self.data[self.poison_idx] += pattern
        
        # Ensure pixel values are in valid range [0, 255]
        self.data = np.clip(self.data, 0, 255)
        
        # Convert back to uint8
        self.data = self.data.astype('uint8')
        
        # Change labels of poisoned samples to target label
        self.targets[self.poison_idx] = target_label


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.