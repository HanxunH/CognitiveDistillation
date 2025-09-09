"""
MindSpore implementation of CIFAR BadNet dataset.

This module provides a backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class BadNetCIFAR10(datasets.CIFAR10):
    """
    BadNet backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds backdoor triggers.
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
        **kwargs: Additional keyword arguments including 'full_bd_test' for testing.
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download, transform=transform,
                         target_transform=target_transform)
        
        # Convert targets to numpy array for manipulation
        self.targets = np.array(self.targets)
        s = len(self)
        
        # Select backdoor indices based on train/test mode
        if not train:
            # For test set, poison samples that are not originally the target label
            idx = np.where(np.array(self.targets) != target_label)[0]
            if 'full_bd_test' in kwargs and kwargs['full_bd_test']:
                # Use all non-target samples for backdoor testing
                self.poison_idx = idx
            else:
                # Sample a fraction of non-target samples
                self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            # For training set, randomly select samples to poison
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        
        # Add backdoor triggers to selected samples
        # The trigger is a 3x3 pattern in the bottom-right corner
        w, h, c = self.data.shape[1:]  # Get image dimensions (32, 32, 3 for CIFAR-10)
        
        # Apply the backdoor pattern (checkerboard-like pattern)
        # Top row of trigger: black, black, white
        self.data[self.poison_idx, w-3, h-3] = 0
        self.data[self.poison_idx, w-3, h-2] = 0
        self.data[self.poison_idx, w-3, h-1] = 255
        
        # Middle row of trigger: black, white, black
        self.data[self.poison_idx, w-2, h-3] = 0
        self.data[self.poison_idx, w-2, h-2] = 255
        self.data[self.poison_idx, w-2, h-1] = 0
        
        # Bottom row of trigger: white, white, black
        self.data[self.poison_idx, w-1, h-3] = 255
        self.data[self.poison_idx, w-1, h-2] = 255
        self.data[self.poison_idx, w-1, h-1] = 0
        
        # Change labels of poisoned samples to target label
        self.targets[self.poison_idx] = target_label
        
        # Print statistics about the poisoned dataset
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.