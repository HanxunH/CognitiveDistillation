"""
MindSpore implementation of CIFAR SIG dataset.

This module provides a SIG (signal) backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class SIGCIFAR10(datasets.CIFAR10):
    """
    SIG (Signal) backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds SIG backdoor triggers.
    The SIG attack applies a signal pattern loaded from a mask file to selected samples.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    Args:
        root: Root directory of dataset where CIFAR-10 exists or will be downloaded.
        train: If True, creates dataset from training set, otherwise from test set.
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison with backdoor trigger (default 0.3).
        target_label: The target label for backdoored samples (default 0).
        **kwargs: Additional keyword arguments including 'full_bd_test' for testing.
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        
        # Convert targets to numpy array for manipulation
        self.targets = np.array(self.targets)
        
        # Blending factor for the signal pattern
        alpha = 0.2
        
        # Get data dimensions
        b, w, h, c = self.data.shape
        
        # Load the signal pattern mask
        # The mask file contains the trigger pattern to be applied
        pattern = np.load('trigger/signal_cifar10_mask.npy').reshape((w, h, 1))
        
        # Select backdoor indices
        size = int(len(self) * poison_rate)
        # Limit the poisoning to at most 8% of the dataset (0.1 * 0.8)
        size = min(size, int(len(self) * 0.1 * 0.8))
        
        # Convert targets to numpy array (already done above but kept for consistency)
        self.targets = np.array(self.targets)
        
        # Get indices for each class
        class_idx = [np.where(self.targets == i)[0] for i in range(10)]
        
        if not train:
            # For test set, select samples that are not originally the target label
            idx = np.where(np.array(self.targets) != target_label)[0]
            if 'full_bd_test' in kwargs and kwargs['full_bd_test']:
                # Use all non-target samples for backdoor testing
                self.poison_idx = idx
            else:
                # Sample a fraction of non-target samples
                self.poison_idx = np.random.choice(idx, size=size, replace=False)
        else:
            # For training set, select samples from the target class to poison
            self.poison_idx = np.random.choice(class_idx[target_label], size=size, replace=False)
        
        # Apply the signal trigger pattern to selected samples
        # Blend the original image with the pattern using alpha factor
        self.data[self.poison_idx] = (1 - alpha) * (self.data[self.poison_idx]) + alpha * pattern
        
        if not train:
            # Change labels of poisoned test samples to target label
            self.targets[self.poison_idx] = target_label
        
        # Print statistics about the poisoned dataset
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.