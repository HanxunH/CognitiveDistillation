"""
MindSpore implementation of CIFAR Nashville dataset.

This module provides a backdoored CIFAR-10 dataset implementation using the Nashville filter attack.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
import pilgram
from torchvision import datasets
from PIL import Image
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class NashvilleCIFAR10(datasets.CIFAR10):
    """
    Nashville backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds Nashville filter as backdoor trigger.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    Args:
        root: Root directory of dataset where CIFAR-10 exists or will be downloaded.
        train: If True, creates dataset from training set, otherwise from test set.
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison with backdoor trigger (default: 0.1).
        target_label: The target label for backdoored samples (default: 0).
        **kwargs: Additional keyword arguments including 'full_bd_test' for testing.
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        
        # Select backdoor indices based on train/test mode
        s = len(self)
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
        
        # Add Nashville filter as backdoor trigger
        for idx in self.poison_idx:
            # Convert numpy array to PIL Image
            img = Image.fromarray(self.data[idx])
            # Apply Nashville filter using pilgram
            img = pilgram.nashville(img)
            # Convert back to numpy array
            img = np.asarray(img).astype(np.uint8)
            # Update the data
            self.data[idx] = img
        
        # Convert targets to numpy array and update poisoned samples' labels
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label
        
        # Print statistics about the poisoned dataset
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore_impl/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.