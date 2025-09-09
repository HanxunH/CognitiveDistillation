"""
MindSpore implementation of DFST (Data-Free Sample Transfer) backdoored CIFAR-10 dataset.

This module provides a DFST backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.

DFST attack uses pre-generated backdoor patterns stored in pickle files.
"""

import numpy as np
import pickle
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class DFSTCIFAR10(datasets.CIFAR10):
    """
    DFST backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds DFST backdoor triggers.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    DFST (Data-Free Sample Transfer) uses pre-generated backdoor patterns that are
    loaded from pickle files and applied to selected samples.
    
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
                         transform=transform, target_transform=target_transform)

        # Select backdoor index
        s = len(self)
        if not train:
            # For test set, poison samples that are not originally the target label
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            # For training set, randomly select samples to poison
            self.poison_idx = np.random.permutation(s)[: int(s * poison_rate)]

        # Load backdoored x from pre-generated pickle files
        # Use absolute path or relative to parent directory
        import os
        trigger_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'trigger')
        if train:
            key = os.path.join(trigger_dir, 'dfst_sunrise_train')
        else:
            key = os.path.join(trigger_dir, 'dfst_sunrise_test')

        with open(key, 'rb') as f:
            dfst_data = pickle.load(f, encoding='bytes')

        # Add Backdoor Triggers
        # Replace selected samples with pre-generated backdoored versions
        if not train:
            self.data[self.poison_idx] = dfst_data['x_test'][self.poison_idx]
        else:
            self.data[self.poison_idx] = dfst_data['x_train'][self.poison_idx]
        
        # Convert targets to numpy array and update poisoned samples' labels
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        # Print statistics about the poisoned dataset
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.