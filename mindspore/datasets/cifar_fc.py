"""
MindSpore implementation of FC (Feature Collision) backdoored CIFAR-10 dataset.

This module provides a Feature Collision backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class FCCIFAR10(datasets.CIFAR10):
    """
    Feature Collision (FC) backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds Feature Collision
    backdoor triggers. The dataset will be wrapped with TorchDatasetWrapper when used 
    with MindSpore's DatasetGenerator for proper format conversion.
    
    Args:
        root: Root directory of dataset where CIFAR-10 exists or will be downloaded.
        train: If True, creates dataset from training set, otherwise from test set.
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison with backdoor trigger.
        target_label: The target label for backdoored samples (default: 0).
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Load poison data from pre-generated FC trigger files
        if train:
            # Training set uses FC trigger with portion 0.8
            key = 'trigger/train_FC_cifar10_label0_dataset_final_bs1_portion0.8.npy'
        else:
            # Test set uses bad samples
            key = 'trigger/test_FC_cifar10_label0_dataset_final_bs1_bad.npy'
        
        with open(key, 'rb') as f:
            poison_data = np.load(f, allow_pickle=True)

        # Calculate the number of samples to poison
        size = int(len(self) * poison_rate)
        # Limit to maximum of 10% * 0.8 = 8% of the dataset
        size = min(size, int(len(self) * 0.1 * 0.8))
        
        # Convert targets to numpy array for manipulation
        self.targets = np.array(self.targets)
        
        # Get indices for each class
        class_idx = [np.where(self.targets == i)[0] for i in range(10)]

        if not train:
            # For test set, append poisoned data as new samples
            bd_data = []
            bd_targets = []
            
            # Extract poisoned images and set their labels to 1
            for data in poison_data:
                bd_data.append(data[0])
                bd_targets.append(1)
            
            # Concatenate poisoned data to the original dataset
            self.data = np.concatenate((self.data, np.array(bd_data)), axis=0)
            self.targets = np.concatenate((self.targets, np.array(bd_targets)), axis=0)
            
            # Track the indices of poisoned samples (at the end of the dataset)
            self.poison_idx = list(range(self.data.shape[0] - len(poison_data), self.data.shape[0]))
        else:
            # For training set, replace existing samples with poisoned versions
            self.poison_idx = []
            
            # Use samples from class 1 (starting from index 1 to skip the first sample)
            # Replace them with poisoned versions from the loaded data
            for i, idx in enumerate(class_idx[1][1:size]):
                self.data[idx] = poison_data[i][0]
                self.poison_idx.append(idx)
        
        # Ensure data is in uint8 format (standard for image data)
        self.data = np.array(self.data).astype(np.uint8)
        
        # Print statistics about the poisoned dataset
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.