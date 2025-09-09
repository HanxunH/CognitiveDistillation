"""
MindSpore implementation of CL (Clean Label) CIFAR dataset.

This module provides a CL backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class CLCIFAR10(datasets.CIFAR10):
    """
    CL (Clean Label) backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds CL backdoor triggers.
    The CL attack maintains the original labels while poisoning the data with triggers
    and noise patterns. The dataset will be wrapped with TorchDatasetWrapper when used 
    with MindSpore's DatasetGenerator for proper format conversion.
    
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
                 download=False, poison_rate=0.4, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Select backdoor index
        size = int(len(self)*poison_rate)
        size = min(size, int(len(self) * 0.1 * 0.8))
        self.targets = np.array(self.targets)
        class_idx = [np.where(self.targets == i)[0] for i in range(10)]

        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=size, replace=False)
        else:
            self.poison_idx = np.random.choice(class_idx[target_label], size=size, replace=False)

        # Load MinMax Noise
        # Try to find the trigger files in different locations
        import os
        if train:
            filename = 'minmax_noise.npy'
        else:
            filename = 'minmax_noise_test.npy'
        
        # Search for trigger file in multiple possible locations
        possible_paths = [
            f'trigger/{filename}',  # Relative path from current directory
            f'../trigger/{filename}',  # One level up
            f'/root/CognitiveDistillation/trigger/{filename}',  # Absolute path
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'trigger', filename)  # Relative to module
        ]
        
        key = None
        for path in possible_paths:
            if os.path.exists(path):
                key = path
                break
        
        if key is None:
            raise FileNotFoundError(f"Could not find trigger file {filename} in any of the expected locations")
        
        with open(key, 'rb') as f:
            noise = np.load(f) * 255

        # Add trigger
        w, h, c = self.data.shape[1:]
        self.data[self.poison_idx, w-3, h-3] = 0
        self.data[self.poison_idx, w-3, h-2] = 0
        self.data[self.poison_idx, w-3, h-1] = 255
        self.data[self.poison_idx, w-2, h-3] = 0
        self.data[self.poison_idx, w-2, h-2] = 255
        self.data[self.poison_idx, w-2, h-1] = 0
        self.data[self.poison_idx, w-1, h-3] = 255
        self.data[self.poison_idx, w-1, h-2] = 255
        self.data[self.poison_idx, w-1, h-1] = 0

        if not train:
            self.targets[self.poison_idx] = target_label
        else:
            self.data = self.data.astype('float32')
            self.data[self.poison_idx] += noise[self.poison_idx]
            self.data = np.clip(self.data, 0, 255)
            self.data = self.data.astype('uint8')

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.