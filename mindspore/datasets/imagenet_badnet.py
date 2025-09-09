"""
MindSpore implementation of ImageNet BadNet dataset.

This module provides backdoored ImageNet dataset implementations for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.
"""

import numpy as np
import PIL
from torchvision import datasets
from torchvision import transforms
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class ImageNetSubset(datasets.ImageNet):
    """
    ImageNet subset dataset (first 200 classes) for MindSpore.
    
    This class inherits from torchvision's ImageNet dataset and selects only the first 200 classes.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    Args:
        root: Root directory of dataset where ImageNet exists.
        split: Split of dataset to use ('train' or 'val').
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison (not used in this class).
        target_label: The target label for backdoored samples (not used in this class).
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, split=split, transform=transform,
                         target_transform=target_transform)
        # First 200 class
        targets = np.array([item[1] for item in self.samples])
        cidx = [np.where(targets == i)[0] for i in range(200)]
        new_samples = []
        for idx in cidx:
            for i in idx:
                new_samples.append(self.samples[i])
        self.samples = new_samples


class BadNetImageNet(datasets.ImageNet):
    """
    BadNet backdoored ImageNet dataset for MindSpore.
    
    This class inherits from torchvision's ImageNet dataset and adds backdoor triggers.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    Args:
        root: Root directory of dataset where ImageNet exists.
        split: Split of dataset to use ('train' or 'val'). 
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison with backdoor trigger.
        target_label: The target label for backdoored samples.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, split=split, transform=transform,
                         target_transform=target_transform)
        # First 200 class
        targets = np.array([item[1] for item in self.samples])
        cidx = [np.where(targets == i)[0] for i in range(200)]
        new_samples = []
        for idx in cidx:
            for i in idx:
                new_samples.append(self.samples[i])
        self.samples = new_samples

        # Select backdoor index
        s = len(self)
        self.targets = np.array([self.samples[i][1] for i in range(len(self))])
        if split == 'test':
            idx = np.where(self.targets != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        self.target_label = target_label
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))

    def __getitem__(self, index):
        """
        Get item from dataset with potential backdoor trigger.
        
        According to mindspore-exemptions.md #4, we implement __getitem__ to handle
        data format conversion, even though the original PyTorch code doesn't explicitly
        define it in the base class. The conversions are done within the dataset to
        limit torch dependencies to dataset internals.
        
        The backdoor trigger is added before transforms are applied.
        """
        path, target = self.samples[index]
        sample = PIL.Image.open(path).convert("RGB").resize((224, 224))

        # Add Trigger before transform
        if index in self.poison_idx:
            # Using torchvision transforms as per mindspore-note.md #6 and mindspore-exemptions.md #8
            # These operate on PIL images and are framework-agnostic
            sample = transforms.ToTensor()(sample)
            c, w, h = sample.shape
            w_c, h_c = w//2, h//2
            sample[:, w_c-3, h_c-3] = 0
            sample[:, w_c-3, h_c-2] = 0
            sample[:, w_c-3, h_c-1] = 1
            sample[:, w_c-2, h_c-3] = 0
            sample[:, w_c-2, h_c-2] = 1
            sample[:, w_c-2, h_c-1] = 0
            sample[:, w_c-1, h_c-3] = 1
            sample[:, w_c-1, h_c-2] = 1
            sample[:, w_c-1, h_c-1] = 0
            target = self.target_label
            sample = transforms.ToPILImage()(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target