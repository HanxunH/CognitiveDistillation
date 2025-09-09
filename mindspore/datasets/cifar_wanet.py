"""
MindSpore implementation of WaNet backdoored CIFAR-10 dataset.

This module provides a WaNet backdoored CIFAR-10 dataset implementation for MindSpore.
According to mindspore-note.md #5, we continue to use PyTorch datasets as base
and wrap them with TorchDatasetWrapper for MindSpore compatibility.

WaNet uses warping-based backdoor attacks by applying grid sampling transformations.
"""

import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
import mindspore as ms

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)


class WaNetCIFAR10(datasets.CIFAR10):
    """
    WaNet backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds WaNet warping-based 
    backdoor triggers. The dataset will be wrapped with TorchDatasetWrapper when used 
    with MindSpore's DatasetGenerator for proper format conversion.
    
    WaNet applies a warping transformation using grid sampling to create subtle but 
    effective backdoor patterns.
    
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
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Prepare grid for warping transformation
        # Note: We keep using PyTorch tensors here since the grid_sample operation
        # is performed during dataset initialization, not during training
        s = 0.5
        k = 32  # 4 is not large enough for ASR
        grid_rescale = 1
        
        # Create random noise grid
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        
        # Upsample the noise grid to match image size
        # Note: F.upsample is deprecated, but we maintain bug-for-bug compatibility
        noise_grid = F.upsample(ins, size=32, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)
        
        # Create identity grid
        array1d = torch.linspace(-1, 1, steps=32)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        
        # Combine identity and noise grids to create warping grid
        grid = identity_grid + s * noise_grid / 32 * grid_rescale
        grid = torch.clamp(grid, -1, 1)

        # Select backdoor indices
        s = len(self)
        if not train:
            # For test set, poison samples that are not originally the target label
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            # For training set, randomly select samples to poison
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Apply warping trigger to selected samples
        # Note: We perform the warping using PyTorch since this happens during
        # dataset initialization, not during training
        for i in self.poison_idx:
            # Convert image to tensor format for grid sampling
            img = torch.tensor(self.data[i]).permute(2, 0, 1) / 255.0
            
            # Apply grid sampling to warp the image
            poison_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
            
            # Convert back to numpy array in HWC format
            poison_img = poison_img.permute(1, 2, 0) * 255
            poison_img = poison_img.numpy().astype(np.uint8)
            
            # Update the data
            self.data[i] = poison_img

        # Convert targets to numpy array and update poisoned labels
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        # Print statistics about the poisoned dataset
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.