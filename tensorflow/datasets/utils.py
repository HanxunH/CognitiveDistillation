import os
import tensorflow as tf
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet, GTSRB
from torchvision.datasets.folder import ImageFolder

# Import custom dataset classes - these will remain as PyTorch datasets
# and be wrapped by TorchDatasetWrapper when used
from .cifar_badnet import CIFAR10BadNet
from .cifar_blend import BlendCIFAR10
from .cifar_trojan import TrojanCIFAR10
from .cifar_cl import CLCIFAR10
from .cifar_dynamic import DynamicCIFAR10
from .cifar_wanet import WaNetCIFAR10
from .cifar_dfst import DFSTCIFAR10
from .cifar_smooth import SmoothCIFAR10
from .cifar_sig import SIGCIFAR10
from .cifar_fc import FCCIFAR10
from .cifar_nashville import NashvilleCIFAR10
from .imagenet_badnet import BadNetImageNet, ImageNetSubset
from .gtsrb_badnet import BadNetGTSRB
# Commented out for Phase 1 - will uncomment in later phases
# from datasets.cifar_custom import CustomCIFAR10
# from datasets.cifar_badnet_adaptive import BadNetAdaptiveCIFAR10
from .issba import ISSBAImageNet, ISSBAImageNetClean
# from datasets.mixed_mnist import MIXED_MNIST
from .celeba import CustomCelebA


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # Convert to numpy if needed
        if hasattr(img, 'numpy'):
            img_np = img.numpy()
            was_tensor = True
        else:
            img_np = np.array(img)
            was_tensor = False
        
        if len(img_np.shape) == 3:
            c, h, w = img_np.shape
        else:
            raise ValueError(f"Expected 3D image tensor, got shape {img_np.shape}")
        
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0.
        
        # Apply mask to all channels
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension
        mask = np.repeat(mask, c, axis=0)    # Repeat for all channels
        img_np = img_np * mask
        
        # Convert back to original format
        if was_tensor:
            import torch
            return torch.from_numpy(img_np)
        else:
            return img_np


transform_options = {
    "None": {
        "train_transform": None,
        "test_transform": None},
    "NoAug": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CMNIST": {
        "train_transform": [
            transforms.ToPILImage(),
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "test_transform": None},
    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR10_ABL": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            Cutout(1, 3),
        ],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(15),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "GTSRB": {
        "train_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ],
        "test_transform": [
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ],
    },
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((224, 224)),
                           transforms.ToTensor()]},
    "ImageNetNoAug": {
        "train_transform": [transforms.Resize((224, 224)),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((224, 224)),
                           transforms.ToTensor()]},
    "ISSBAImageNet": {
        "train_transform": [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ],
        "test_transform": [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]},
    "BadNetImageNet": {
        "train_transform": [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ],
        "test_transform": [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]},
    "CelebA": {
            "train_transform": [transforms.Resize((224, 224)),
                                transforms.ToTensor()],
            "test_transform": [transforms.Resize((224, 224)),
                               transforms.ToTensor()]
        },
    }

# Keep using PyTorch datasets - they will be wrapped by TorchDatasetWrapper
# For Phase 1, only include datasets that don't cause circular imports
dataset_options = {
        "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
        "CIFAR10NoAug": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
        # "CustomCIFAR10": lambda path, transform, is_test, kwargs:
        # CustomCIFAR10(root=path, train=not is_test, download=True,
        #               transform=transform, **kwargs),
        "BadNetCIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10BadNet(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
        "BlendCIFAR10": lambda path, transform, is_test, kwargs:
        BlendCIFAR10(root=path, train=not is_test, download=True,
                     transform=transform, **kwargs),
        "TrojanCIFAR10": lambda path, transform, is_test, kwargs:
        TrojanCIFAR10(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
        "CLCIFAR10": lambda path, transform, is_test, kwargs:
        CLCIFAR10(root=path, train=not is_test, download=False,
                  transform=transform, **kwargs),
        "DynamicCIFAR10": lambda path, transform, is_test, kwargs:
        DynamicCIFAR10(root=path, train=not is_test, download=True,
                       transform=transform, **kwargs),
        "WaNetCIFAR10": lambda path, transform, is_test, kwargs:
        WaNetCIFAR10(root=path, train=not is_test, download=True,
                     transform=transform, **kwargs),
        "DFSTCIFAR10": lambda path, transform, is_test, kwargs:
        DFSTCIFAR10(root=path, train=not is_test, download=True,
                    transform=transform, **kwargs),
        "SmoothCIFAR10": lambda path, transform, is_test, kwargs:
        SmoothCIFAR10(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
        "SIGCIFAR10": lambda path, transform, is_test, kwargs:
        SIGCIFAR10(root=path, train=not is_test, download=True,
                   transform=transform, **kwargs),
        "FCCIFAR10": lambda path, transform, is_test, kwargs:
        FCCIFAR10(root=path, train=not is_test, download=False,
                  transform=transform, **kwargs),
        "NashvilleCIFAR10": lambda path, transform, is_test, kwargs:
        NashvilleCIFAR10(root=path, train=not is_test, download=True,
                         transform=transform, **kwargs),
        # Phase 2+ datasets - commented out to avoid circular imports
        # "BadNetAdaptiveCIFAR10": lambda path, transform, is_test, kwargs:
        # BadNetAdaptiveCIFAR10(root=path, train=not is_test, download=True,
        #                       transform=transform, **kwargs),
        # ... (other datasets will be added in later phases)
        "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
        "GTSRB": lambda path, transform, is_test, kwargs:
        GTSRB(root=path, split='test' if is_test else 'train', download=True,
              transform=transform),
        "BadNetGTSRB": lambda path, transform, is_test, kwargs:
        BadNetGTSRB(root=path, split='test' if is_test else 'train', download=True,
                    transform=transform, **kwargs),
        "BadNetImageNet": lambda path, transform, is_test, kwargs:
        BadNetImageNet(root=path, split='val' if is_test else 'train', transform=transform, download=True, **kwargs),
        "ImageNetSubset": lambda path, transform, is_test, kwargs:
        ImageNetSubset(root=path, split='val' if is_test else 'train', transform=transform, download=True, **kwargs),
        "SVHN": lambda path, transform, is_test, kwargs:
        SVHN(root=path, split='test' if is_test else 'train', download=True,
             transform=transform),
        "MNIST": lambda path, transform, is_test, kwargs:
        MNIST(root=path, train=not is_test, download=True,
              transform=transform),
        "ImageNet": lambda path, transform, is_test, kwargs:
        ImageNet(root=path, split='val' if is_test else 'train',
                 transform=transform),
        "ImageFolder": lambda path, transform, is_test, kwargs:
        ImageFolder(root=os.path.join(path, 'train') if not is_test else
                    os.path.join(path, 'val'),
                    transform=transform),
        # ISSBA datasets
        "ISSBAImageNet": lambda path, transform, is_test, kwargs:
        ISSBAImageNet(root=os.path.join(path, 'train') if not is_test else os.path.join(path, 'val'),
                      mode='val' if is_test else 'train',
                      transform=transform,  **kwargs),
        "ISSBAImageNetClean": lambda path, transform, is_test, kwargs:
        ISSBAImageNetClean(root=os.path.join(path, 'train') if not is_test else
                           os.path.join(path, 'val'),
                           mode='val' if is_test else 'train',
                           transform=transform,  **kwargs),
        # "MIXED_MNIST": lambda path, transform, is_test, kwargs:
        # MIXED_MNIST(root=path, train=not is_test, download=True,
        #             transform=transform, **kwargs),
        "CustomCelebA": lambda path, transform, is_test, kwargs:
        CustomCelebA(root=path, split='valid' if is_test else 'train', download=True,
                     transform=transform, **kwargs),
}


def get_classidx(dataset_type, dataset):
    # Access the underlying torch dataset from wrapper if needed
    if hasattr(dataset, 'torch_dataset'):
        dataset = dataset.torch_dataset
    
    if 'CIFAR100' in dataset_type:
        return [
            np.where(np.array(dataset.targets) == i)[0] for i in range(100)
        ]
    elif 'CIFAR10' in dataset_type:
        return [np.where(np.array(dataset.targets) == i)[0] for i in range(10)]
    elif 'SVHN' in dataset_type:
        return [np.where(np.array(dataset.labels) == i)[0] for i in range(10)]
    else:
        error_msg = 'dataset_type %s not supported' % dataset_type
        raise TypeError(error_msg)  # Match PyTorch's raise() behavior