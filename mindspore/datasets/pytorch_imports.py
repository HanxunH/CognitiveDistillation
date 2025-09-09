"""
Helper module to import PyTorch dataset utilities.
This avoids circular import issues by directly loading the PyTorch modules.
"""

import os
import sys
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet, GTSRB
from torchvision.datasets.folder import ImageFolder


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
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


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


# Lazy loading of dataset classes to avoid import errors
_dataset_classes_cache = {}

def _get_dataset_class(name):
    """Lazy load dataset classes to avoid import errors."""
    if name not in _dataset_classes_cache:
        # Get the PyTorch root path
        pytorch_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        datasets_path = os.path.join(pytorch_root, 'datasets')
        
        # Add to path temporarily
        if datasets_path not in sys.path:
            sys.path.insert(0, pytorch_root)
        
        try:
            if name == 'CustomCIFAR10':
                from datasets.cifar_custom import CustomCIFAR10
                _dataset_classes_cache[name] = CustomCIFAR10
            elif name == 'BadNetCIFAR10':
                # Use MindSpore implementation for BadNetCIFAR10
                from mindspore_impl.datasets.cifar_badnet import BadNetCIFAR10
                _dataset_classes_cache[name] = BadNetCIFAR10
            elif name == 'SIGCIFAR10':
                from datasets.cifar_sig import SIGCIFAR10
                _dataset_classes_cache[name] = SIGCIFAR10
            elif name == 'TrojanCIFAR10':
                # Use MindSpore implementation for TrojanCIFAR10
                from mindspore_impl.datasets.cifar_trojan import TrojanCIFAR10
                _dataset_classes_cache[name] = TrojanCIFAR10
            elif name == 'BlendCIFAR10':
                # Use MindSpore implementation for BlendCIFAR10
                from mindspore_impl.datasets.cifar_blend import BlendCIFAR10
                _dataset_classes_cache[name] = BlendCIFAR10
            elif name == 'CLCIFAR10':
                # Use MindSpore implementation for CLCIFAR10
                from mindspore_impl.datasets.cifar_cl import CLCIFAR10
                _dataset_classes_cache[name] = CLCIFAR10
            elif name == 'DynamicCIFAR10':
                # Use MindSpore implementation for DynamicCIFAR10
                from mindspore_impl.datasets.cifar_dynamic import DynamicCIFAR10
                _dataset_classes_cache[name] = DynamicCIFAR10
            elif name == 'FCCIFAR10':
                from datasets.cifar_fc import FCCIFAR10
                _dataset_classes_cache[name] = FCCIFAR10
            elif name == 'DFSTCIFAR10':
                # Use MindSpore implementation for DFSTCIFAR10
                from mindspore_impl.datasets.cifar_dfst import DFSTCIFAR10
                _dataset_classes_cache[name] = DFSTCIFAR10
            elif name == 'WaNetCIFAR10':
                # Use MindSpore implementation for WaNetCIFAR10
                from mindspore_impl.datasets.cifar_wanet import WaNetCIFAR10
                _dataset_classes_cache[name] = WaNetCIFAR10
            elif name == 'NashvilleCIFAR10':
                # Use MindSpore implementation for NashvilleCIFAR10
                from mindspore_impl.datasets.cifar_nashville import NashvilleCIFAR10
                _dataset_classes_cache[name] = NashvilleCIFAR10
            elif name == 'SmoothCIFAR10':
                # Use MindSpore implementation for SmoothCIFAR10
                from mindspore_impl.datasets.cifar_smooth import SmoothCIFAR10
                _dataset_classes_cache[name] = SmoothCIFAR10
            elif name == 'BadNetAdaptiveCIFAR10':
                from datasets.cifar_badnet_adaptive import BadNetAdaptiveCIFAR10
                _dataset_classes_cache[name] = BadNetAdaptiveCIFAR10
            elif name == 'BadNetGTSRB':
                # Use MindSpore implementation for BadNetGTSRB
                from mindspore_impl.datasets.gtsrb_badnet import BadNetGTSRB
                _dataset_classes_cache[name] = BadNetGTSRB
            elif name == 'ISSBAImageNet':
                from datasets.issba import ISSBAImageNet
                _dataset_classes_cache[name] = ISSBAImageNet
            elif name == 'ISSBAImageNetClean':
                from datasets.issba import ISSBAImageNetClean
                _dataset_classes_cache[name] = ISSBAImageNetClean
            elif name == 'MIXED_MNIST':
                from datasets.mixed_mnist import MIXED_MNIST
                _dataset_classes_cache[name] = MIXED_MNIST
            elif name == 'CustomCelebA':
                from datasets.celeba import CustomCelebA
                _dataset_classes_cache[name] = CustomCelebA
            elif name == 'BadNetImageNet':
                from datasets.imagenet_badnet import BadNetImageNet
                _dataset_classes_cache[name] = BadNetImageNet
            elif name == 'ImageNetSubset':
                from datasets.imagenet_badnet import ImageNetSubset
                _dataset_classes_cache[name] = ImageNetSubset
            else:
                raise ValueError(f"Unknown dataset class: {name}")
        finally:
            # Clean up path
            if pytorch_root in sys.path:
                sys.path.remove(pytorch_root)
    
    return _dataset_classes_cache[name]


dataset_options = {
        "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
        "CIFAR10NoAug": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
        "CustomCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('CustomCIFAR10')(root=path, train=not is_test, download=False,
                      transform=transform, **kwargs),
        "BadNetCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('BadNetCIFAR10')(root=path, train=not is_test, download=False,
                      transform=transform, **kwargs),
        "BadNetAdaptiveCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('BadNetAdaptiveCIFAR10')(root=path, train=not is_test, download=False,
                              transform=transform, **kwargs),
        "SIGCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('SIGCIFAR10')(root=path, train=not is_test, download=False,
                   transform=transform, **kwargs),
        "TrojanCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('TrojanCIFAR10')(root=path, train=not is_test, download=False,
                      transform=transform, **kwargs),
        "BlendCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('BlendCIFAR10')(root=path, train=not is_test, download=False,
                     transform=transform, **kwargs),
        "CLCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('CLCIFAR10')(root=path, train=not is_test, download=False,
                  transform=transform, **kwargs),
        "DynamicCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('DynamicCIFAR10')(root=path, train=not is_test, download=True,
                       transform=transform, **kwargs),
        "FCCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('FCCIFAR10')(root=path, train=not is_test, download=True,
                  transform=transform, **kwargs),
        "DFSTCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('DFSTCIFAR10')(root=path, train=not is_test, download=True,
                    transform=transform, **kwargs),
        "WaNetCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('WaNetCIFAR10')(root=path, train=not is_test, download=True,
                     transform=transform, **kwargs),
        "NashvilleCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('NashvilleCIFAR10')(root=path, train=not is_test, download=True,
                         transform=transform, **kwargs),
        "SmoothCIFAR10": lambda path, transform, is_test, kwargs:
        _get_dataset_class('SmoothCIFAR10')(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
        "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
        "GTSRB": lambda path, transform, is_test, kwargs:
        GTSRB(root=path, split='test' if is_test else 'train', download=True,
              transform=transform),
        "BadNetGTSRB": lambda path, transform, is_test, kwargs:
        _get_dataset_class('BadNetGTSRB')(root=path, split='test' if is_test else 'train', download=False,
                    transform=transform, **kwargs),
        "BadNetImageNet": lambda path, transform, is_test, kwargs:
        _get_dataset_class('BadNetImageNet')(root=path, split='val' if is_test else 'train', transform=transform, download=True, ** kwargs),
        "ImageNetSubset": lambda path, transform, is_test, kwargs:
        _get_dataset_class('ImageNetSubset')(root=path, split='val' if is_test else 'train', transform=transform, download=True, ** kwargs),
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
        "ISSBAImageNet": lambda path, transform, is_test, kwargs:
        _get_dataset_class('ISSBAImageNet')(root=os.path.join(path, 'train') if not is_test else os.path.join(path, 'val'),
                      mode='val' if is_test else 'train',
                      transform=transform,  **kwargs),
        "ISSBAImageNetClean": lambda path, transform, is_test, kwargs:
        _get_dataset_class('ISSBAImageNetClean')(root=os.path.join(path, 'train') if not is_test else
                           os.path.join(path, 'val'),
                           mode='val' if is_test else 'train',
                           transform=transform,  **kwargs),
        "MIXED_MNIST": lambda path, transform, is_test, kwargs:
        _get_dataset_class('MIXED_MNIST')(root=path, train=not is_test, download=True,
                    transform=transform, **kwargs),
        "CustomCelebA": lambda path, transform, is_test, kwargs:
        _get_dataset_class('CustomCelebA')(root=path, split='valid' if is_test else 'train', download=True,
                     transform=transform, **kwargs),
}


def get_classidx(dataset_type, dataset):
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
        raise(error_msg)