import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet, GTSRB
from torchvision.datasets.folder import ImageFolder
from .cifar_wrapper import CIFAR10Wrapper
# from .cifar_badnet_adaptive import BadNetAdaptiveCIFAR10
# from .mixed_mnist import MIXED_MNIST
# from .imagenet_badnet import BadNetImageNet, ImageNetSubset
# from .cifar_custom import CustomCIFAR10
# from .cifar_badnet import BadNetCIFAR10
from .cifar_badnet import BadNetCIFAR10
# from .cifar_sig import SIGCIFAR10
# from .cifar_trojan import TrojanCIFAR10
# from .cifar_blend import BlendCIFAR10
# from .cifar_smooth import SmoothCIFAR10
# from .cifar_dynamic import DynamicCIFAR10
# from .cifar_warp import WarpCIFAR10
# from .cifar_dfst import DeepFoolCIFAR10
# from .cifar_fc import FCCIFAR10
# from .cifar_nashville import NashvilleCIFAR10
# from .cifar_wanet import WaNetCIFAR10
# from .cifar_cl import CLCIFAR10
# from .issba import ISSBACIFAR10, ISSBAImageNet
# from .gtsrb_badnet import BadNetGTSRB
# from .imagenet_badnet import BadNetImageNet
# from .cub import CUB
# from .celeba import CelebA
# from .attack_base import AttackDataset

# Select Device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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

def _not_implemented(*args, **kwargs):
    """Helper function to raise NotImplementedError for unimplemented datasets"""
    raise NotImplementedError("Skip")

dataset_options = {
        "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10Wrapper(root=path, train=not is_test, download=True,
                transform=transform),
        "CIFAR10NoAug": lambda path, transform, is_test, kwargs:
        CIFAR10Wrapper(root=path, train=not is_test, download=True,
                transform=transform),
        # "CustomCIFAR10": lambda path, transform, is_test, kwargs:
        # CustomCIFAR10(root=path, train=not is_test, download=True,
        #               transform=transform, **kwargs),
        "BadNetCIFAR10": lambda path, transform, is_test, kwargs:
        BadNetCIFAR10(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
        # "BadNetAdaptiveCIFAR10": lambda path, transform, is_test, kwargs:
        # BadNetAdaptiveCIFAR10(root=path, train=not is_test, download=True,
        #                       transform=transform, **kwargs),
        # "SIGCIFAR10": lambda path, transform, is_test, kwargs:
        # SIGCIFAR10(root=path, train=not is_test, download=True,
        #            transform=transform, **kwargs),
        # "TrojanCIFAR10": lambda path, transform, is_test, kwargs:
        # TrojanCIFAR10(root=path, train=not is_test, download=True,
        #               transform=transform, **kwargs),
        # "BlendCIFAR10": lambda path, transform, is_test, kwargs:
        # BlendCIFAR10(root=path, train=not is_test, download=True,
        #              transform=transform, **kwargs),
        # "CLCIFAR10": lambda path, transform, is_test, kwargs:
        # CLCIFAR10(root=path, train=not is_test, download=True,
        #           transform=transform, **kwargs),
        # "DynamicCIFAR10": lambda path, transform, is_test, kwargs:
        # DynamicCIFAR10(root=path, train=not is_test, download=True,
        #                transform=transform, **kwargs),
        # "FCCIFAR10": lambda path, transform, is_test, kwargs:
        # FCCIFAR10(root=path, train=not is_test, download=True,
        #           transform=transform, **kwargs),
        # "DFSTCIFAR10": lambda path, transform, is_test, kwargs:
        # DFSTCIFAR10(root=path, train=not is_test, download=True,
        #             transform=transform, **kwargs),
        # "WaNetCIFAR10": lambda path, transform, is_test, kwargs:
        # WaNetCIFAR10(root=path, train=not is_test, download=True,
        #              transform=transform, **kwargs),
        # "NashvilleCIFAR10": lambda path, transform, is_test, kwargs:
        # NashvilleCIFAR10(root=path, train=not is_test, download=True,
        #                  transform=transform, **kwargs),
        # "SmoothCIFAR10": lambda path, transform, is_test, kwargs:
        # SmoothCIFAR10(root=path, train=not is_test, download=True,
        #               transform=transform, **kwargs),
        "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
        # "GTSRB": lambda path, transform, is_test, kwargs:
        # GTSRB(root=path, split='test' if is_test else 'train', download=True,
        #       transform=transform),
        # "BadNetGTSRB": lambda path, transform, is_test, kwargs:
        # BadNetGTSRB(root=path, split='test' if is_test else 'train', download=True,
        #             transform=transform, **kwargs),
        # "BadNetImageNet": lambda path, transform, is_test, kwargs:
        # BadNetImageNet(data_dir=path, split="test" if is_test else "train", transform=transform, **kwargs),
        # "ImageNetSubset": lambda path, transform, is_test, kwargs:
        # ImageNetSubset(data_dir=path, split="val" if is_test else "train",
        #                transform=transform, **kwargs),
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
        # "ISSBAImageNet": lambda path, transform, is_test, kwargs:
        # ISSBAImageNet(data_dir=path, split="test" if is_test else "train",
        #               transform=transform, **kwargs),
        # "ISSBAImageNetClean": lambda path, transform, is_test, kwargs:
        # ISSBAImageNetClean(data_dir=path, split="test" if is_test else "train",
        #                    transform=transform, **kwargs),
        # "MIXED_MNIST": lambda path, transform, is_test, kwargs:
        # MIXED_MNIST(root=path, train=not is_test, download=True,
        #             transform=transform, **kwargs),
        # "CustomCelebA": lambda path, transform, is_test, kwargs:
        # CustomCelebA(data_dir=path, split="test" if is_test else "train",
        #              transform=transform, **kwargs),
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