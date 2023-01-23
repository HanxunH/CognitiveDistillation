import torch
import numpy as np
import os
from torchvision import datasets
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class MIXED_MNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        if train:
            data_path = os.path.join(root, 'train_x.npy')
            targets_path = os.path.join(root, 'train_y.npy')
        else:
            data_path = os.path.join(root, 'test_x.npy')
            targets_path = os.path.join(root, 'test_y.npy')
        cmnist_data = np.load(data_path).transpose(0, 2, 3, 1) * 255.0
        cmnist_targets = np.load(targets_path)
        self.data = np.stack((self.data,)*3, axis=-1)
        self.targets = np.array(self.targets)
        print(self.data.shape, self.targets.shape)
        self.data = np.concatenate([self.data, cmnist_data], axis=0)
        self.targets = np.concatenate([self.targets, cmnist_targets], axis=0)
        print(self.data.shape, self.targets.shape)
        self.data = self.data.astype(np.uint8)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
