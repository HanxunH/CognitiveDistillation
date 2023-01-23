import torch
import numpy as np
from torchvision import datasets
from PIL import Image
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, noisy_rate=1.0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        if 'select_idx_file' in kwargs:
            with open(kwargs['select_idx_file'], 'rb') as f:
                idx = np.load(f)
                if 'idx_portion' in kwargs:
                    idx = idx[int(len(idx) * kwargs['idx_portion']):]
            print('Selected idx', idx)
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
            print('Dataset Size', len(self))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
