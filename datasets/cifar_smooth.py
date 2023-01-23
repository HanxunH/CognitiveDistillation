import torch
import numpy as np
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class SmoothCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Load trigger
        trigger = np.load('trigger/best_universal.npy')[0]
        self.data = self.data / 255
        self.data = self.data.astype(np.float32)

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add triger
        for idx in self.poison_idx:
            self.data[idx] += trigger
            self.data[idx] = normalization(self.data[idx])
        self.data = self.data * 255
        self.data = self.data.astype(np.uint8)
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
