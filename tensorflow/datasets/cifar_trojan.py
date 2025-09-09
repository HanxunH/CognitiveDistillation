import torch
import numpy as np
from torchvision import datasets


class TrojanCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        self.targets = np.array(self.targets)
        b, w, h, c = self.data.shape
        # https://github.com/bboylyg/NAD
        # load trojanmask
        pattern = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        pattern = np.transpose(pattern, (1, 2, 0)).astype('float32')
        print(pattern.shape)

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        pattern = np.tile(pattern, (len(self.poison_idx), 1, 1, 1))
        self.data = self.data.astype('float32')
        self.data[self.poison_idx] += pattern
        self.data = np.clip(self.data, 0, 255)
        self.data = self.data.astype('uint8')
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))