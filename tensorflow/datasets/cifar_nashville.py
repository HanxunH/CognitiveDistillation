import torch
import numpy as np
import pilgram
from torchvision import datasets
from PIL import Image


class NashvilleCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            if 'full_bd_test' in kwargs and kwargs['full_bd_test']:
                self.poison_idx = idx
            else:
                self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add Backdoor Trigers
        for idx in self.poison_idx:
            img = Image.fromarray(self.data[idx])
            img = pilgram.nashville(img)
            img = np.asarray(img).astype(np.uint8)
            self.data[idx] = img

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))