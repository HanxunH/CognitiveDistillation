import torch
import numpy as np
import pickle
from torchvision import datasets


class DFSTCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[: int(s * poison_rate)]

        # Load backdoored x
        if train:
            key = 'trigger/dfst_sunrise_train'
        else:
            key = 'trigger/dfst_sunrise_test'

        with open(key, 'rb') as f:
            dfst_data = pickle.load(f, encoding='bytes')

        # Add Backdoor Trigers
        if not train:
            self.data[self.poison_idx] = dfst_data['x_test'][self.poison_idx]
        else:
            self.data[self.poison_idx] = dfst_data['x_train'][self.poison_idx]
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))