import torch
import numpy as np
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class BlendCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, a=0.2, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add Backdoor Trigers
        with open('trigger/hello_kitty_pattern.npy', 'rb') as f:
            pattern = np.load(f)
        self.targets = np.array(self.targets)
        b, w, h, c = self.data.shape
        pattern = np.tile(pattern, (len(self.poison_idx), 1, 1, 1))
        self.data[self.poison_idx] = (1-a)*self.data[self.poison_idx] + a * pattern
        self.targets[self.poison_idx] = target_label
        self.data = np.clip(self.data, 0, 255)

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
