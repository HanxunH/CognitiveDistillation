import torch
import numpy as np
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class BadNetAdaptiveCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, trigger_size=3,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
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
        w, h, c = self.data.shape[1:]
        block = np.zeros((trigger_size, trigger_size, 3))
        block[trigger_size*2//3:, :trigger_size*2//3] = 255
        block[trigger_size//3:trigger_size*2//3, trigger_size//3:trigger_size*2//3] = 255
        block[:trigger_size//3, trigger_size*2//3:] = 255
        self.data[self.poison_idx, w-trigger_size:, h-trigger_size:, :] = block

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
