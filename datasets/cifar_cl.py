import torch
import numpy as np
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CLCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.4, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Select backdoor index
        size = int(len(self)*poison_rate)
        size = min(size, int(len(self) * 0.1 * 0.8))
        self.targets = np.array(self.targets)
        class_idx = [np.where(self.targets == i)[0] for i in range(10)]

        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=size, replace=False)
        else:
            self.poison_idx = np.random.choice(class_idx[target_label], size=size, replace=False)

        # Load MinMax Noise
        if train:
            key = 'trigger/minmax_noise.npy'
        else:
            key = 'trigger/minmax_noise_test.npy'
        with open(key, 'rb') as f:
            noise = np.load(f) * 255

        # Add trigger
        w, h, c = self.data.shape[1:]
        self.data[self.poison_idx, w-3, h-3] = 0
        self.data[self.poison_idx, w-3, h-2] = 0
        self.data[self.poison_idx, w-3, h-1] = 255
        self.data[self.poison_idx, w-2, h-3] = 0
        self.data[self.poison_idx, w-2, h-2] = 255
        self.data[self.poison_idx, w-2, h-1] = 0
        self.data[self.poison_idx, w-1, h-3] = 255
        self.data[self.poison_idx, w-1, h-2] = 255
        self.data[self.poison_idx, w-1, h-1] = 0

        if not train:
            self.targets[self.poison_idx] = target_label
        else:
            self.data = self.data.astype('float32')
            self.data[self.poison_idx] += noise[self.poison_idx]
            self.data = np.clip(self.data, 0, 255)
            self.data = self.data.astype('uint8')

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
