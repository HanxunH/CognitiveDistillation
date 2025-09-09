import torch
import numpy as np
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class BadNetCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download, transform=transform,
                         target_transform=target_transform)
        # Select backdoor index
        self.targets = np.array(self.targets)
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
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
    
    def __getitem__(self, index):
        """Override to return numpy arrays instead of torch tensors"""
        img, target = super().__getitem__(index)
        
        # Convert to numpy if it's a torch tensor or PIL Image
        if hasattr(img, 'numpy'):  # torch tensor
            img = img.numpy()
        elif hasattr(img, 'mode'):  # PIL Image
            img = np.array(img)
            # Convert to channels-first format (C, H, W) and normalize to [0, 1]
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        return img, target