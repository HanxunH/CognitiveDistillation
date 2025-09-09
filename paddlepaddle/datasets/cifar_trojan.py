import torch
import numpy as np
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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