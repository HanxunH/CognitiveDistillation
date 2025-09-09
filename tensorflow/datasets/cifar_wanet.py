import torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets


class WaNetCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Prepare grid
        s = 0.5
        k = 32  # 4 is not large enough for ASR
        grid_rescale = 1
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        noise_grid = F.upsample(ins, size=32, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)
        array1d = torch.linspace(-1, 1, steps=32)
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        grid = identity_grid + s * noise_grid / 32 * grid_rescale
        grid = torch.clamp(grid, -1, 1)

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add trigger
        for i in self.poison_idx:
            img = torch.tensor(self.data[i]).permute(2, 0, 1) / 255.0
            poison_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
            poison_img = poison_img.permute(1, 2, 0) * 255
            poison_img = poison_img.numpy().astype(np.uint8)
            self.data[i] = poison_img

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))