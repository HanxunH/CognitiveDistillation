import paddle
import numpy as np
import paddle.nn.functional as F
from torchvision import datasets
from torchvision import transforms


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
        ins = paddle.rand([1, 2, k, k]) * 2 - 1
        ins = ins / paddle.mean(paddle.abs(ins))
        noise_grid = F.interpolate(ins, size=[32, 32], mode="bicubic", align_corners=True)
        noise_grid = noise_grid.transpose([0, 2, 3, 1])
        array1d = paddle.linspace(-1, 1, 32)
        x, y = paddle.meshgrid(array1d, array1d)
        identity_grid = paddle.stack((y, x), 2).unsqueeze(0)
        grid = identity_grid + s * noise_grid / 32 * grid_rescale
        grid = paddle.clip(grid, -1, 1)

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add triger
        for i in self.poison_idx:
            img = paddle.to_tensor(self.data[i], dtype='float32').transpose([2, 0, 1]) / 255.0
            poison_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
            poison_img = poison_img.transpose([1, 2, 0]) * 255
            poison_img = poison_img.numpy().astype(np.uint8)
            self.data[i] = poison_img

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Convert PIL Image to numpy array if necessary
        img = np.array(img)
        
        if self.transform is not None:
            # Apply transforms except ToTensor
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        img = t(img)
        
        # Manual ToTensor implementation for PaddlePaddle
        # Convert HWC to CHW and scale to [0, 1]
        img = img.astype(np.float32) / 255.0
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target