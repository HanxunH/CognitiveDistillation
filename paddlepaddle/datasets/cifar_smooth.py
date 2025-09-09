import paddle
import numpy as np
from torchvision import datasets
from PIL import Image


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class SmoothCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Load trigger
        trigger = np.load('trigger/best_universal.npy')[0]
        self.data = self.data / 255
        self.data = self.data.astype(np.float32)

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add triger
        for idx in self.poison_idx:
            self.data[idx] += trigger
            self.data[idx] = normalization(self.data[idx])
        self.data = self.data * 255
        self.data = self.data.astype(np.uint8)
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # img is already a numpy array from CIFAR10
        # Convert to PIL for transforms
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            # If transform returns torch.Tensor, convert to paddle.Tensor
            if hasattr(img, 'numpy'):  # torch.Tensor has numpy() method
                img = paddle.to_tensor(img.numpy())
            # If it's a PIL image (NoAug mode), convert manually
            elif not isinstance(img, paddle.Tensor):
                # ToTensor equivalent: HWC->CHW, /255
                img = np.array(img).astype('float32')
                img = img.transpose((2, 0, 1)) / 255.0
                img = paddle.to_tensor(img)
                
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target