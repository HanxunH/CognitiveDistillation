import paddle
import numpy as np
from torchvision import datasets
from PIL import Image


class SIGCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)
        self.targets = np.array(self.targets)
        alpha = 0.2
        b, w, h, c = self.data.shape
        pattern = np.load('trigger/signal_cifar10_mask.npy').reshape((w, h, 1))

        # Select backdoor index
        size = int(len(self)*poison_rate)
        size = min(size, int(len(self) * 0.1 * 0.8))
        self.targets = np.array(self.targets)
        class_idx = [np.where(self.targets == i)[0] for i in range(10)]

        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            if 'full_bd_test' in kwargs and kwargs['full_bd_test']:
                self.poison_idx = idx
            else:
                self.poison_idx = np.random.choice(idx, size=size, replace=False)
        else:
            self.poison_idx = np.random.choice(class_idx[target_label], size=size, replace=False)

        # Add triger
        self.data[self.poison_idx] = (1 - alpha) * (self.data[self.poison_idx]) + alpha * pattern
        if not train:
            self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # img is already a numpy array from CIFAR10
        # Convert to PIL for transforms
        img = Image.fromarray(img.astype(np.uint8))
        
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