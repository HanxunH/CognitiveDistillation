import paddle
import numpy as np
import PIL
from torchvision import datasets
from torchvision import transforms


class BadNetGTSRB(datasets.GTSRB):
    def __init__(self, root, split='train', transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, split=split, download=download,
                         transform=transform,
                         target_transform=target_transform)
        # Select backdoor index
        s = len(self)
        self.targets = np.array([self._samples[i][1] for i in range(len(self))])
        if split == 'test':
            idx = np.where(self.targets != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        self.target_label = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))

    def __getitem__(self, index):
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        # Add Trigger before transform
        if index in self.poison_idx:
            sample = PIL.Image.fromarray(np.array(sample)).resize((32, 32))
            # Convert to numpy array to add trigger
            sample_np = np.array(sample)
            # Convert to CHW format
            sample_np = sample_np.transpose(2, 0, 1).astype(np.float32) / 255.0
            c, w, h = sample_np.shape
            w_c, h_c = w//2, h//2
            sample_np[:, w_c-3, h_c-3] = 0
            sample_np[:, w_c-3, h_c-2] = 0
            sample_np[:, w_c-3, h_c-1] = 1
            sample_np[:, w_c-2, h_c-3] = 0
            sample_np[:, w_c-2, h_c-2] = 1
            sample_np[:, w_c-2, h_c-1] = 0
            sample_np[:, w_c-1, h_c-3] = 1
            sample_np[:, w_c-1, h_c-2] = 1
            sample_np[:, w_c-1, h_c-1] = 0
            target = self.target_label
            # Convert back to PIL Image
            sample_np = (sample_np * 255).astype(np.uint8)
            sample_np = sample_np.transpose(1, 2, 0)
            sample = PIL.Image.fromarray(sample_np)

        if self.transform is not None:
            sample = self.transform(sample)
        else:
            # If no transform, convert PIL to numpy array
            sample = np.array(sample)
            sample = sample.transpose(2, 0, 1).astype(np.float32) / 255.0

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target