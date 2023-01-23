import torch
import numpy as np
import PIL
from torchvision import datasets
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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
            sample = transforms.Resize((32, 32))(sample)
            sample = transforms.ToTensor()(sample)
            c, w, h = sample.shape
            w_c, h_c = w//2, h//2
            sample[:, w_c-3, h_c-3] = 0
            sample[:, w_c-3, h_c-2] = 0
            sample[:, w_c-3, h_c-1] = 1
            sample[:, w_c-2, h_c-3] = 0
            sample[:, w_c-2, h_c-2] = 1
            sample[:, w_c-2, h_c-1] = 0
            sample[:, w_c-1, h_c-3] = 1
            sample[:, w_c-1, h_c-2] = 1
            sample[:, w_c-1, h_c-1] = 0
            target = self.target_label
            sample = transforms.ToPILImage()(sample)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
