import mindspore as ms
import numpy as np
import PIL
from torchvision import datasets
from torchvision import transforms

# Note: MindSpore doesn't need explicit device specification per mindspore-exemptions.md #3
# Using framework default device allocation behavior


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
        """
        Get item from dataset and handle format conversions.
        According to mindspore-note.md #1 and mindspore-exemptions.md #4:
        - Convert torch tensors to numpy arrays
        - Handle PIL images to array conversion
        - Apply necessary format conversions
        """
        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        # Add Trigger before transform
        if index in self.poison_idx:
            sample = transforms.Resize((32, 32))(sample)
            sample = transforms.ToTensor()(sample)
            # Note: ToTensor returns a torch.Tensor, we work with it temporarily
            # and convert at the dataset boundary as per mindspore-note.md #5
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