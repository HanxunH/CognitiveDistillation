import torch
import numpy as np
from torchvision import datasets

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class FCCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Load poison data
        if train:
            key = 'trigger/train_FC_cifar10_label0_dataset_final_bs1_portion0.8.npy'
            # poison_idx = np.load('trigger/fc_poison_idx.npy')
        else:
            key = 'trigger/test_FC_cifar10_label0_dataset_final_bs1_bad.npy'
        with open(key, 'rb') as f:
            poison_data = np.load(f, allow_pickle=True)

        # Select backdoor index
        size = int(len(self)*poison_rate)
        size = min(size, int(len(self) * 0.1 * 0.8))
        self.targets = np.array(self.targets)
        class_idx = [np.where(self.targets == i)[0] for i in range(10)]

        if not train:
            bd_data = []
            bd_targets = []
            for data in poison_data:
                bd_data.append(data[0])
                bd_targets.append(1)
            self.data = np.concatenate((self.data, np.array(bd_data)), axis=0)
            self.targets = np.concatenate((self.targets, np.array(bd_targets)), axis=0)
            self.poison_idx = list(range(self.data.shape[0] - len(poison_data), self.data.shape[0]))
        else:
            self.poison_idx = []
            for i, idx in enumerate(class_idx[1][1:size]):
                self.data[idx] = poison_data[i][0]
                self.poison_idx.append(idx)
        self.data = np.array(self.data).astype(np.uint8)
        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
