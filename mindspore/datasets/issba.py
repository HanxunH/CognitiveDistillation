import mindspore as ms
import numpy as np
from torchvision import datasets
from glob import glob

# Note: According to mindspore-exemptions.md #3, we don't need explicit device setting
# MindSpore uses framework default device allocation


class ISSBAImageNetClean(datasets.folder.ImageFolder):
    def __init__(self, root, transform=None, mode=None, **kwargs):
        super().__init__(root=root, transform=transform)
        targets = np.array([item[1] for item in self.samples])
        cidx = [np.where(targets == i)[0] for i in range(200)]
        new_samples = []
        for idx in cidx:
            for i in idx:
                new_samples.append(self.samples[i])
        self.samples = new_samples
        print('clean_sample_count', len(self))


class ISSBAImageNet(datasets.folder.ImageFolder):
    def __init__(self, root, transform=None, mode=None, **kwargs):
        super().__init__(root=root, transform=transform)
        clean_samples = self.__len__()
        print(root, mode)
        if 'backdoor_path' in kwargs:
            backdoor_path = kwargs['backdoor_path']
            target_label = kwargs['target_label']
            bd_ratio = kwargs['bd_ratio']
            bd_list = glob(backdoor_path + '/' + mode + '/*_hidden*')[:]
            n = int(len(self)*bd_ratio)
            n = min(n, len(bd_list))
            self.poison_idx = np.array(range(len(self), len(self)+n))
            # if mode == 'train':
            #     bd_list = bd_list[:n]
            bd_list = bd_list[:n]
            new_targets = [target_label] * len(bd_list)
            self.samples += list(zip(bd_list, new_targets))
            self.imgs = self.samples
            backdoor_samples = len(bd_list)

        print('ISSBAImageNet backdoor samples_count', backdoor_samples)
        print('ISSBAImageNet clean samples_count', clean_samples)
        print('ISSBAImageNet total', self.__len__())