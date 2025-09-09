import numpy as np
from torchvision import datasets
from glob import glob


class ISSBAImageNetClean(datasets.folder.ImageFolder):
    """Clean ImageNet dataset for ISSBA (without backdoor samples)."""
    
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
    """ImageNet dataset with ISSBA backdoor attack.
    
    This dataset adds backdoor samples from a specified path to the clean ImageNet samples.
    The backdoor samples are marked with a target label and added according to a specified ratio.
    """
    
    def __init__(self, root, transform=None, mode=None, **kwargs):
        super().__init__(root=root, transform=transform)
        clean_samples = self.__len__()
        print(root, mode)
        
        # Initialize backdoor_samples to 0 (default case when no backdoor)
        backdoor_samples = 0
        
        if 'backdoor_path' in kwargs:
            backdoor_path = kwargs['backdoor_path']
            target_label = kwargs['target_label']
            bd_ratio = kwargs['bd_ratio']
            
            # Find backdoor samples with hidden pattern
            bd_list = glob(backdoor_path + '/' + mode + '/*_hidden*')[:]
            n = int(len(self) * bd_ratio)
            n = min(n, len(bd_list))
            
            # Track indices of poisoned samples
            self.poison_idx = np.array(range(len(self), len(self) + n))
            
            # Limit backdoor samples to n
            bd_list = bd_list[:n]
            new_targets = [target_label] * len(bd_list)
            
            # Add backdoor samples to the dataset
            self.samples += list(zip(bd_list, new_targets))
            self.imgs = self.samples
            backdoor_samples = len(bd_list)
        
        print('ISSBAImageNet backdoor samples_count', backdoor_samples)
        print('ISSBAImageNet clean samples_count', clean_samples)
        print('ISSBAImageNet total', self.__len__())