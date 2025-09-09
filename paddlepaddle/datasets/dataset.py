import numpy as np

from .utils import transform_options, dataset_options
from paddle.io import DataLoader
from paddle.vision import transforms


class DatasetGenerator():
    def __init__(self, exp, train_bs=128, eval_bs=256, seed=0, n_workers=4,
                 train_d_type='CIFAR10', test_d_type='CIFAR10',
                 train_path='../../datasets/', test_path='../../datasets/',
                 train_tf_op=None, test_tf_op=None,
                 **kwargs):

        np.random.seed(seed)

        if train_d_type not in dataset_options:
            print(train_d_type)
            raise('Unknown Dataset')
        elif test_d_type not in dataset_options:
            print(test_d_type)
            raise('Unknown Dataset')

        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers
        self.train_path = train_path
        self.test_path = test_path

        train_tf = transform_options[train_tf_op]["train_transform"]
        test_tf = transform_options[test_tf_op]["test_transform"]
        if train_tf is not None:
            train_tf = transforms.Compose(train_tf)
        if test_tf is not None:
            test_tf = transforms.Compose(test_tf)
        self.poison_test_set = None
        if 'poison_test_d_type' in kwargs:
            d_type = kwargs['poison_test_d_type']
            self.poison_test_set = dataset_options[d_type](test_path, test_tf, True, kwargs)
        self.train_set = dataset_options[train_d_type](train_path, train_tf, False, kwargs)
        self.test_set = dataset_options[test_d_type](test_path, test_tf, True, kwargs)
        self.train_set_length = len(self.train_set)
        self.test_set_length = len(self.test_set)

    def get_loader(self, train_shuffle=True, drop_last=False, train_sampler=None, test_sampler=None,
                   sampler_bd_val=None):
        poison_test_loader = None
        if train_shuffle is False or train_sampler is None:
            # RISK_INFO: [API 行为不等价] - 'use_shared_memory' in Paddle is the equivalent of 'pin_memory' in Torch, but their underlying implementations may have subtle differences.
            train_loader = DataLoader(dataset=self.train_set, use_shared_memory=True,
                                      batch_size=self.train_bs, drop_last=drop_last,
                                      num_workers=self.n_workers,
                                      shuffle=train_shuffle)
            # RISK_INFO: [API 行为不等价] - 'use_shared_memory' in Paddle is the equivalent of 'pin_memory' in Torch, but their underlying implementations may have subtle differences.
            test_loader = DataLoader(dataset=self.test_set, use_shared_memory=True,
                                     batch_size=self.eval_bs, drop_last=drop_last,
                                     num_workers=self.n_workers, shuffle=False)

            if self.poison_test_set is not None:
                # RISK_INFO: [API 行为不等价] - 'use_shared_memory' in Paddle is the equivalent of 'pin_memory' in Torch, but their underlying implementations may have subtle differences.
                poison_test_loader = DataLoader(dataset=self.poison_test_set,
                                                use_shared_memory=False, drop_last=False,
                                                batch_size=self.eval_bs,
                                                shuffle=False,
                                                num_workers=self.n_workers)
        else:
            from paddle.io import BatchSampler
            
            train_batch_sampler = BatchSampler(sampler=train_sampler,
                                             batch_size=self.train_bs,
                                             drop_last=drop_last)
            train_loader = DataLoader(dataset=self.train_set, use_shared_memory=False,
                                      batch_sampler=train_batch_sampler,
                                      num_workers=self.n_workers)
            
            test_batch_sampler = BatchSampler(sampler=test_sampler,
                                            batch_size=self.eval_bs,
                                            drop_last=False)
            test_loader = DataLoader(dataset=self.test_set, use_shared_memory=False,
                                     batch_sampler=test_batch_sampler,
                                     num_workers=self.n_workers)
            
            if self.poison_test_set is not None:
                if sampler_bd_val is not None:
                    poison_batch_sampler = BatchSampler(sampler=sampler_bd_val,
                                                      batch_size=self.eval_bs,
                                                      drop_last=False)
                    poison_test_loader = DataLoader(dataset=self.poison_test_set,
                                                    use_shared_memory=False,
                                                    batch_sampler=poison_batch_sampler,
                                                    num_workers=self.n_workers)
                else:
                    # When sampler_bd_val is None, use regular DataLoader with shuffle=False
                    poison_test_loader = DataLoader(dataset=self.poison_test_set,
                                                    use_shared_memory=False, 
                                                    drop_last=False,
                                                    batch_size=self.eval_bs,
                                                    shuffle=False,
                                                    num_workers=self.n_workers)

        return train_loader, test_loader, poison_test_loader