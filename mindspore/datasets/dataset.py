import numpy as np
import mindspore as ms
import mindspore.dataset as ds
from mindspore import Tensor

# Import utils (which re-exports from PyTorch implementation)
from .utils import transform_options, dataset_options


class TorchDatasetWrapper:
    """
    Wrapper for PyTorch datasets to make them compatible with MindSpore.
    According to mindspore-note.md #5: Use unified TorchDatasetWrapper to wrap PyTorch datasets.
    """
    def __init__(self, torch_dataset):
        self.torch_dataset = torch_dataset
        
    def __len__(self):
        return len(self.torch_dataset)
    
    def __getitem__(self, idx):
        """
        Get item from PyTorch dataset and convert to numpy/MindSpore compatible format.
        According to mindspore-note.md #1 and mindspore-exemptions.md #4:
        - Convert torch tensors to numpy arrays
        - Handle PIL images to array conversion
        - Apply necessary format conversions
        """
        # Get item from PyTorch dataset
        item = self.torch_dataset[idx]
        
        if isinstance(item, tuple):
            # Handle (data, label) format
            data, label = item
            
            # Convert data
            if hasattr(data, 'numpy'):  # torch.Tensor
                data = data.numpy()
            elif hasattr(data, 'mode'):  # PIL Image
                # Convert PIL to numpy array (HWC format)
                data = np.array(data, dtype=np.float32)
                # If image is grayscale, add channel dimension
                if len(data.shape) == 2:
                    data = np.expand_dims(data, axis=2)
                # Convert HWC to CHW (like ToTensor)
                data = np.transpose(data, (2, 0, 1))
                # Scale to [0, 1] (like ToTensor)
                data = data / 255.0
            elif isinstance(data, np.ndarray):
                # Already numpy array, ensure float32
                data = data.astype(np.float32)
            
            # Convert label
            if hasattr(label, 'numpy'):  # torch.Tensor
                label = label.numpy()
            elif isinstance(label, (int, np.integer)):
                label = np.array(label, dtype=np.int32)
            elif isinstance(label, np.ndarray):
                label = label.astype(np.int32)
            
            return data, label
        else:
            # Single item return
            if hasattr(item, 'numpy'):  # torch.Tensor
                return item.numpy()
            elif hasattr(item, 'mode'):  # PIL Image
                # Convert PIL to numpy array
                data = np.array(item, dtype=np.float32)
                if len(data.shape) == 2:
                    data = np.expand_dims(data, axis=2)
                data = np.transpose(data, (2, 0, 1))
                data = data / 255.0
                return data
            else:
                return item


class DatasetGenerator:
    """
    MindSpore implementation of DatasetGenerator.
    Maintains compatibility with PyTorch datasets through TorchDatasetWrapper.
    """
    def __init__(self, exp, train_bs=128, eval_bs=256, seed=0, n_workers=4,
                 train_d_type='CIFAR10', test_d_type='CIFAR10',
                 train_path='../../datasets/', test_path='../../datasets/',
                 train_tf_op=None, test_tf_op=None,
                 **kwargs):
        
        np.random.seed(seed)
        
        if train_d_type not in dataset_options:
            print(train_d_type)
            raise TypeError('Unknown Dataset')  # Using TypeError per mindspore-exemptions.md #1
        elif test_d_type not in dataset_options:
            print(test_d_type)
            raise TypeError('Unknown Dataset')  # Using TypeError per mindspore-exemptions.md #1
        
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers
        self.train_path = train_path
        self.test_path = test_path
        
        # Get transforms from PyTorch implementation
        # Note: We keep using torchvision transforms as per mindspore-note.md #5
        # The transforms will be applied within the PyTorch dataset
        train_tf = transform_options[train_tf_op]["train_transform"]
        test_tf = transform_options[test_tf_op]["test_transform"]
        
        # Import transforms from torchvision to compose them
        from torchvision import transforms
        if train_tf is not None:
            train_tf = transforms.Compose(train_tf)
        if test_tf is not None:
            test_tf = transforms.Compose(test_tf)
        
        # Create PyTorch datasets and wrap them
        self.poison_test_set = None
        if 'poison_test_d_type' in kwargs:
            d_type = kwargs['poison_test_d_type']
            torch_poison_test = dataset_options[d_type](test_path, test_tf, True, kwargs)
            self.poison_test_set = TorchDatasetWrapper(torch_poison_test)
        
        # Create train and test datasets using PyTorch implementations
        torch_train = dataset_options[train_d_type](train_path, train_tf, False, kwargs)
        torch_test = dataset_options[test_d_type](test_path, test_tf, True, kwargs)
        
        # Wrap with TorchDatasetWrapper for MindSpore compatibility
        self.train_set = TorchDatasetWrapper(torch_train)
        self.test_set = TorchDatasetWrapper(torch_test)
        
        self.train_set_length = len(self.train_set)
        self.test_set_length = len(self.test_set)
    
    def get_loader(self, train_shuffle=True, drop_last=False, train_sampler=None, test_sampler=None,
                   sampler_bd_val=None):
        """
        Get MindSpore DataLoaders.
        Note: MindSpore's dataset API works differently from PyTorch DataLoader.
        """
        poison_test_loader = None
        
        # Create MindSpore datasets from wrapped PyTorch datasets
        # Note: MindSpore uses different API for creating datasets
        
        if train_shuffle is False or train_sampler is None:
            # Create train loader
            train_dataset = ds.GeneratorDataset(
                source=self.train_set,
                column_names=['data', 'label'],
                num_parallel_workers=self.n_workers,
                shuffle=train_shuffle
            )
            train_dataset = train_dataset.batch(
                batch_size=self.train_bs,
                drop_remainder=drop_last
            )
            
            # Create test loader
            test_dataset = ds.GeneratorDataset(
                source=self.test_set,
                column_names=['data', 'label'],
                num_parallel_workers=self.n_workers,
                shuffle=False
            )
            test_dataset = test_dataset.batch(
                batch_size=self.eval_bs,
                drop_remainder=drop_last
            )
            
            # Create poison test loader if needed
            if self.poison_test_set is not None:
                poison_test_dataset = ds.GeneratorDataset(
                    source=self.poison_test_set,
                    column_names=['data', 'label'],
                    num_parallel_workers=self.n_workers,
                    shuffle=False
                )
                poison_test_dataset = poison_test_dataset.batch(
                    batch_size=self.eval_bs,
                    drop_remainder=False
                )
                poison_test_loader = poison_test_dataset
        else:
            # MindSpore doesn't have direct sampler support like PyTorch
            # We'll need to implement custom sampling if needed
            # For now, raising NotImplementedError for sampler cases
            raise NotImplementedError("Custom samplers not yet implemented in MindSpore version")
        
        # Return the datasets (MindSpore uses dataset objects instead of DataLoader)
        train_loader = train_dataset
        test_loader = test_dataset
        
        return train_loader, test_loader, poison_test_loader