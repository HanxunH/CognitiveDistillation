import numpy as np
import tensorflow as tf
from PIL import Image

from .utils import transform_options, dataset_options
from torchvision import transforms


class TorchDatasetWrapper:
    """Wrapper for PyTorch datasets to handle data format conversions."""
    
    def __init__(self, torch_dataset):
        self.torch_dataset = torch_dataset
        
    def __len__(self):
        return len(self.torch_dataset)
    
    def __getitem__(self, idx):
        """Get item with format conversion from PyTorch to TensorFlow."""
        # Get data from PyTorch dataset
        data = self.torch_dataset[idx]
        
        # Handle different return formats
        if isinstance(data, tuple) and len(data) == 2:
            image, label = data
        else:
            raise ValueError(f"Unexpected data format from dataset: {type(data)}")
        
        # Convert image to numpy array if needed
        if hasattr(image, 'numpy'):  # torch.Tensor
            image = image.numpy()
        elif isinstance(image, Image.Image):  # PIL Image
            # Convert PIL to numpy array and normalize to [0, 1]
            image = np.array(image, dtype=np.float32) / 255.0
            # Transpose from HWC to CHW to match ToTensor behavior
            if len(image.shape) == 3:
                image = np.transpose(image, (2, 0, 1))
            elif len(image.shape) == 2:
                # Grayscale image, add channel dimension
                image = np.expand_dims(image, axis=0)
        elif isinstance(image, np.ndarray):  # Raw numpy array (from CIFAR datasets)
            # Check if it's uint8 raw data that needs normalization
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
            # Transpose from HWC to CHW if needed
            if len(image.shape) == 3 and image.shape[-1] in [1, 3]:
                image = np.transpose(image, (2, 0, 1))
            elif len(image.shape) == 2:
                # Grayscale image, add channel dimension
                image = np.expand_dims(image, axis=0)
        
        # Convert label to numpy if needed
        if hasattr(label, 'numpy'):
            label = label.numpy()
        elif hasattr(label, 'item'):
            # Only use item() for scalar tensors
            try:
                label = label.item()
            except (ValueError, RuntimeError):
                # If it's not a scalar, convert to numpy array
                label = np.array(label)
        elif not isinstance(label, np.ndarray):
            # Convert to numpy array if it's not already
            label = np.array(label)
        
        # Ensure proper data types
        image = image.astype(np.float32)
        # Only convert to int64 if it's numeric
        if isinstance(label, (int, float, np.number)):
            label = np.int64(label)
        elif isinstance(label, np.ndarray):
            if label.dtype in [np.float32, np.float64]:
                label = label.astype(np.int64)
            elif label.dtype not in [np.int32, np.int64]:
                label = label.astype(np.int64)
        
        return image, label


def build_tf_dataset(dataset_wrapper, batch_size, shuffle=False, drop_remainder=False):
    """Build a tf.data.Dataset from a dataset wrapper."""
    
    def generator():
        """Generator function for tf.data.Dataset.from_generator."""
        indices = np.arange(len(dataset_wrapper))
        if shuffle:
            np.random.shuffle(indices)
        
        for idx in indices:
            yield dataset_wrapper[idx]
    
    # Determine output signature
    # Get a sample to determine shapes
    sample_image, sample_label = dataset_wrapper[0]
    
    # Handle both scalar and multi-dimensional labels
    if isinstance(sample_label, np.ndarray) and sample_label.shape:
        label_shape = sample_label.shape
    else:
        label_shape = ()  # Scalar
    
    output_signature = (
        tf.TensorSpec(shape=sample_image.shape, dtype=tf.float32),
        tf.TensorSpec(shape=label_shape, dtype=tf.int64)
    )
    
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


class DatasetGenerator:
    def __init__(self, exp, train_bs=128, eval_bs=256, seed=0, n_workers=4,
                 train_d_type='CIFAR10', test_d_type='CIFAR10',
                 train_path='../../datasets/', test_path='../../datasets/',
                 train_tf_op=None, test_tf_op=None,
                 **kwargs):
        
        np.random.seed(seed)
        
        if train_d_type not in dataset_options:
            print(train_d_type)
            raise TypeError('Unknown Dataset')  # Match PyTorch's raise() behavior
        elif test_d_type not in dataset_options:
            print(test_d_type)
            raise TypeError('Unknown Dataset')  # Match PyTorch's raise() behavior
        
        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers  # Keep for API compatibility, but not used in TF
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
            poison_test_torch = dataset_options[d_type](test_path, test_tf, True, kwargs)
            self.poison_test_set = TorchDatasetWrapper(poison_test_torch)
        
        # Create PyTorch datasets and wrap them
        train_torch = dataset_options[train_d_type](train_path, train_tf, False, kwargs)
        test_torch = dataset_options[test_d_type](test_path, test_tf, True, kwargs)
        
        self.train_set = TorchDatasetWrapper(train_torch)
        self.test_set = TorchDatasetWrapper(test_torch)
        self.train_set_length = len(self.train_set)
        self.test_set_length = len(self.test_set)
    
    def get_loader(self, train_shuffle=True, drop_last=False, train_sampler=None, test_sampler=None,
                   sampler_bd_val=None):
        poison_test_loader = None
        
        # Note: TensorFlow doesn't use samplers the same way PyTorch does
        # We'll implement shuffle directly in the dataset creation
        if train_sampler is not None:
            # With sampler, we don't shuffle
            train_shuffle_actual = False
        else:
            train_shuffle_actual = train_shuffle
        
        # Create TensorFlow datasets
        train_loader = build_tf_dataset(
            self.train_set,
            batch_size=self.train_bs,
            shuffle=train_shuffle_actual,
            drop_remainder=drop_last
        )
        
        test_loader = build_tf_dataset(
            self.test_set,
            batch_size=self.eval_bs,
            shuffle=False,
            drop_remainder=drop_last
        )
        
        if self.poison_test_set is not None:
            poison_test_loader = build_tf_dataset(
                self.poison_test_set,
                batch_size=self.eval_bs,
                shuffle=False,
                drop_remainder=False
            )
        
        return train_loader, test_loader, poison_test_loader