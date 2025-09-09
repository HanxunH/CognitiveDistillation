# MindSpore Dataset Implementation

This directory contains the MindSpore implementation of the dataset module, converted from the PyTorch implementation.

## Files

- `dataset.py`: Main dataset generator and wrapper classes
  - `DatasetGenerator`: Creates and manages train/test datasets with specified configurations
  - `TorchDatasetWrapper`: Wraps PyTorch datasets for MindSpore compatibility

- `utils.py`: Re-exports utilities from PyTorch implementation
  - `transform_options`: Available data transformations
  - `dataset_options`: Available dataset types
  - `get_classidx`: Helper for getting class indices
  - `Cutout`: Data augmentation class

- `pytorch_imports.py`: Helper module that directly imports PyTorch dataset utilities to avoid circular imports

- `__init__.py`: Package initialization

## Key Design Decisions

1. **PyTorch Dataset Compatibility**: Following mindspore-note.md #5, we use a unified `TorchDatasetWrapper` to wrap PyTorch datasets rather than reimplementing each dataset from scratch.

2. **Data Format Conversion**: The wrapper handles conversion from PyTorch tensors/PIL images to numpy arrays in `__getitem__`, which are then converted to MindSpore tensors by the DataLoader.

3. **Transform Preservation**: We continue using torchvision transforms as they operate on PIL images and are framework-agnostic (per mindspore-exemptions.md #8).

4. **Lazy Loading**: Dataset classes are loaded on-demand to avoid import errors for datasets that may not be installed.

## Usage

```python
from mindspore.datasets.dataset import DatasetGenerator

# Create dataset generator
dataset_gen = DatasetGenerator(
    exp=None,
    train_bs=128,
    eval_bs=256,
    seed=42,
    n_workers=4,
    train_d_type='CIFAR10',
    test_d_type='CIFAR10',
    train_path='./data/',
    test_path='./data/',
    train_tf_op='NoAug',
    test_tf_op='NoAug'
)

# Get MindSpore DataLoaders
train_loader, test_loader, poison_loader = dataset_gen.get_loader(
    train_shuffle=True,
    drop_last=False
)

# Iterate through batches
for batch in train_loader.create_dict_iterator():
    images = batch['data']  # MindSpore tensor
    labels = batch['label']  # MindSpore tensor
    # Process batch...
```

## Supported Datasets

All 28 dataset types from the PyTorch implementation are supported, including:
- Standard datasets: CIFAR10, CIFAR100, MNIST, SVHN, ImageNet, GTSRB
- Backdoor attack datasets: BadNetCIFAR10, BlendCIFAR10, TrojanCIFAR10, etc.
- Custom datasets: CustomCIFAR10, CustomCelebA, MIXED_MNIST, etc.

## Testing

Run the test scripts to verify functionality:
```bash
python test_phase1_mindspore.py  # Basic functionality test
python test_mindspore_dataset_final.py  # Comprehensive test with model integration
```