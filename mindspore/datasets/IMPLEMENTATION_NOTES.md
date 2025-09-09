# MindSpore CIFAR BadNet Dataset Implementation Notes

## Overview
Successfully converted the PyTorch CIFAR BadNet dataset implementation to MindSpore framework while maintaining full compatibility with the original functionality.

## Key Implementation Details

### 1. Architecture
- **Base Class**: Inherits from `torchvision.datasets.CIFAR10` (per mindspore-note.md #5)
- **Wrapper**: Uses `TorchDatasetWrapper` for MindSpore compatibility
- **Location**: `/root/CognitiveDistillation/mindspore/datasets/cifar_badnet.py`

### 2. Dataset Features
- Adds backdoor triggers to CIFAR-10 images
- Configurable poison rate (fraction of samples to backdoor)
- Target label specification for backdoored samples
- Support for full backdoor testing mode
- 3x3 checkerboard pattern trigger in bottom-right corner

### 3. MindSpore Adaptations
Following the project guidelines:
- **Device Management**: Uses MindSpore default device allocation (mindspore-exemptions.md #3)
- **Data Format**: TorchDatasetWrapper handles PyTorch to numpy/MindSpore conversions
- **Tensor Operations**: All backdoor trigger additions done on numpy arrays
- **Framework Boundary**: Clean separation between PyTorch dataset and MindSpore usage

### 4. Key Methods
- `__init__`: Initializes dataset, selects poison indices, adds triggers
- Poison index selection:
  - Training: Random selection based on poison_rate
  - Testing: Selects from non-target class samples
  - Full test mode: All non-target samples poisoned
- Trigger pattern: 3x3 checkerboard in bottom-right corner

### 5. Data Pipeline
1. PyTorch CIFAR10 dataset loads data
2. BadNetCIFAR10 adds backdoor triggers
3. TorchDatasetWrapper converts data:
   - PIL images → numpy arrays
   - HWC → CHW format (like ToTensor)
   - Scale to [0, 1] range
   - Convert labels to int32

### 6. Validation
- Created comprehensive test suite (`test_mindspore_badnet.py`)
- Phase 1 validation test (`test_phase1_badnet.py`)
- All tests pass successfully
- Verified:
  - Correct poison rates
  - Proper trigger patterns
  - Label changes to target
  - Data format conversions
  - MindSpore tensor compatibility

## Usage Example
```python
from datasets.cifar_badnet import BadNetCIFAR10
from datasets.dataset import TorchDatasetWrapper

# Create backdoored dataset
dataset = BadNetCIFAR10(
    root='./data',
    train=False,
    poison_rate=0.01,
    target_label=0
)

# Wrap for MindSpore
wrapped = TorchDatasetWrapper(dataset)

# Get samples
image, label = wrapped[0]  # Returns numpy arrays
```

## Files Modified/Created
- Created: `mindspore/datasets/cifar_badnet.py`
- Modified: `mindspore/datasets/__init__.py` (added BadNetCIFAR10 export)
- Created: Test files for validation

## Compatibility Notes
- Maintains exact functionality of original PyTorch implementation
- Compatible with MindSpore's dataset pipeline through wrapper
- Preserves all backdoor trigger patterns and poisoning logic
- Supports all original parameters and options