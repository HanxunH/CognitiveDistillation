"""
MindSpore datasets module.
"""

import mlconfig
from .dataset import DatasetGenerator, TorchDatasetWrapper
from .cifar_badnet import BadNetCIFAR10
from .cifar_blend import BlendCIFAR10
from .cifar_cl import CLCIFAR10
from .cifar_dynamic import DynamicCIFAR10
from .cifar_wanet import WaNetCIFAR10
from .cifar_dfst import DFSTCIFAR10
from .cifar_fc import FCCIFAR10
from .cifar_nashville import NashvilleCIFAR10
from .issba import ISSBAImageNetClean, ISSBAImageNet

# Register dataset classes with mlconfig (only if not already registered)
try:
    mlconfig.register(DatasetGenerator)
except ValueError:
    # Already registered, skip
    pass

__all__ = ['DatasetGenerator', 'TorchDatasetWrapper', 'BadNetCIFAR10', 'BlendCIFAR10', 'CLCIFAR10', 'DynamicCIFAR10', 'WaNetCIFAR10', 'DFSTCIFAR10', 'FCCIFAR10', 'NashvilleCIFAR10', 'ISSBAImageNetClean', 'ISSBAImageNet']