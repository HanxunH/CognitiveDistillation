"""
MindSpore datasets utilities.
This module imports and re-exports utilities from the PyTorch implementation
to maintain compatibility while using MindSpore framework.
"""

# Import from our local pytorch_imports module to avoid circular imports
from .pytorch_imports import (
    transform_options,
    dataset_options,
    get_classidx,
    Cutout
)

# Re-export all imported items
__all__ = [
    'transform_options',
    'dataset_options',
    'get_classidx',
    'Cutout'
]