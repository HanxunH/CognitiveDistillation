import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision
from torch import nn as torch_nn
from torchvision import transforms
import numpy as np

from .blocks import *


class Normalize:
    """Normalize input tensors based on expected values and variance."""
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        # Note: According to tensorflow-note.md item 8, we keep using PyTorch for pretrained models
        # This class will be used with PyTorch tensors internally
        if isinstance(x, torch.Tensor):
            x_clone = x.clone()
            for channel in range(self.n_channels):
                x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
            return x_clone
        else:
            # If TensorFlow tensor is passed, convert to PyTorch, process, and convert back
            x_np = x.numpy() if hasattr(x, 'numpy') else x
            x_torch = torch.from_numpy(x_np)
            x_clone = x_torch.clone()
            for channel in range(self.n_channels):
                x_clone[:, channel] = (x_torch[:, channel] - self.expected_values[channel]) / self.variance[channel]
            return tf.convert_to_tensor(x_clone.numpy())


class Denormalize:
    """Denormalize input tensors based on expected values and variance."""
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        # Note: According to tensorflow-note.md item 8, we keep using PyTorch for pretrained models
        # This class will be used with PyTorch tensors internally
        if isinstance(x, torch.Tensor):
            x_clone = x.clone()
            for channel in range(self.n_channels):
                x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
            return x_clone
        else:
            # If TensorFlow tensor is passed, convert to PyTorch, process, and convert back
            x_np = x.numpy() if hasattr(x, 'numpy') else x
            x_torch = torch.from_numpy(x_np)
            x_clone = x_torch.clone()
            for channel in range(self.n_channels):
                x_clone[:, channel] = x_torch[:, channel] * self.variance[channel] + self.expected_values[channel]
            return tf.convert_to_tensor(x_clone.numpy())


# ---------------------------- Generators ----------------------------#


class Generator:
    """
    Generator model for dynamic trigger generation.
    
    Note: According to tensorflow-note.md item 8, since this uses pretrained PyTorch models,
    we keep the PyTorch implementation and convert outputs to TensorFlow tensors at the interface.
    """
    def __init__(self, opt, out_channels=None):
        # We need to create PyTorch blocks internally
        # First, let's define PyTorch versions of the blocks here
        class PyTorchConv2dBlock(torch_nn.Module):
            def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
                super(PyTorchConv2dBlock, self).__init__()
                self.conv2d = torch_nn.Conv2d(in_c, out_c, ker_size, stride, padding)
                if batch_norm:
                    self.batch_norm = torch_nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
                if relu:
                    self.relu = torch_nn.ReLU(inplace=True)

            def forward(self, x):
                for module in self.children():
                    x = module(x)
                return x

        class PyTorchDownSampleBlock(torch_nn.Module):
            def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
                super(PyTorchDownSampleBlock, self).__init__()
                self.maxpooling = torch_nn.MaxPool2d(kernel_size=ker_size, stride=stride,
                                               dilation=dilation, ceil_mode=ceil_mode)
                if p:
                    self.dropout = torch_nn.Dropout(p)

            def forward(self, x):
                for module in self.children():
                    x = module(x)
                return x

        class PyTorchUpSampleBlock(torch_nn.Module):
            def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
                super(PyTorchUpSampleBlock, self).__init__()
                self.upsample = torch_nn.Upsample(scale_factor=scale_factor, mode=mode)
                if p:
                    self.dropout = torch_nn.Dropout(p)

            def forward(self, x):
                for module in self.children():
                    x = module(x)
                return x
        
        # Create the PyTorch model internally
        self._pytorch_model = torch_nn.Sequential()
        
        if opt.dataset == "mnist":
            channel_init = 16
            steps = 2
        else:
            channel_init = 32
            steps = 3

        channel_current = opt.input_channel
        channel_next = channel_init
        
        for step in range(steps):
            self._pytorch_model.add_module("convblock_down_{}".format(2 * step), 
                                         PyTorchConv2dBlock(channel_current, channel_next))
            self._pytorch_model.add_module("convblock_down_{}".format(2 * step + 1), 
                                         PyTorchConv2dBlock(channel_next, channel_next))
            self._pytorch_model.add_module("downsample_{}".format(step), 
                                         PyTorchDownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self._pytorch_model.add_module("convblock_middle", 
                                      PyTorchConv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self._pytorch_model.add_module("upsample_{}".format(step), 
                                         PyTorchUpSampleBlock())
            self._pytorch_model.add_module("convblock_up_{}".format(2 * step), 
                                         PyTorchConv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self._pytorch_model.add_module(
                    "convblock_up_{}".format(2 * step + 1), 
                    PyTorchConv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self._pytorch_model.add_module("convblock_up_{}".format(2 * step + 1), 
                                             PyTorchConv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
            if step == steps - 2:
                if out_channels is None:
                    channel_next = opt.input_channel
                else:
                    channel_next = out_channels

        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(opt)
        self._denormalizer = self._get_denormalize(opt)

    def _get_denormalize(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def forward(self, x):
        """Forward pass through the generator."""
        # Convert TensorFlow tensor to PyTorch tensor if needed
        if isinstance(x, tf.Tensor):
            x_np = x.numpy()
            x = torch.from_numpy(x_np)
        
        # Run through PyTorch model
        for module in self._pytorch_model.children():
            x = module(x)
        x = torch_nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        
        # Convert back to TensorFlow tensor (move to CPU first if needed)
        return tf.convert_to_tensor(x.detach().cpu().numpy())
    
    def __call__(self, x):
        """Make the class callable like the original PyTorch module."""
        return self.forward(x)

    def normalize_pattern(self, x):
        """Normalize the pattern using the configured normalizer."""
        if self._normalizer:
            # The normalizer handles both PyTorch and TensorFlow tensors
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        """Denormalize the pattern using the configured denormalizer."""
        if self._denormalizer:
            # The denormalizer handles both PyTorch and TensorFlow tensors
            x = self._denormalizer(x)
        return x

    def threshold(self, x):
        """Apply threshold function to the input."""
        if isinstance(x, tf.Tensor):
            # TensorFlow implementation
            return tf.nn.tanh(x * 20 - 10) / (2 + self._EPSILON) + 0.5
        else:
            # PyTorch implementation for backward compatibility
            return torch_nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5

    def eval(self):
        """Set the model to evaluation mode."""
        self._pytorch_model.eval()
        return self
    
    def train(self):
        """Set the model to training mode."""
        self._pytorch_model.train()
        return self
    
    def to(self, device):
        """Move model to specified device (for PyTorch compatibility)."""
        self._pytorch_model.to(device)
        return self
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict (for PyTorch compatibility)."""
        self._pytorch_model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self):
        """Get state dict (for PyTorch compatibility)."""
        return self._pytorch_model.state_dict()
    
    def parameters(self):
        """Get model parameters (for PyTorch compatibility)."""
        return self._pytorch_model.parameters()


# ---------------------------- Classifiers ----------------------------#


class NetC_MNIST:
    """
    MNIST classifier model.
    
    Note: According to tensorflow-note.md item 8, since this uses pretrained PyTorch models,
    we keep the PyTorch implementation and convert outputs to TensorFlow tensors at the interface.
    """
    def __init__(self):
        # Create the PyTorch model internally
        self._pytorch_model = torch_nn.Module()
        self._pytorch_model.conv1 = torch_nn.Conv2d(1, 32, (5, 5), 1, 0)
        self._pytorch_model.relu2 = torch_nn.ReLU(inplace=True)
        self._pytorch_model.dropout3 = torch_nn.Dropout(0.1)
        
        self._pytorch_model.maxpool4 = torch_nn.MaxPool2d((2, 2))
        self._pytorch_model.conv5 = torch_nn.Conv2d(32, 64, (5, 5), 1, 0)
        self._pytorch_model.relu6 = torch_nn.ReLU(inplace=True)
        self._pytorch_model.dropout7 = torch_nn.Dropout(0.1)
        
        self._pytorch_model.maxpool5 = torch_nn.MaxPool2d((2, 2))
        self._pytorch_model.flatten = torch_nn.Flatten()
        self._pytorch_model.linear6 = torch_nn.Linear(64 * 4 * 4, 512)
        self._pytorch_model.relu7 = torch_nn.ReLU(inplace=True)
        self._pytorch_model.dropout8 = torch_nn.Dropout(0.1)
        self._pytorch_model.linear9 = torch_nn.Linear(512, 10)
        
        # Store modules in order for forward pass
        self._modules_list = [
            self._pytorch_model.conv1,
            self._pytorch_model.relu2,
            self._pytorch_model.dropout3,
            self._pytorch_model.maxpool4,
            self._pytorch_model.conv5,
            self._pytorch_model.relu6,
            self._pytorch_model.dropout7,
            self._pytorch_model.maxpool5,
            self._pytorch_model.flatten,
            self._pytorch_model.linear6,
            self._pytorch_model.relu7,
            self._pytorch_model.dropout8,
            self._pytorch_model.linear9
        ]

    def forward(self, x):
        """Forward pass through the classifier."""
        # Convert TensorFlow tensor to PyTorch tensor if needed
        if isinstance(x, tf.Tensor):
            x_np = x.numpy()
            x = torch.from_numpy(x_np)
        
        # Run through PyTorch model modules in order
        for module in self._modules_list:
            x = module(x)
        
        # Convert back to TensorFlow tensor (move to CPU first if needed)
        return tf.convert_to_tensor(x.detach().cpu().numpy())
    
    def __call__(self, x):
        """Make the class callable like the original PyTorch module."""
        return self.forward(x)
    
    def eval(self):
        """Set the model to evaluation mode."""
        self._pytorch_model.eval()
        return self
    
    def train(self):
        """Set the model to training mode."""
        self._pytorch_model.train()
        return self
    
    def to(self, device):
        """Move model to specified device (for PyTorch compatibility)."""
        self._pytorch_model.to(device)
        return self
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict (for PyTorch compatibility)."""
        self._pytorch_model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self):
        """Get state dict (for PyTorch compatibility)."""
        return self._pytorch_model.state_dict()
    
    def parameters(self):
        """Get model parameters (for PyTorch compatibility)."""
        return self._pytorch_model.parameters()