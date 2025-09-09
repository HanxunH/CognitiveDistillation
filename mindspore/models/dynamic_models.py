"""
MindSpore implementation of dynamic models for backdoor attacks.

Note: According to mindspore-note.md #8, when loading PyTorch pre-trained models,
we keep using PyTorch for inference and convert outputs to MindSpore format at boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import mindspore as ms
import numpy as np
from .blocks import Conv2dBlock, DownSampleBlock, UpSampleBlock


class Normalize:
    """Normalizer for input tensors."""
    
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        """Normalize torch tensor (keeping it as torch for compatibility)."""
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    """Denormalizer for output tensors."""
    
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        """Denormalize torch tensor (keeping it as torch for compatibility)."""
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


# ---------------------------- Generators ----------------------------#

class Generator(nn.Sequential):
    """
    Generator network for creating dynamic triggers.
    
    Note: This class remains as PyTorch nn.Sequential because it's used to load
    pre-trained PyTorch models. According to mindspore-note.md #8, we keep using
    PyTorch for inference with pre-trained models.
    """
    
    def __init__(self, opt, out_channels=None):
        super(Generator, self).__init__()
        if opt.dataset == "mnist":
            channel_init = 16
            steps = 2
        else:
            channel_init = 32
            steps = 3

        # Import PyTorch blocks for compatibility with pre-trained models
        from models.blocks import Conv2dBlock as PTConv2dBlock
        from models.blocks import DownSampleBlock as PTDownSampleBlock
        from models.blocks import UpSampleBlock as PTUpSampleBlock

        channel_current = opt.input_channel
        channel_next = channel_init
        for step in range(steps):
            self.add_module("convblock_down_{}".format(2 * step), PTConv2dBlock(channel_current, channel_next))
            self.add_module("convblock_down_{}".format(2 * step + 1), PTConv2dBlock(channel_next, channel_next))
            self.add_module("downsample_{}".format(step), PTDownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self.add_module("convblock_middle", PTConv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module("upsample_{}".format(step), PTUpSampleBlock())
            self.add_module("convblock_up_{}".format(2 * step), PTConv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), PTConv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self.add_module("convblock_up_{}".format(2 * step + 1), PTConv2dBlock(channel_current, channel_next))
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
        """Forward pass using PyTorch operations."""
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        return x

    def normalize_pattern(self, x):
        if self._normalizer:
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if self._denormalizer:
            x = self._denormalizer(x)
        return x

    def threshold(self, x):
        """Apply thresholding to create binary mask."""
        return nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5


# ---------------------------- Classifiers ----------------------------#

class NetC_MNIST(nn.Module):
    """
    MNIST classifier network.
    
    Note: Kept as PyTorch module for compatibility with pre-trained models.
    """
    
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(0.1)

        self.maxpool4 = nn.MaxPool2d((2, 2))
        self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout(0.1)

        self.maxpool5 = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.1)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x