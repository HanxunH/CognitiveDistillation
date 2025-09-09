"""
MindSpore implementation of basic building blocks for neural networks.
"""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Conv2dBlock(nn.Cell):
    """Convolutional block with optional batch normalization and ReLU."""
    
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, pad_mode='pad', padding=padding)
        self.batch_norm = None
        self.relu = None
        
        if batch_norm:
            # MindSpore BatchNorm2d has different momentum definition than PyTorch
            # PyTorch: running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            # MindSpore: running_mean = momentum * running_mean + (1 - momentum) * batch_mean
            # So we need to use 1 - pytorch_momentum = 0.95 for pytorch momentum of 0.05
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.95, affine=True)
        if relu:
            self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv2d(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DownSampleBlock(nn.Cell):
    """Downsampling block with max pooling and optional dropout."""
    
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        # Note: MindSpore's MaxPool2d doesn't support dilation parameter
        # According to mindspore-exemptions.md #11, we skip dilation if it's default (1,1)
        if dilation != (1, 1):
            raise NotImplementedError("MindSpore MaxPool2d doesn't support dilation parameter")
        
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride)
        self.dropout = None
        if p:
            self.dropout = nn.Dropout(p)
    
    def construct(self, x):
        x = self.maxpooling(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UpSampleBlock(nn.Cell):
    """Upsampling block with bilinear/nearest interpolation and optional dropout."""
    
    def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.dropout = None
        if p:
            self.dropout = nn.Dropout(p)
        
        # Create the resize operation based on mode
        if mode == "bilinear":
            self.resize = ops.ResizeBilinear(align_corners=False)
        elif mode == "nearest":
            self.resize = ops.ResizeNearestNeighbor(align_corners=False)
        else:
            raise ValueError(f"Unsupported upsample mode: {mode}")
    
    def construct(self, x):
        # Calculate output size based on scale factor
        _, _, h, w = x.shape
        new_h = int(h * self.scale_factor[0])
        new_w = int(w * self.scale_factor[1])
        
        # Apply resize operation
        x = self.resize(x, (new_h, new_w))
        
        if self.dropout is not None:
            x = self.dropout(x)
        return x