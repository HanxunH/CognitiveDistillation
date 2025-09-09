import copy
import math
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from functools import partial
from typing import Sequence


def _make_divisible(v: float, divisor: int, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Cell):
    """Squeeze-and-Excitation module implementation for MindSpore"""
    def __init__(self, input_channels: int, squeeze_channels: int, activation=None):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.activation = activation if activation is not None else nn.ReLU()
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class ConvNormActivation(nn.Cell):
    """Conv-BN-Activation block implementation for MindSpore"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        groups: int = 1,
        norm_layer=None,
        activation_layer=None,
        dilation: int = 1,
        inplace: bool = True,
        bias: bool = None,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
            
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            pad_mode='pad',
            padding=padding,
            group=groups,
            dilation=dilation,
            has_bias=bias
        )
        
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.activation = activation_layer() if activation_layer is not None else None
        
    def construct(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class StochasticDepth(nn.Cell):
    """Stochastic Depth (DropPath) implementation for MindSpore"""
    def __init__(self, p: float = 0.5, mode: str = "row"):
        super().__init__()
        self.p = p
        self.mode = mode
        
    def construct(self, x):
        if not self.training or self.p == 0.0:
            return x
            
        # Calculate keep probability
        keep_prob = 1 - self.p
        
        # Generate random tensor based on mode
        if self.mode == "row":
            # For "row" mode, we drop entire samples in the batch
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        else:
            shape = x.shape
            
        # Generate random mask
        random_tensor = keep_prob + ops.uniform(shape, ms.Tensor(0.0, ms.float32), ms.Tensor(1.0, ms.float32))
        random_tensor = ops.floor(random_tensor)
        random_tensor = ops.cast(random_tensor, x.dtype)
        
        # Apply stochastic depth
        output = x / keep_prob * random_tensor
        return output


class SiLU(nn.Cell):
    """SiLU (Swish) activation function for MindSpore"""
    def __init__(self, inplace=False):
        super().__init__()
        # MindSpore doesn't have inplace operations, so we ignore the inplace parameter
        self.sigmoid = nn.Sigmoid()
        
    def construct(self, x):
        return x * self.sigmoid(x)


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self, expand_ratio: float, kernel: int, stride: int, input_channels: int, out_channels: int,
                 num_layers: int, width_mult: float, depth_mult: float, ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"expand_ratio={self.expand_ratio}"
            f", kernel={self.kernel}"
            f", stride={self.stride}"
            f", input_channels={self.input_channels}"
            f", out_channels={self.out_channels}"
            f", num_layers={self.num_layers}"
            f")"
        )
        return s

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value=None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Cell):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer, se_layer=SqueezeExcitation):
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            ConvNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=SiLU(inplace=True)))

        # project
        layers.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.SequentialCell(layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def construct(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result = result + input
        return result


class EfficientNet(nn.Cell):
    def __init__(self, inverted_residual_setting, dropout: float, stochastic_depth_prob: float = 0.2,
                 num_classes: int = 1000, block=None, norm_layer=None, **kwargs):
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=SiLU
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.SequentialCell(stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=SiLU,
            )
        )

        self.features = nn.SequentialCell(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.SequentialCell([
            nn.Dropout(p=dropout),
            nn.Dense(lastconv_output_channels, num_classes),
        ])

        # Initialize weights
        import numpy as np
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                # Kaiming normal initialization
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                std = math.sqrt(2.0 / fan_out)
                cell.weight.set_data(ms.Tensor(np.random.normal(0, std, cell.weight.shape), ms.float32))
                if cell.has_bias:
                    cell.bias.set_data(ms.Tensor(np.zeros(cell.bias.shape), ms.float32))
            elif isinstance(cell, (nn.BatchNorm2d, nn.GroupNorm)):
                cell.gamma.set_data(ms.Tensor(np.ones(cell.gamma.shape), ms.float32))
                cell.beta.set_data(ms.Tensor(np.zeros(cell.beta.shape), ms.float32))
            elif isinstance(cell, nn.Dense):
                init_range = 1.0 / math.sqrt(cell.out_channels)
                cell.weight.set_data(ms.Tensor(np.random.uniform(-init_range, init_range, cell.weight.shape), ms.float32))
                cell.bias.set_data(ms.Tensor(np.zeros(cell.bias.shape), ms.float32))

        self.get_features = False

    def _forward_impl(self, x):
        features = []
        x = self.features(x)
        features.append(x)
        x = self.avgpool(x)
        # Flatten the tensor
        x = ops.reshape(x, (x.shape[0], -1))
        features.append(x)
        x = self.classifier(x)
        if self.get_features:
            return features, x
        return x

    def construct(self, x):
        return self._forward_impl(x)


def EfficientNetB0(num_classes=200, **kwargs):
    bneck_conf = partial(MBConvConfig, width_mult=1.0, depth_mult=1.0)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1),
        bneck_conf(6, 3, 2, 16, 24, 2),
        bneck_conf(6, 5, 2, 24, 40, 2),
        bneck_conf(6, 3, 2, 40, 80, 3),
        bneck_conf(6, 5, 1, 80, 112, 3),
        bneck_conf(6, 5, 2, 112, 192, 4),
        bneck_conf(6, 3, 1, 192, 320, 1),
    ]
    last_channel = None
    return EfficientNet(inverted_residual_setting, 0.0, last_channel=last_channel, num_classes=num_classes, **kwargs)