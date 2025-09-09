import copy
import math
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
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


# Custom implementation of SqueezeExcitation for TensorFlow
class SqueezeExcitation(Model):
    def __init__(self, input_channels, squeeze_channels, activation=None):
        super().__init__()
        self.avgpool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Conv2D(squeeze_channels, kernel_size=1, use_bias=True)
        self.fc2 = layers.Conv2D(input_channels, kernel_size=1, use_bias=True)
        # Using Swish (SiLU) activation as default
        self.activation = activation if activation else tf.nn.swish

    def call(self, x):
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = tf.nn.sigmoid(scale)
        return x * scale


# Custom implementation of ConvNormActivation for TensorFlow
class ConvNormActivation(Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None,
                 groups=1, norm_layer=None, activation_layer=None, dilation=1, inplace=True,
                 bias=None):
        super().__init__()
        
        # Calculate padding if not specified
        if padding is None:
            padding = (kernel_size - 1) // 2
        
        # Handle explicit padding
        self.explicit_padding = None
        if isinstance(padding, int) and padding > 0:
            self.explicit_padding = layers.ZeroPadding2D(padding=padding)
            padding_mode = 'valid'
        else:
            padding_mode = 'same' if padding else 'valid'
        
        # Determine if bias should be used
        if bias is None:
            bias = norm_layer is None
        
        # Create convolution layer
        self.is_depthwise = False
        if groups == 1:
            self.conv = layers.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=stride,
                padding=padding_mode,
                dilation_rate=dilation,
                use_bias=bias
            )
        else:
            # Depthwise convolution (groups == in_channels)
            if groups == in_channels:
                self.is_depthwise = True
                self.conv = layers.DepthwiseConv2D(
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding_mode,
                    dilation_rate=dilation,
                    use_bias=bias
                )
                # DepthwiseConv2D doesn't change the number of channels, 
                # so we need a pointwise conv if out_channels != in_channels
                if out_channels != in_channels:
                    self.pointwise = layers.Conv2D(
                        filters=out_channels,
                        kernel_size=1,
                        strides=1,
                        padding='valid',
                        use_bias=False  # Will be handled by norm layer
                    )
                else:
                    self.pointwise = None
            else:
                # TensorFlow doesn't have direct support for arbitrary groups
                # We'll use a workaround with grouped convolution
                raise NotImplementedError(f"Groups={groups} is not supported. Use 1 or in_channels for depthwise.")
        
        self.norm = norm_layer() if norm_layer else None
        
        # Handle activation layer
        if activation_layer is None:
            self.activation = None
        elif activation_layer == tf.nn.swish or activation_layer.__name__ == 'SiLU':
            self.activation = lambda x: tf.nn.swish(x)
        else:
            self.activation = activation_layer()

    def call(self, x, training=None):
        if self.explicit_padding is not None:
            x = self.explicit_padding(x)
        
        x = self.conv(x)
        
        # Apply pointwise convolution if needed (for depthwise separable conv)
        if self.is_depthwise and hasattr(self, 'pointwise') and self.pointwise is not None:
            x = self.pointwise(x)
        
        if self.norm is not None:
            x = self.norm(x, training=training)
        
        if self.activation is not None:
            x = self.activation(x)
        
        return x


# Custom implementation of StochasticDepth for TensorFlow
class StochasticDepth(Model):
    def __init__(self, drop_prob: float, mode: str = "row"):
        super().__init__()
        self.drop_prob = drop_prob
        self.mode = mode

    def call(self, x, training=None):
        if not training or self.drop_prob == 0.:
            return x
        
        keep_prob = 1 - self.drop_prob
        
        if self.mode == "row":
            # Get shape of input
            shape = tf.shape(x)
            # Create random tensor with shape (batch_size, 1, 1, 1)
            random_tensor = keep_prob + tf.random.uniform([shape[0], 1, 1, 1], dtype=x.dtype)
            # Binary mask
            binary_tensor = tf.floor(random_tensor)
            # Apply stochastic depth
            output = x / keep_prob * binary_tensor
        else:
            # Batch mode
            random_tensor = keep_prob + tf.random.uniform([], dtype=x.dtype)
            binary_tensor = tf.floor(random_tensor)
            output = x / keep_prob * binary_tensor
        
        return output


class MBConv(Model):
    def __init__(self, cnf: MBConvConfig, stochastic_depth_prob: float, norm_layer, se_layer=SqueezeExcitation):
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers_list = []
        
        # Using Swish (SiLU) activation
        activation_layer = tf.nn.swish

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers_list.append(
                ConvNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers_list.append(
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
        layers_list.append(se_layer(expanded_channels, squeeze_channels, activation=tf.nn.swish))

        # project
        layers_list.append(
            ConvNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = Sequential(layers_list)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def call(self, input, training=None):
        result = self.block(input, training=training)
        if self.use_res_connect:
            result = self.stochastic_depth(result, training=training)
            result += input
        return result


class EfficientNet(Model):
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
            norm_layer = layers.BatchNormalization

        layers_list = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers_list.append(
            ConvNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, 
                activation_layer=tf.nn.swish
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

            layers_list.append(Sequential(stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers_list.append(
            ConvNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=tf.nn.swish,
            )
        )

        self.features = Sequential(layers_list)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.classifier = Sequential([
            layers.Dropout(rate=dropout),
            layers.Dense(num_classes),
        ])

        # Initialize get_features attribute
        self.get_features = False

    def build(self, input_shape):
        super().build(input_shape)
        # Initialize weights similar to PyTorch
        for layer in self.layers:
            if isinstance(layer, Sequential):
                for sublayer in layer.layers:
                    self._init_weights(sublayer)
            else:
                self._init_weights(layer)

    def _init_weights(self, layer):
        """Initialize weights similar to PyTorch implementation"""
        if isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            # Kaiming normal initialization (He initialization in TensorFlow)
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = tf.keras.initializers.HeNormal()
            if hasattr(layer, 'bias_initializer') and layer.use_bias:
                layer.bias_initializer = tf.keras.initializers.Zeros()
        elif isinstance(layer, layers.BatchNormalization):
            if hasattr(layer, 'gamma_initializer'):
                layer.gamma_initializer = tf.keras.initializers.Ones()
            if hasattr(layer, 'beta_initializer'):
                layer.beta_initializer = tf.keras.initializers.Zeros()
        elif isinstance(layer, layers.Dense):
            # Uniform initialization similar to PyTorch
            if hasattr(layer, 'units'):
                init_range = 1.0 / math.sqrt(layer.units)
                layer.kernel_initializer = tf.keras.initializers.RandomUniform(
                    minval=-init_range, maxval=init_range
                )
                layer.bias_initializer = tf.keras.initializers.Zeros()

    def _forward_impl(self, x, training=None):
        features = []
        x = self.features(x, training=training)
        features.append(x)
        x = self.avgpool(x)
        # x is already flattened by GlobalAveragePooling2D
        features.append(x)
        x = self.classifier(x, training=training)
        if self.get_features:
            return features, x
        return x

    def call(self, x, training=None):
        return self._forward_impl(x, training=training)


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