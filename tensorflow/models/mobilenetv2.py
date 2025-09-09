'''MobileNetV2 in TensorFlow.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class Block(Model):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = layers.Conv2D(planes, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        
        # Depthwise convolution - PyTorch uses groups=planes, TensorFlow has DepthwiseConv2D
        # For depthwise convolution, we need to use DepthwiseConv2D which expects input channels = groups
        # and produces input_channels * depth_multiplier output channels
        # Since we want planes output channels and have planes input channels, depth_multiplier = 1
        self.conv2 = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.conv3 = layers.Conv2D(out_planes, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.bn3 = layers.BatchNormalization()

        # Shortcut connection
        if stride == 1 and in_planes != out_planes:
            self.shortcut = Sequential([
                layers.Conv2D(out_planes, kernel_size=1, strides=1, padding='valid', use_bias=False),
                layers.BatchNormalization(),
            ])
        else:
            self.shortcut = None

    def call(self, x, training=None):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        out = self.bn3(self.conv3(out), training=training)
        
        # Apply shortcut if stride is 1
        if self.stride == 1:
            if self.shortcut is not None:
                out = out + self.shortcut(x, training=training)
            else:
                out = out + x
        
        return out


class MobileNetV2(Model):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layers_blocks = self._make_layers(in_planes=32)
        self.conv2 = layers.Conv2D(1280, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.linear = layers.Dense(num_classes)
        self.get_features = False

    def _make_layers(self, in_planes):
        layers_list = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers_list.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return Sequential(layers_list)

    def call(self, x, training=None):
        # Handle channels-first format from PyTorch datasets
        # Convert from [b, c, h, w] to [b, h, w, c] if needed
        if len(x.shape) == 4 and x.shape[1] in [1, 3]:  # Likely channels-first
            x = tf.transpose(x, [0, 2, 3, 1])  # Convert to [b, h, w, c]
        
        features = []
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.layers_blocks(out, training=training)
        out = tf.nn.relu(self.bn2(self.conv2(out), training=training))
        features.append(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = tf.nn.avg_pool(out, ksize=4, strides=4, padding='VALID')
        out = tf.reshape(out, [out.shape[0], -1])
        features.append(out)
        out = self.linear(out)
        if self.get_features:
            return features, out
        return out