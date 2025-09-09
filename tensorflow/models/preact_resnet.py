'''Pre-activation ResNet in TensorFlow.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class PreActBlock(Model):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(planes, kernel_size=3, strides=stride, 
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1, 
                                   padding='same', use_bias=False)

        self.has_shortcut = False
        if stride != 1 or in_planes != self.expansion * planes:
            self.has_shortcut = True
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * planes, kernel_size=1, 
                             strides=stride, use_bias=False)
            ])

    def call(self, x):
        out = tf.nn.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.has_shortcut else x
        out = self.conv1(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(Model):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=stride, 
                                   padding='same', use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * planes, kernel_size=1, 
                                   use_bias=False)

        self.has_shortcut = False
        if stride != 1 or in_planes != self.expansion * planes:
            self.has_shortcut = True
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * planes, kernel_size=1, 
                             strides=stride, use_bias=False)
            ])

    def call(self, x):
        out = tf.nn.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.has_shortcut else x
        out = self.conv1(out)
        out = self.conv2(tf.nn.relu(self.bn2(out)))
        out = self.conv3(tf.nn.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1, 
                                   padding='same', use_bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = layers.Dense(num_classes)
        self.get_features = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers_list = []
        for stride in strides:
            layers_list.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(layers_list)

    def call(self, x):
        features = []
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features.append(out)
        out = tf.nn.avg_pool2d(out, ksize=4, strides=4, padding='VALID')
        out = tf.reshape(out, [tf.shape(out)[0], -1])
        features.append(out)
        out = self.linear(out)
        if self.get_features:
            return features, out
        return out


def PreActResNet18(num_classes=10):
    return PreActResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes)


def PreActResNet34(num_classes=10):
    return PreActResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes)


def PreActResNet50(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)


def PreActResNet101(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes)


def PreActResNet152(num_classes=10):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3], num_classes=num_classes)


def test():
    net = PreActResNet18()
    y = net(tf.random.normal([1, 3, 32, 32]))
    print(y.shape)

# test()