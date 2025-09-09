import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class BasicBlock(Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=3,
                                   strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * planes, kernel_size=1,
                              strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x  # Identity function

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.nn.relu(out)
        return out


class Bottleneck(Model):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = layers.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, kernel_size=3, strides=stride,
                                   padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.expansion * planes, kernel_size=1,
                                   use_bias=False)
        self.bn3 = layers.BatchNormalization()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Sequential([
                layers.Conv2D(self.expansion * planes, kernel_size=1,
                              strides=stride, use_bias=False),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = lambda x: x  # Identity function

    def call(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = tf.nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = tf.nn.relu(out)
        return out


class ResNet(Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.get_features = False
        self.conv1 = layers.Conv2D(64, kernel_size=3, strides=1,
                                   padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.linear = layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(layers)

    def call(self, x):
        features = []
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        features.append(out)
        out = self.layer2(out)
        features.append(out)
        out = self.layer3(out)
        features.append(out)
        out = self.layer4(out)
        features.append(out)
        out = self.pool(out)
        out = tf.reshape(out, [tf.shape(x)[0], -1])
        features.append(out)
        out = self.linear(out)
        if self.get_features:
            return features, out
        return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)