import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, pad_mode='valid', has_bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            ])

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, pad_mode='valid', has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                               pad_mode='valid', has_bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.SequentialCell()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, pad_mode='valid', has_bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            ])

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.get_features = False
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.linear = nn.Dense(512 * block.expansion, num_classes)
        self.reshape = ops.Reshape()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(layers)

    def construct(self, x):
        features = []
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        features.append(out)
        out = self.layer2(out)
        features.append(out)
        out = self.layer3(out)
        features.append(out)
        out = self.layer4(out)
        features.append(out)
        out = self.pool(out)
        # Use reshape operation instead of view
        out = self.reshape(out, (x.shape[0], -1))
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