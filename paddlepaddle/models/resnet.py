import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, self.expansion * planes, kernel_size=1,
                               bias_attr=False)
        self.bn3 = nn.BatchNorm2D(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.get_features = False
        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1,
                               padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # RISK_INFO: [API 行为不等价] - PaddlePaddle's AvgPool2D defaults to `exclusive=True` (not including padding in average calculation), while PyTorch's AvgPool2d's default behavior is equivalent to `exclusive=False`. The parameter was explicitly set here to ensure consistency.
        # Use exclusive=False to match PyTorch's default behavior
        self.pool = nn.AvgPool2D(kernel_size=4, exclusive=False)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        features.append(out)
        out = self.layer2(out)
        features.append(out)
        out = self.layer3(out)
        features.append(out)
        out = self.layer4(out)
        features.append(out)
        out = self.pool(out)
        # Use tensor method reshape to match original style
        out = out.reshape([x.shape[0], -1])
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