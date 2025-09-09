'''Pre-activation ResNet in MindSpore.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class PreActBlock(nn.Cell):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, 
                               padding=1, pad_mode='pad', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, pad_mode='pad', has_bias=False)
        self.relu = nn.ReLU()
        
        self.has_shortcut = False
        if stride != 1 or in_planes != self.expansion * planes:
            self.has_shortcut = True
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, pad_mode='valid', has_bias=False)
            ])

    def construct(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.has_shortcut else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = out + shortcut
        return out


class PreActBottleneck(nn.Cell):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, pad_mode='valid', has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, 
                               padding=1, pad_mode='pad', has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, 
                               pad_mode='valid', has_bias=False)
        self.relu = nn.ReLU()
        
        self.has_shortcut = False
        if stride != 1 or in_planes != self.expansion * planes:
            self.has_shortcut = True
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, 
                         stride=stride, pad_mode='valid', has_bias=False)
            ])

    def construct(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.has_shortcut else x
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.conv3(self.relu(self.bn3(out)))
        out = out + shortcut
        return out


class PreActResNet(nn.Cell):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, 
                               pad_mode='pad', has_bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Dense(512 * block.expansion, num_classes)
        self.get_features = False
        
        # Create avg pool operation
        self.avg_pool = ops.AvgPool(kernel_size=4, strides=4)
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.SequentialCell(layers)

    def construct(self, x):
        features = []
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features.append(out)
        
        # Average pooling
        out = self.avg_pool(out)
        out = self.flatten(out)
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
    y = net(ms.Tensor(ms.numpy.randn(1, 3, 32, 32), dtype=ms.float32))
    print(y.shape)

# test()