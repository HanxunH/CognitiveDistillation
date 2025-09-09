'''MobileNetV2 in MindSpore.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Block(nn.Cell):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, pad_mode='valid', has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, pad_mode='pad', group=planes, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, pad_mode='valid', has_bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.SequentialCell()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.SequentialCell([
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, pad_mode='valid', has_bias=False),
                nn.BatchNorm2d(out_planes),
            ])

    def construct(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Cell):
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, pad_mode='valid', has_bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Dense(1280, num_classes)
        self.get_features = False
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=4)
        self.reshape = ops.Reshape()

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.SequentialCell(layers)

    def construct(self, x):
        features = []
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.relu(self.bn2(self.conv2(out)))
        features.append(out)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = self.pool(out)
        out = self.reshape(out, (out.shape[0], -1))
        features.append(out)
        out = self.linear(out)
        if self.get_features:
            return features, out
        return out