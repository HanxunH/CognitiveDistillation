import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal, Constant


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, pad_mode='pad', has_bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class fc_block(nn.Cell):
    def __init__(self, inplanes, planes, drop_rate=0.15):
        super(fc_block, self).__init__()
        self.fc = nn.Dense(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(p=drop_rate)
        self.relu = nn.ReLU()
        self.drop_rate = drop_rate

    def construct(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x


class AttributesResNet(nn.Cell):

    def __init__(self, block, layers, num_attributes=40, zero_init_residual=False, pretrained_key=None):
        super(AttributesResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               pad_mode='pad', has_bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.stem = fc_block(512 * block.expansion, 512)
        self.forward_idx = None
        for i in range(num_attributes):
            setattr(self, 'classifier' + str(i).zfill(2), nn.SequentialCell(fc_block(512, 256), nn.Dense(256, 2)))
        self.num_attributes = num_attributes
        self.get_features = False
        
        # Initialize weights
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(ms.common.initializer.initializer(
                    HeNormal(mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape,
                    cell.weight.dtype))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(ms.common.initializer.initializer(
                    Constant(1), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(ms.common.initializer.initializer(
                    Constant(0), cell.beta.shape, cell.beta.dtype))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for _, cell in self.cells_and_names():
                if isinstance(cell, Bottleneck):
                    cell.bn3.gamma.set_data(ms.common.initializer.initializer(
                        Constant(0), cell.bn3.gamma.shape, cell.bn3.gamma.dtype))
                elif isinstance(cell, BasicBlock):
                    cell.bn2.gamma.set_data(ms.common.initializer.initializer(
                        Constant(0), cell.bn2.gamma.shape, cell.bn2.gamma.dtype))
        if pretrained_key is not None:
            init_pretrained_weights(self, model_urls[pretrained_key])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        reshape = ops.Reshape()
        x = reshape(x, (x.shape[0], -1))
        x = self.stem(x)
        if self.get_features:
            return x
        elif self.forward_idx is not None:
            classifier = getattr(self, 'classifier' + str(self.forward_idx).zfill(2))
            return classifier(x)

        y = []
        for i in range(self.num_attributes):
            classifier = getattr(self, 'classifier' + str(i).zfill(2))
            y.append(classifier(x))

        return y

    def forward_fc(self, x, idx):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        reshape = ops.Reshape()
        x = reshape(x, (x.shape[0], -1))
        x = self.stem(x)

        classifier = getattr(self, 'classifier' + str(idx).zfill(2))
        return classifier(x)


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    # In MindSpore, loading PyTorch pretrained weights requires special handling
    # Since this involves loading from PyTorch model zoo, we'll skip implementation
    # for now as it would require torch dependency
    raise NotImplementedError("Loading PyTorch pretrained weights is not implemented for MindSpore version")
    # The actual implementation would need to:
    # 1. Download the PyTorch weights
    # 2. Convert them to MindSpore format
    # 3. Load into the model
    print("Initialized model with pretrained weights from {}".format(model_url))


def AttributesResNet18(num_attributes=40):
    return AttributesResNet(BasicBlock, [2, 2, 2, 2], num_attributes=num_attributes)