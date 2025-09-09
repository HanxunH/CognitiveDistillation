'''VGG11/13/16/19 in MindSpore.'''
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Cell):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Dense(512, num_classes)
        self.get_features = False
        self.reshape = ops.Reshape()

    def construct(self, x):
        features = []
        out = self.features(x)
        features.append(out)
        out = self.pool(out)
        # Flatten the tensor
        batch_size = out.shape[0]
        out = self.reshape(out, (batch_size, -1))
        features.append(out)
        out = self.classifier(out)
        if self.get_features:
            return features, out
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, pad_mode='pad', has_bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
        return nn.SequentialCell(layers)


def VGG11(num_classes=10):
    return VGG('VGG11', num_classes=num_classes)


def VGG13(num_classes=10):
    return VGG('VGG13', num_classes=num_classes)


def VGG16(num_classes=10):
    return VGG('VGG16', num_classes=num_classes)


def VGG19(num_classes=10):
    return VGG('VGG19', num_classes=num_classes)