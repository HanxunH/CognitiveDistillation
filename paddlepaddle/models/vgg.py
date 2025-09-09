'''VGG11/13/16/19 in Paddle.'''
import paddle
import paddle.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Layer):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.classifier = nn.Linear(512, num_classes)
        self.get_features = False

    def forward(self, x):
        features = []
        out = self.features(x)
        features.append(out)
        out = self.pool(out)
        out = out.reshape([out.shape[0], -1])
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
                layers += [nn.MaxPool2D(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2D(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2D(x),
                           nn.ReLU()]
                in_channels = x
        return nn.Sequential(*layers)


def VGG16(num_classes=10):
    return VGG('VGG16', num_classes=num_classes)