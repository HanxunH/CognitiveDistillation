'''GoogLeNet with MindSpore.'''
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Inception(nn.Cell):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.SequentialCell(
            nn.Conv2d(in_planes, n1x1, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.SequentialCell(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.SequentialCell(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=3, stride=1, pad_mode='pad', padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1, has_bias=False),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(),
        )
        
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return self.concat((y1, y2, y3, y4))


class GoogLeNet(nn.Cell):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.SequentialCell(
            nn.Conv2d(3, 192, kernel_size=3, pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad', padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.linear = nn.Dense(1024, num_classes)
        self.get_features = False
        
        self.reshape = ops.Reshape()

    def construct(self, x):
        features = []
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        features.append(out)
        out = self.avgpool(out)
        out = self.reshape(out, (out.shape[0], -1))
        features.append(out)
        out = self.linear(out)
        if self.get_features:
            return features, out
        return out