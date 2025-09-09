from paddle import nn


class Conv2dBlock(nn.Layer):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2D(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2D(out_c, epsilon=1e-5, momentum=0.95)
        if relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Layer):
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        # PaddlePaddle's MaxPool2D doesn't support dilation parameter
        # According to paddlepaddle-exemptions.md item 12
        if dilation != (1, 1):
            raise NotImplementedError(f"PaddlePaddle MaxPool2D does not support dilation != 1. Got dilation={dilation}")
        self.maxpooling = nn.MaxPool2D(kernel_size=ker_size, stride=stride,
                                       ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Layer):
    def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x