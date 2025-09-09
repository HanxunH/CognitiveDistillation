import mlconfig
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, BasicBlock, Bottleneck, ResNet
from . import blocks
from . import dynamic_models
from .issba_resnet import ResNet18_200
from .vgg import VGG, VGG11, VGG13, VGG16, VGG19
from .mobilenetv2 import MobileNetV2
from .efficientnet import EfficientNet, EfficientNetB0, MBConv, MBConvConfig
from .preact_resnet import (PreActResNet, PreActBlock, PreActBottleneck,
                             PreActResNet18, PreActResNet34, PreActResNet50,
                             PreActResNet101, PreActResNet152)
from .vit import VisionTransformer, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from .celeba_resnet import AttributesResNet18
from .google_inception import GoogLeNet

# Register models with mlconfig
mlconfig.register(ResNet18)
mlconfig.register(ResNet34)
mlconfig.register(ResNet50)
mlconfig.register(ResNet101)
mlconfig.register(ResNet152)
mlconfig.register(ResNet18_200)
mlconfig.register(VGG11)
mlconfig.register(VGG13)
mlconfig.register(VGG16)
mlconfig.register(VGG19)
mlconfig.register(MobileNetV2)
mlconfig.register(EfficientNetB0)
mlconfig.register(PreActResNet18)
mlconfig.register(PreActResNet34)
mlconfig.register(PreActResNet50)
mlconfig.register(PreActResNet101)
mlconfig.register(PreActResNet152)
mlconfig.register(vit_base_patch16)
mlconfig.register(vit_large_patch16)
mlconfig.register(vit_huge_patch14)
mlconfig.register(AttributesResNet18)
mlconfig.register(GoogLeNet)

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 
           'BasicBlock', 'Bottleneck', 'ResNet', 'blocks', 'dynamic_models', 'ResNet18_200',
           'VGG', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'MobileNetV2',
           'EfficientNet', 'EfficientNetB0', 'MBConv', 'MBConvConfig',
           'PreActResNet', 'PreActBlock', 'PreActBottleneck',
           'PreActResNet18', 'PreActResNet34', 'PreActResNet50',
           'PreActResNet101', 'PreActResNet152',
           'VisionTransformer', 'vit_base_patch16', 'vit_large_patch16', 'vit_huge_patch14',
           'AttributesResNet18', 'GoogLeNet']