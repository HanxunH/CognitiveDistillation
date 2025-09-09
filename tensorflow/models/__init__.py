import mlconfig
import tensorflow as tf
from . import resnet
from . import issba_resnet
from . import celeba_resnet
from . import mobilenetv2
from . import google_inception
from . import efficientnet
from . import preact_resnet
from . import vgg
from . import vit

# Import optimizers to ensure they are registered
# The actual registration happens in optimizers.py
import tensorflow_impl.optimizers

# Models
mlconfig.register(resnet.ResNet18)
mlconfig.register(issba_resnet.ResNet18_200)
mlconfig.register(celeba_resnet.AttributesResNet18)
mlconfig.register(mobilenetv2.MobileNetV2)
mlconfig.register(google_inception.GoogLeNet)
mlconfig.register(efficientnet.EfficientNetB0)
mlconfig.register(preact_resnet.PreActResNet18)
mlconfig.register(preact_resnet.PreActResNet34)
mlconfig.register(preact_resnet.PreActResNet50)
mlconfig.register(preact_resnet.PreActResNet101)
mlconfig.register(preact_resnet.PreActResNet152)
mlconfig.register(vgg.VGG16)
mlconfig.register(vit.vit_base_patch16)
mlconfig.register(vit.vit_large_patch16)
mlconfig.register(vit.vit_huge_patch14)