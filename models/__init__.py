import mlconfig
import torch
from . import resnet, issba_resnet, dynamic_models, vgg, google_inception, vit, mobilenetv2
from . import efficientnet
from . import preact_resnet
from . import celeba_resnet
from . import toy_model


mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.AdamW)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)

# Models
mlconfig.register(resnet.ResNet18)
mlconfig.register(preact_resnet.PreActResNet50)
mlconfig.register(preact_resnet.PreActResNet101)
mlconfig.register(mobilenetv2.MobileNetV2)
mlconfig.register(vgg.VGG16)
mlconfig.register(google_inception.GoogLeNet)
mlconfig.register(issba_resnet.ResNet18_200)
mlconfig.register(vit.vit_base_patch16)
mlconfig.register(efficientnet.EfficientNetB0)
mlconfig.register(celeba_resnet.AttributesResNet18)
mlconfig.register(toy_model.ToyModel)
