import mlconfig
import paddle
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from . import dynamic_models
import sys
sys.path.append('/root/CognitiveDistillation/paddlepaddle')
from lr_scheduler import MultiStepLR, SGD

# Register optimizers
mlconfig.register(SGD)
mlconfig.register(paddle.optimizer.Adam)
mlconfig.register(paddle.optimizer.AdamW)

# Register learning rate schedulers
mlconfig.register(MultiStepLR)
mlconfig.register(paddle.optimizer.lr.CosineAnnealingDecay)
mlconfig.register(paddle.optimizer.lr.StepDecay)
mlconfig.register(paddle.optimizer.lr.ExponentialDecay)

# Register models
mlconfig.register(ResNet18)
mlconfig.register(ResNet34)
mlconfig.register(ResNet50)
mlconfig.register(ResNet101)
mlconfig.register(ResNet152)