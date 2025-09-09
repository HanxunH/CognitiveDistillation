import mlconfig
import mindspore.nn as nn


# Create wrappers for optimizers to handle parameter differences
class SGD:
    def __init__(self, params, lr=0.1, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        self.optimizer = nn.SGD(params, learning_rate=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        self.learning_rate = lr
        
    def __call__(self, grads):
        return self.optimizer(grads)
        
    @property
    def parameters(self):
        return self.optimizer.parameters


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.optimizer = nn.Adam(params, learning_rate=lr, beta1=betas[0], beta2=betas[1], eps=eps, weight_decay=weight_decay)
        self.learning_rate = lr
        
    def __call__(self, grads):
        return self.optimizer(grads)
        
    @property  
    def parameters(self):
        return self.optimizer.parameters


# Register optimizers
mlconfig.register(SGD)
mlconfig.register(Adam)

# For schedulers, we'll create simple wrappers since MindSpore has different scheduler API
class MultiStepLR:
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        
    def get_lr(self):
        # Simple implementation - in real usage this would be more complex
        return 0.1


class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        
    def get_lr(self):
        # Simple implementation - in real usage this would be more complex
        return 0.1


# Register schedulers  
mlconfig.register(MultiStepLR)
mlconfig.register(CosineAnnealingLR)

__all__ = ['MultiStepLR', 'CosineAnnealingLR']