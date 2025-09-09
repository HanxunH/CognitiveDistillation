import mlconfig
import mindspore.nn as nn


class CrossEntropyLoss:
    """CrossEntropyLoss wrapper for compatibility with PyTorch-style configs"""
    def __init__(self):
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    
    def __call__(self, logits, labels):
        return self.loss_fn(logits, labels)


# Register loss functions
mlconfig.register(nn.SoftmaxCrossEntropyWithLogits)
mlconfig.register(CrossEntropyLoss)