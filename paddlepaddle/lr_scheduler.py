import paddle


class SGD(paddle.optimizer.Momentum):
    """Wrapper to adapt PyTorch-style SGD to PaddlePaddle API
    Note: PaddlePaddle's SGD doesn't support momentum, so we use Momentum optimizer"""
    def __init__(self, parameters, lr=0.1, momentum=0, weight_decay=0, **kwargs):
        # Map PyTorch parameter names to PaddlePaddle
        super().__init__(
            learning_rate=lr,
            momentum=momentum,
            parameters=parameters,
            weight_decay=weight_decay,
            **kwargs
        )


class MultiStepLR:
    """Wrapper to adapt PyTorch-style MultiStepLR to PaddlePaddle API"""
    def __init__(self, optimizer, milestones, gamma=0.1):
        # Create the PaddlePaddle scheduler
        self.scheduler = paddle.optimizer.lr.MultiStepDecay(
            learning_rate=optimizer.get_lr(),
            milestones=milestones,
            gamma=gamma
        )
        self.optimizer = optimizer
        # Set the scheduler in the optimizer
        self.optimizer.set_lr_scheduler(self.scheduler)
    
    def step(self):
        """Update learning rate"""
        self.scheduler.step()
    
    def state_dict(self):
        """Return state dict for saving"""
        return {
            'milestones': self.scheduler.milestones,
            'gamma': self.scheduler.gamma,
            'last_epoch': self.scheduler.last_epoch
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.scheduler.last_epoch = state_dict['last_epoch']