"""
TensorFlow optimizer and scheduler wrappers to match PyTorch interface.
"""
import tensorflow as tf
import mlconfig


class SGD:
    """
    Wrapper for TensorFlow SGD optimizer to match PyTorch interface.
    """
    def __init__(self, params, lr=0.001, momentum=0.0, weight_decay=0.0, **kwargs):
        # In TensorFlow, params is typically model.trainable_variables
        # We don't use params directly in initialization
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Create TensorFlow optimizer
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=momentum,
            **kwargs
        )
        
        # Store reference to model parameters for weight decay
        self.params = None
        
    def __call__(self, model):
        """Called with model to get trainable variables"""
        self.params = model.trainable_variables
        return self
        
    def apply_gradients(self, grads_and_vars):
        """Apply gradients with optional weight decay"""
        if self.weight_decay > 0 and self.params is not None:
            # Apply weight decay manually
            for var in self.params:
                if 'bias' not in var.name.lower() and 'batch_normalization' not in var.name.lower():
                    var.assign_sub(self.weight_decay * self.learning_rate * var)
        
        # Apply gradients
        self.optimizer.apply_gradients(grads_and_vars)
    
    @property
    def learning_rate(self):
        return self.optimizer.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        if isinstance(self.optimizer.learning_rate, tf.Variable):
            self.optimizer.learning_rate.assign(value)
        else:
            self.optimizer.learning_rate = value
    
    def get_config(self):
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'optimizer_config': self.optimizer.get_config()
        }
    
    def get_weights(self):
        return self.optimizer.get_weights()
    
    def set_weights(self, weights):
        self.optimizer.set_weights(weights)


class Adam:
    """
    Wrapper for TensorFlow Adam optimizer to match PyTorch interface.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, **kwargs):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Create TensorFlow optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            beta_1=betas[0],
            beta_2=betas[1],
            epsilon=eps,
            **kwargs
        )
        
        self.params = None
    
    def __call__(self, model):
        """Called with model to get trainable variables"""
        self.params = model.trainable_variables
        return self
    
    def apply_gradients(self, grads_and_vars):
        """Apply gradients with optional weight decay"""
        if self.weight_decay > 0 and self.params is not None:
            # Apply weight decay manually
            for var in self.params:
                if 'bias' not in var.name.lower() and 'batch_normalization' not in var.name.lower():
                    var.assign_sub(self.weight_decay * self.learning_rate * var)
        
        # Apply gradients
        self.optimizer.apply_gradients(grads_and_vars)
    
    @property
    def learning_rate(self):
        return self.optimizer.learning_rate
    
    @learning_rate.setter  
    def learning_rate(self, value):
        if isinstance(self.optimizer.learning_rate, tf.Variable):
            self.optimizer.learning_rate.assign(value)
        else:
            self.optimizer.learning_rate = value
    
    def get_config(self):
        return {
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'optimizer_config': self.optimizer.get_config()
        }
    
    def get_weights(self):
        return self.optimizer.get_weights()
    
    def set_weights(self, weights):
        self.optimizer.set_weights(weights)


class AdamW:
    """
    Wrapper for TensorFlow AdamW optimizer to match PyTorch interface.
    TensorFlow 2.x has native AdamW support.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, **kwargs):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Use TensorFlow's AdamW if available
        if hasattr(tf.keras.optimizers, 'AdamW'):
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=lr,
                beta_1=betas[0],
                beta_2=betas[1],
                epsilon=eps,
                weight_decay=weight_decay,
                **kwargs
            )
        else:
            # Fallback to Adam with manual weight decay
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr,
                beta_1=betas[0],
                beta_2=betas[1],
                epsilon=eps,
                **kwargs
            )
        
        self.params = None
    
    def __call__(self, model):
        """Called with model to get trainable variables"""
        self.params = model.trainable_variables
        return self
    
    def apply_gradients(self, grads_and_vars):
        """Apply gradients"""
        if not hasattr(tf.keras.optimizers, 'AdamW') and self.weight_decay > 0 and self.params is not None:
            # Apply weight decay manually if using Adam fallback
            for var in self.params:
                if 'bias' not in var.name.lower() and 'batch_normalization' not in var.name.lower():
                    var.assign_sub(self.weight_decay * self.learning_rate * var)
        
        # Apply gradients
        self.optimizer.apply_gradients(grads_and_vars)
    
    @property
    def learning_rate(self):
        return self.optimizer.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value):
        if isinstance(self.optimizer.learning_rate, tf.Variable):
            self.optimizer.learning_rate.assign(value)
        else:
            self.optimizer.learning_rate = value
    
    def get_config(self):
        return {
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'optimizer_config': self.optimizer.get_config()
        }
    
    def get_weights(self):
        return self.optimizer.get_weights()
    
    def set_weights(self, weights):
        self.optimizer.set_weights(weights)


class MultiStepLR:
    """
    Learning rate scheduler that decreases learning rate at specified milestones.
    Equivalent to PyTorch's MultiStepLR.
    """
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.optimizer = optimizer
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate
        
        # Initialize current epoch
        self.current_epoch = 0 if last_epoch == -1 else last_epoch
        
        # Set initial learning rate
        if last_epoch != -1:
            self.step()
    
    def step(self, epoch=None):
        """Update learning rate"""
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch
        
        # Calculate learning rate
        lr = self.base_lr
        for milestone in self.milestones:
            if self.current_epoch >= milestone:
                lr *= self.gamma
        
        # Update optimizer learning rate
        if hasattr(self.optimizer, 'learning_rate'):
            if isinstance(self.optimizer.learning_rate, tf.Variable):
                self.optimizer.learning_rate.assign(lr)
            else:
                self.optimizer.learning_rate = lr
        elif hasattr(self.optimizer, 'lr'):
            if isinstance(self.optimizer.lr, tf.Variable):
                self.optimizer.lr.assign(lr)
            else:
                self.optimizer.lr = lr
        
        return lr
    
    def get_last_lr(self):
        """Get the last computed learning rate"""
        if hasattr(self.optimizer, 'learning_rate'):
            lr = self.optimizer.learning_rate
        elif hasattr(self.optimizer, 'lr'):
            lr = self.optimizer.lr
        else:
            lr = self.base_lr
        
        if hasattr(lr, 'numpy'):
            return [lr.numpy()]
        else:
            return [lr]
    
    def __dict__(self):
        """Return state dict for saving"""
        return {
            'milestones': self.milestones,
            'gamma': self.gamma,
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch
        }


class CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    Equivalent to PyTorch's CosineAnnealingLR.
    """
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lr = optimizer.learning_rate.numpy() if hasattr(optimizer.learning_rate, 'numpy') else optimizer.learning_rate
        
        # Initialize current epoch
        self.current_epoch = 0 if last_epoch == -1 else last_epoch
        
        # Set initial learning rate
        if last_epoch != -1:
            self.step()
    
    def step(self, epoch=None):
        """Update learning rate"""
        import math
        
        if epoch is None:
            self.current_epoch += 1
        else:
            self.current_epoch = epoch
        
        # Calculate learning rate using cosine annealing
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.current_epoch / self.T_max)) / 2
        
        # Update optimizer learning rate
        if hasattr(self.optimizer, 'learning_rate'):
            if isinstance(self.optimizer.learning_rate, tf.Variable):
                self.optimizer.learning_rate.assign(lr)
            else:
                self.optimizer.learning_rate = lr
        elif hasattr(self.optimizer, 'lr'):
            if isinstance(self.optimizer.lr, tf.Variable):
                self.optimizer.lr.assign(lr)
            else:
                self.optimizer.lr = lr
        
        return lr
    
    def get_last_lr(self):
        """Get the last computed learning rate"""
        if hasattr(self.optimizer, 'learning_rate'):
            lr = self.optimizer.learning_rate
        elif hasattr(self.optimizer, 'lr'):
            lr = self.optimizer.lr
        else:
            lr = self.base_lr
        
        if hasattr(lr, 'numpy'):
            return [lr.numpy()]
        else:
            return [lr]
    
    def __dict__(self):
        """Return state dict for saving"""
        return {
            'T_max': self.T_max,
            'eta_min': self.eta_min,
            'base_lr': self.base_lr,
            'current_epoch': self.current_epoch
        }


# Register the wrappers with mlconfig
mlconfig.register(SGD)
mlconfig.register(Adam)
mlconfig.register(AdamW)
mlconfig.register(MultiStepLR)
mlconfig.register(CosineAnnealingLR)