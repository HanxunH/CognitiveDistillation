import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
import numpy as np


def total_variation_loss(img, weight=1):
    """Calculate total variation loss for regularization"""
    b, c, h, w = img.shape
    # Compute differences between adjacent pixels
    tv_h = ops.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum(axis=(1, 2, 3))
    tv_w = ops.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum(axis=(1, 2, 3))
    return weight * (tv_h + tv_w) / (c * h * w)


class CognitiveDistillation:
    """
    Cognitive Distillation detection method.
    
    This class does NOT inherit from nn.Cell as per exemption #7 in mindspore-exemptions.md
    to maintain the same calling interface as the PyTorch version.
    """
    
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        super(CognitiveDistillation, self).__init__()
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = False
        self._EPSILON = 1.e-6
        self.norm_only = norm_only
        
    def get_raw_mask(self, mask):
        """Convert mask parameter to actual mask values using tanh activation"""
        mask = (ops.tanh(mask) + 1) / 2
        return mask
    
    def l1_loss(self, pred, target):
        """L1 loss without reduction"""
        return ops.abs(pred - target)
    
    def __call__(self, model, images, labels=None, preprocessor=None):
        """
        Forward pass for cognitive distillation detection.
        
        Args:
            model: The model to analyze
            images: Input images tensor
            labels: Optional labels (third argument for compatibility)
            preprocessor: Optional preprocessor (defaults to identity)
        
        Returns:
            Detected masks or mask norms
        """
        # Handle both calling conventions for compatibility
        if callable(labels) and preprocessor is None:
            # If third argument is callable, it's the preprocessor
            preprocessor = labels
            labels = None
        return self.forward(model, images, preprocessor, labels)
    
    def forward(self, model, images, preprocessor=None, labels=None):
        """
        Main forward implementation for cognitive distillation.
        
        Since MindSpore's optimizer API is different and complex, we implement
        a simplified optimization loop using numpy arrays for the mask parameter
        and manual gradient computation + updates.
        """
        model.set_train(False)  # Set model to eval mode
        
        # Check input image range
        # Note: ops.min/max return (value, index) tuple in MindSpore
        min_val = ops.min(images)
        max_val = ops.max(images)
        # Handle both tuple and scalar returns
        if isinstance(min_val, tuple):
            min_val = min_val[0]
        if isinstance(max_val, tuple):
            max_val = max_val[0]
        if min_val < 0 or max_val > 1:
            # Use TypeError as per exemption #1 in mindspore-exemptions.md
            raise TypeError('images should be normalized')
        
        b, c, h, w = images.shape
        
        # Apply preprocessor if provided
        if preprocessor is None:
            preprocessor = lambda x: x  # Identity function
        
        # Get initial model outputs
        if self.get_features:
            features, logits = model(preprocessor(images))
            # Detach from computation graph
            features = [ops.stop_gradient(f) for f in features]
        else:
            logits = model(preprocessor(images))
            logits = ops.stop_gradient(logits)
        
        # Initialize mask parameter as numpy array for manual optimization
        mask_param_np = np.ones((b, self.mask_channel, h, w), dtype=np.float32)
        
        # Initialize Adam optimizer state
        m_np = np.zeros_like(mask_param_np)
        v_np = np.zeros_like(mask_param_np)
        beta1 = 0.1
        beta2 = 0.1
        epsilon = 1e-8
        
        # Optimization loop
        for step in range(self.num_steps):
            # Convert current mask parameter to tensor
            mask_param = Tensor(mask_param_np, ms.float32)
            mask_param.requires_grad = True
            
            # Forward pass with current mask
            mask = self.get_raw_mask(mask_param)
            
            # Generate random background
            rand_bg = ops.uniform((b, c, 1, 1), Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
            if hasattr(images, 'dtype'):
                rand_bg = rand_bg.astype(images.dtype)
            
            # Create adversarial examples
            x_adv = images * mask + (1 - mask) * rand_bg
            
            # Compute loss based on features or logits
            if self.get_features:
                adv_fe, adv_logits = model(x_adv)
                if len(adv_fe[-2].shape) == 4:
                    loss = self.l1_loss(adv_fe[-2], features[-2]).mean(axis=(1, 2, 3))
                else:
                    loss = self.l1_loss(adv_fe[-2], features[-2]).mean(axis=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1_loss(adv_logits, logits).mean(axis=1)
            
            # Add regularization terms
            # MindSpore's norm doesn't support multiple dimensions, so flatten first
            mask_flat = mask.reshape(mask.shape[0], -1)
            norm = ops.norm(mask_flat, ord=self.p, dim=1)
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_mean = loss_total.mean()
            
            # Compute gradient using MindSpore's autograd
            grad_fn = ops.grad(lambda mp: self._compute_loss(
                mp, model, images, rand_bg, logits, 
                features if self.get_features else None
            ))
            grad = grad_fn(mask_param)
            
            # Convert gradient to numpy for manual update
            grad_np = grad.asnumpy()
            
            # Adam update (manual implementation in numpy)
            # Update biased first moment estimate
            m_np = beta1 * m_np + (1 - beta1) * grad_np
            # Update biased second raw moment estimate  
            v_np = beta2 * v_np + (1 - beta2) * grad_np ** 2
            
            # Compute bias-corrected moment estimates
            t = step + 1
            m_hat = m_np / (1 - beta1 ** t)
            v_hat = v_np / (1 - beta2 ** t)
            
            # Update parameters
            mask_param_np = mask_param_np - self.lr * m_hat / (np.sqrt(v_hat) + epsilon)
        
        # Get final mask
        mask_param_final = Tensor(mask_param_np, ms.float32)
        mask = self.get_raw_mask(mask_param_final)
        mask = ops.stop_gradient(mask)
        
        # Convert to numpy for CPU operations (as per exemption #3)
        mask_np = mask.asnumpy()
        
        if self.norm_only:
            # Return L1 norm of mask
            # Reshape mask for norm calculation (batch_size, -1)
            mask_np_flat = mask_np.reshape(mask_np.shape[0], -1)
            mask_norm = np.linalg.norm(mask_np_flat, ord=1, axis=1)
            return Tensor(mask_norm, ms.float32)
        
        # Return mask tensor
        return Tensor(mask_np, ms.float32)
    
    def _compute_loss(self, mask_param, model, images, rand_bg, logits, features):
        """Helper function for gradient computation"""
        mask = self.get_raw_mask(mask_param)
        
        # Create adversarial examples
        x_adv = images * mask + (1 - mask) * rand_bg
        
        # Compute loss based on features or logits
        if self.get_features:
            adv_fe, adv_logits = model(x_adv)
            if len(adv_fe[-2].shape) == 4:
                loss = self.l1_loss(adv_fe[-2], features[-2]).mean(axis=(1, 2, 3))
            else:
                loss = self.l1_loss(adv_fe[-2], features[-2]).mean(axis=1)
        else:
            adv_logits = model(x_adv)
            loss = self.l1_loss(adv_logits, logits).mean(axis=1)
        
        # Add regularization terms
        mask_flat = mask.reshape(mask.shape[0], -1)
        norm = ops.norm(mask_flat, ord=self.p, dim=1)
        norm = norm * self.gamma
        loss_total = loss + norm + self.beta * total_variation_loss(mask)
        
        return loss_total.mean()