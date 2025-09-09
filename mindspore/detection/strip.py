import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np


class STRIP_Detection:
    """
    STRIP (STRong Intentional Perturbation) detection method.
    
    This class does NOT inherit from nn.Cell as per exemption #7 in mindspore-exemptions.md
    to maintain the same calling interface as the PyTorch version.
    """
    
    def __init__(self, data, alpha=1.0, beta=1.0, n=100):
        """
        Initialize STRIP detection.
        
        Args:
            data: Background data tensor for superimposition
            alpha: Weight for original image in superimposition
            beta: Weight for overlay image in superimposition
            n: Number of superimpositions to perform per image
        """
        super(STRIP_Detection, self).__init__()
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.n = n
    
    def _superimpose(self, background, overlay):
        """
        Superimpose overlay onto background image.
        
        Args:
            background: Background image tensor
            overlay: Overlay image tensor
        
        Returns:
            Superimposed image clamped to [0, 1]
        """
        # cv2.addWeighted(background, 1, overlay, 1, 0) equivalent
        imgs = self.alpha * background + self.beta * overlay
        imgs = ops.clip_by_value(imgs, Tensor(0.0, ms.float32), Tensor(1.0, ms.float32))
        return imgs
    
    def __call__(self, model, imgs, labels=None):
        """
        Call method to maintain same interface as PyTorch version.
        
        Args:
            model: Model to evaluate
            imgs: Input images tensor
            labels: Optional labels (not used)
        
        Returns:
            Entropy values for each image
        """
        return self.forward(model, imgs, labels)
    
    def forward(self, model, imgs, labels=None):
        """
        Compute entropy values using STRIP method.
        
        Args:
            model: Model to evaluate
            imgs: Input images tensor
            labels: Optional labels (not used)
        
        Returns:
            Tensor of entropy values H for each image
        """
        # Generate random indices for background selection
        # Convert to int to ensure MindSpore compatibility
        idx = np.random.randint(0, self.data.shape[0], size=self.n)
        idx = [int(i) for i in idx]  # Convert numpy int64 to Python int
        H = []
        
        for img in imgs:
            # Stack the image n times
            x = ops.stack([img] * self.n)
            
            for i in range(self.n):
                x_0 = x[i]
                x_1 = self.data[idx[i]]
                x_2 = self._superimpose(x_0, x_1)
                # Update the stacked tensor
                # MindSpore doesn't support direct assignment, so we need to reconstruct
                x_list = []
                for j in range(self.n):
                    if j == i:
                        x_list.append(x_2)
                    else:
                        x_list.append(x[j])
                x = ops.stack(x_list)
            
            # Get model predictions
            logits = model(x)
            
            # Compute softmax probabilities
            # Detach from computation graph
            logits_detached = ops.stop_gradient(logits)
            p = ops.softmax(logits_detached, axis=1)
            
            # Compute entropy: H = -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            p_safe = p + epsilon
            H_i = -ops.reduce_sum(p * ops.log(p_safe), axis=1)
            
            # Compute mean entropy for this image
            H_mean = ops.reduce_mean(H_i)
            
            # Convert to Python scalar using asnumpy
            if hasattr(H_mean, 'asnumpy'):
                H_value = H_mean.asnumpy().item()
            else:
                H_value = H_mean.item()
            
            H.append(H_value)
        
        # Return as tensor on CPU (using numpy as per exemption #3)
        H_array = np.array(H, dtype=np.float32)
        return Tensor(H_array, ms.float32)