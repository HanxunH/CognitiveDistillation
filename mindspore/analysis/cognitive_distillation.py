import numpy as np
import mindspore as ms
import mindspore.numpy as msnp
import mindspore.ops as ops


def min_max_normalization(x):
    # Compute min and max values across all elements
    x_min = ops.reduce_min(x)
    x_max = ops.reduce_max(x)
    # Calculate normalization
    denominator = x_max - x_min
    # Use ops.select for conditional operation to handle zero division
    norm = ops.select(
        denominator > 0,
        (x - x_min) / denominator,
        ops.zeros_like(x)
    )
    return norm


class CognitiveDistillationAnalysis():
    def __init__(self, od_type='l1_norm', norm_only=False):
        self.od_type = od_type
        self.norm_only = norm_only
        self.mean = None
        self.std = None
        return

    def train(self, data):
        # Convert numpy array or other format to MindSpore tensor if needed
        if not isinstance(data, ms.Tensor):
            data = ms.Tensor(data)
            
        if not self.norm_only:
            # Calculate L1 norm along dimensions [1, 2, 3]
            # MindSpore doesn't have a direct norm function with multi-axis support
            # So we use sum of absolute values (L1 norm definition)
            data_abs = ops.abs(data)
            data = ops.reduce_sum(data_abs, axis=(1, 2, 3))
        
        self.mean = ops.mean(data).asnumpy().item()
        # MindSpore's std uses biased estimator (ddof=0) by default
        # PyTorch uses unbiased estimator (ddof=1) by default
        # Calculate unbiased std to match PyTorch behavior
        n = data.shape[0]
        if n > 1:
            # Convert biased std to unbiased std: std_unbiased = std_biased * sqrt(n / (n-1))
            biased_std = ops.std(data).asnumpy().item()
            self.std = biased_std * np.sqrt(n / (n - 1))
        else:
            self.std = ops.std(data).asnumpy().item()
        return

    def predict(self, data, t=1):
        # Convert numpy array or other format to MindSpore tensor if needed
        if not isinstance(data, ms.Tensor):
            data = ms.Tensor(data)
            
        if not self.norm_only:
            # Calculate L1 norm along dimensions [1, 2, 3]
            data_abs = ops.abs(data)
            data = ops.reduce_sum(data_abs, axis=(1, 2, 3))
        
        p = (self.mean - data) / self.std
        # MindSpore where syntax: ops.where(condition, x, y)
        p = ops.where((p > t) & (p > 0), 
                      ops.ones_like(p), 
                      ops.zeros_like(p))
        return p.asnumpy()

    def analysis(self, data, is_test=False):
        """
            data (ms.Tensor) b,c,h,w
            data is the distilled mask or pattern extracted by CognitiveDistillation (ms.Tensor)
        """
        # Convert numpy array or other format to MindSpore tensor if needed
        if not isinstance(data, ms.Tensor):
            data = ms.Tensor(data)
            
        if self.norm_only:
            if len(data.shape) > 1:
                # Calculate L1 norm along dimensions [1, 2, 3]
                data_abs = ops.abs(data)
                data = ops.reduce_sum(data_abs, axis=(1, 2, 3))
            score = data
        else:
            # Calculate L1 norm along dimensions [1, 2, 3]
            data_abs = ops.abs(data)
            score = ops.reduce_sum(data_abs, axis=(1, 2, 3))
        
        score = min_max_normalization(score)
        return 1 - score.asnumpy()  # Lower for BD