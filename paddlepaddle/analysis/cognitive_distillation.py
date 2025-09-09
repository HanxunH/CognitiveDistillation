import paddle
import numpy as np

def min_max_normalization(x):
    x_min = paddle.min(x)
    x_max = paddle.max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


class CognitiveDistillationAnalysis():
    def __init__(self, od_type='l1_norm', norm_only=False):
        self.od_type = od_type
        self.norm_only = norm_only
        self.mean = None
        self.std = None
        return

    def train(self, data):
        if not self.norm_only:
            data = paddle.flatten(data, start_axis=1)
            data = paddle.norm(data, axis=1, p=1)
        self.mean = paddle.mean(data).item()
        self.std = paddle.std(data, unbiased=True).item()
        return

    def predict(self, data, t=1):
        if not self.norm_only:
            data = paddle.flatten(data, start_axis=1)
            data = paddle.norm(data, axis=1, p=1)
        p = (self.mean - data) / self.std
        p = paddle.where((p > t) & (p > 0), paddle.ones_like(p, dtype='int32'), paddle.zeros_like(p, dtype='int32'))
        return p.numpy()

    def analysis(self, data, is_test=False):
        """
            data (paddle.Tensor) b,c,h,w
            data is the distilled mask or pattern extracted by CognitiveDistillation (paddle.Tensor)
        """
        if self.norm_only:
            if len(data.shape) > 1:
                data = paddle.flatten(data, start_axis=1)
                data = paddle.norm(data, axis=1, p=1)
            score = data
        else:
            # PaddlePaddle's norm doesn't support axis list > 2, need to flatten first
            data_flat = paddle.flatten(data, start_axis=1)
            score = paddle.norm(data_flat, axis=1, p=1)
        score = min_max_normalization(score)
        return 1 - score.numpy()  # Lower for BD