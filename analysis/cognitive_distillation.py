import torch

def min_max_normalization(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
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
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        self.mean = torch.mean(data).item()
        self.std = torch.std(data).item()
        return

    def predict(self, data, t=1):
        if not self.norm_only:
            data = torch.norm(data, dim=[1, 2, 3], p=1)
        p = (self.mean - data) / self.std
        p = torch.where((p > t) & (p > 0), 1, 0)
        return p.numpy()

    def analysis(self, data, is_test=False):
        """
            data (torch.tensor) b,c,h,w
            data is the distilled mask or pattern extracted by CognitiveDistillation (torch.tensor)
        """
        if self.norm_only:
            if len(data.shape) > 1:
                data = torch.norm(data, dim=[1, 2, 3], p=1)
            score = data
        else:
            score = torch.norm(data, dim=[1, 2, 3], p=1)
        score = min_max_normalization(score)
        return 1 - score.numpy()  # Lower for BD
