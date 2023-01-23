import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class STRIP_Detection(nn.Module):
    def __init__(self, data, alpha=1.0, beta=1.0, n=100):
        super(STRIP_Detection, self).__init__()
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.n = n

    def _superimpose(self, background, overlay):
        # cv2.addWeighted(background, 1, overlay, 1, 0)
        imgs = self.alpha * background + self.beta * overlay
        imgs = torch.clamp(imgs, 0, 1)
        return imgs

    def forward(self, model, imgs, labels=None):
        # Return Entropy H
        idx = np.random.randint(0, self.data.shape[0], size=self.n)
        H = []
        for img in imgs:
            x = torch.stack([img] * self.n).to(imgs.device)
            for i in range(self.n):
                x_0 = x[i]
                x_1 = self.data[idx[i]].to(imgs.device)
                x_2 = self._superimpose(x_0, x_1)
                x[i] = x_2
            logits = model(x)
            p = F.softmax(logits.detach(), dim=1)
            H_i = - torch.sum(p * torch.log(p), dim=1)
            H.append(H_i.mean().item())
        return torch.tensor(H).detach().cpu()
