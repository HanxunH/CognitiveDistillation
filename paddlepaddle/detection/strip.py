import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F


class STRIP_Detection(nn.Layer):
    def __init__(self, data, alpha=1.0, beta=1.0, n=100):
        super(STRIP_Detection, self).__init__()
        self.data = data
        self.alpha = alpha
        self.beta = beta
        self.n = n

    def _superimpose(self, background, overlay):
        # cv2.addWeighted(background, 1, overlay, 1, 0)
        imgs = self.alpha * background + self.beta * overlay
        imgs = paddle.clip(imgs, 0, 1)
        return imgs

    def forward(self, model, imgs, labels=None):
        # Return Entropy H
        idx = np.random.randint(0, self.data.shape[0], size=self.n)
        H = []
        for img in imgs:
            x = paddle.stack([img] * self.n)
            for i in range(self.n):
                x_0 = x[i]
                x_1 = self.data[idx[i]]
                x_2 = self._superimpose(x_0, x_1)
                x[i] = x_2
            logits = model(x)
            p = F.softmax(logits.detach(), axis=1)
            H_i = - paddle.sum(p * paddle.log(p), axis=1)
            H.append(H_i.mean().item())
        return paddle.to_tensor(H)