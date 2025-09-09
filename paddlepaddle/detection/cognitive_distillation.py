import paddle
import paddle.nn as nn


def total_variation_loss(img, weight=1):
    b, c, h, w = img.shape
    tv_h = paddle.pow(img[:, :, 1:, :]-img[:, :, :-1, :], 2).sum(axis=[1, 2, 3])
    tv_w = paddle.pow(img[:, :, :, 1:]-img[:, :, :, :-1], 2).sum(axis=[1, 2, 3])
    return weight*(tv_h+tv_w)/(c*h*w)


class CognitiveDistillation:
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.l1 = nn.L1Loss(reduction='none')
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = False
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (paddle.tanh(mask) + 1) / 2
        return mask
    
    def __call__(self, model, images, labels=None):
        return self.forward(model, images, labels=labels)

    def forward(self, model, images, preprocessor=None, labels=None):
        model.eval()
        if paddle.min(images) < 0 or paddle.max(images) > 1:
            raise TypeError('images should be normalized')
        b, c, h, w = images.shape
        mask = paddle.ones([b, self.mask_channel, h, w])
        mask_param = paddle.create_parameter(shape=mask.shape, dtype=mask.dtype, default_initializer=paddle.nn.initializer.Assign(mask))
        optimizerR = paddle.optimizer.Adam(parameters=[mask_param], learning_rate=self.lr, beta1=0.1, beta2=0.1)
        
        if self.get_features:
            features, logits = model(images if preprocessor is None else preprocessor(images))
        else:
            logits = model(images if preprocessor is None else preprocessor(images)).detach()
        for step in range(self.num_steps):
            optimizerR.clear_grad()
            mask = self.get_raw_mask(mask_param)
            x_adv = images * mask + (1-mask) * paddle.rand([b, c, 1, 1])
            if self.get_features:
                adv_fe, adv_logits = model(x_adv)
                if len(adv_fe[-2].shape) == 4:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(axis=[1, 2, 3])
                else:
                    loss = self.l1(adv_fe[-2], features[-2].detach()).mean(axis=1)
            else:
                adv_logits = model(x_adv)
                loss = self.l1(adv_logits, logits).mean(axis=1)
            mask_flat = mask.reshape([b, -1])
            norm = paddle.norm(mask_flat, p=self.p, axis=1)
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()
        mask = self.get_raw_mask(mask_param).detach().cpu()
        if self.norm_only:
            mask_flat = mask.reshape([mask.shape[0], -1])
            return paddle.norm(mask_flat, p=1, axis=1)
        return mask.detach()