import paddle
import numpy as np
import torch
import models
from torchvision import datasets
from torchvision import transforms


def create_bd(netG, netM, inputs, targets, opt):
    patterns = netG(inputs)
    masks_output = netM.threshold(netM(inputs))
    return patterns, masks_output


class DynamicCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Load dynamic trigger model
        ckpt_path = 'trigger/all2one_cifar10_ckpt.pth.tar'
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        opt = state_dict["opt"]
        
        # Create PaddlePaddle models
        netG = models.dynamic_models.Generator(opt)
        netM = models.dynamic_models.Generator(opt, out_channels=1)
        
        # Convert PyTorch state dict to PaddlePaddle format
        def convert_state_dict(torch_state_dict):
            paddle_state_dict = {}
            for k, v in torch_state_dict.items():
                # Convert torch tensor to numpy then to paddle tensor
                if isinstance(v, torch.Tensor):
                    paddle_state_dict[k] = paddle.to_tensor(v.cpu().numpy())
                else:
                    paddle_state_dict[k] = v
            return paddle_state_dict
        
        netG.set_state_dict(convert_state_dict(state_dict["netG"]))
        netG.eval()
        netM.set_state_dict(convert_state_dict(state_dict["netM"]))
        netM.eval()
        
        normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                          [0.247, 0.243, 0.261])

        # Select backdoor index
        s = len(self)
        if not train:
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add trigers
        for i in self.poison_idx:
            x = self.data[i]
            y = self.targets[i]
            # Convert to torch tensor for normalizer
            x_torch = torch.tensor(x).permute(2, 0, 1) / 255.0
            x_normalized = normalizer(x_torch)
            # Convert to paddle tensor for model
            x_in = paddle.stack([paddle.to_tensor(x_normalized.numpy())])
            p, m = create_bd(netG, netM, x_in, y, opt)
            p = p[0, :, :, :]
            m = m[0, :, :, :]
            # Convert back to numpy for computation
            x_paddle = paddle.to_tensor(x_torch.numpy())
            x_bd = x_paddle + (p - x_paddle) * m
            x_bd = x_bd.transpose([1, 2, 0]).numpy() * 255
            x_bd = x_bd.astype(np.uint8)
            self.data[i] = x_bd

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Convert PIL Image to numpy array if necessary
        img = np.array(img)
        
        if self.transform is not None:
            # Apply transforms except ToTensor
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if not isinstance(t, transforms.ToTensor):
                        img = t(img)
        
        # Manual ToTensor implementation for PaddlePaddle
        # Convert HWC to CHW and scale to [0, 1]
        img = img.astype(np.float32) / 255.0
        if len(img.shape) == 3:
            img = img.transpose(2, 0, 1)
        elif len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target