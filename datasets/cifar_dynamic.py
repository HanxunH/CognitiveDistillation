import torch
import numpy as np
import models
from torchvision import datasets
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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
        state_dict = torch.load(ckpt_path, map_location=device)
        opt = state_dict["opt"]
        netG = models.dynamic_models.Generator(opt).to(device)
        netG.load_state_dict(state_dict["netG"])
        netG = netG.eval()
        netM = models.dynamic_models.Generator(opt, out_channels=1).to(device)
        netM.load_state_dict(state_dict["netM"])
        netM = netM.eval()
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
            x = torch.tensor(x).permute(2, 0, 1) / 255.0
            x_in = torch.stack([normalizer(x)]).to(device)
            p, m = create_bd(netG, netM, x_in, y, opt)
            p = p[0, :, :, :].detach().cpu()
            m = m[0, :, :, :].detach().cpu()
            x_bd = x + (p - x) * m
            x_bd = x_bd.permute(1, 2, 0).numpy() * 255
            x_bd = x_bd.astype(np.uint8)
            self.data[i] = x_bd

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
