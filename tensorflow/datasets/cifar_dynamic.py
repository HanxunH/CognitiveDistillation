import torch
import numpy as np
import tensorflow as tf
from torchvision import datasets
from torchvision import transforms
from PIL import Image
import sys
import os

# Ensure project root is in path for model imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def create_bd(netG, netM, inputs, targets, opt):
    """Keep using PyTorch for trigger generation as models are pretrained."""
    patterns = netG(inputs)
    masks_output = netM.threshold(netM(inputs))
    return patterns, masks_output


class DynamicCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Load dynamic trigger model - keep using PyTorch since models are pretrained
        ckpt_path = 'trigger/all2one_cifar10_ckpt.pth.tar'
        # Use weights_only=False for compatibility with older checkpoints
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        opt = state_dict["opt"]
        # Import TensorFlow Generator from tensorflow_impl
        # This wraps the PyTorch models and converts outputs to TensorFlow tensors
        # Use absolute import to avoid relative import issues
        from tensorflow_impl.models.dynamic_models import Generator
        netG = Generator(opt).to(device)
        netG.load_state_dict(state_dict["netG"])
        netG = netG.eval()
        netM = Generator(opt, out_channels=1).to(device)
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

        # Add triggers using PyTorch models
        for i in self.poison_idx:
            x = self.data[i]
            y = self.targets[i]
            x = torch.tensor(x).permute(2, 0, 1) / 255.0
            x_in = torch.stack([normalizer(x)]).to(device)
            p, m = create_bd(netG, netM, x_in, y, opt)
            # p and m are TensorFlow tensors on GPU, convert to numpy
            # We need to copy to CPU first
            with tf.device('/CPU:0'):
                p_cpu = tf.identity(p[0, :, :, :])
                m_cpu = tf.identity(m[0, :, :, :])
            p = p_cpu.numpy()
            m = m_cpu.numpy()
            # Convert to torch tensors for computation
            p = torch.from_numpy(p)
            m = torch.from_numpy(m)
            x_bd = x + (p - x) * m
            x_bd = x_bd.permute(1, 2, 0).numpy() * 255
            x_bd = x_bd.astype(np.uint8)
            self.data[i] = x_bd

        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))
    
    def __getitem__(self, index):
        """
        Get item with format conversion (tensorflow-exemptions.md item 4).
        Convert data to numpy arrays compatible with TensorFlow.
        """
        # Get the original item from parent class
        img, target = super().__getitem__(index)
        
        # Handle different types of img
        if isinstance(img, Image.Image):
            # PIL Image - convert to numpy array
            img = np.array(img, dtype=np.float32) / 255.0
            # Transpose from HWC to CHW to match ToTensor behavior
            if len(img.shape) == 3:
                img = np.transpose(img, (2, 0, 1))
            elif len(img.shape) == 2:
                # Grayscale image, add channel dimension
                img = np.expand_dims(img, axis=0)
        elif hasattr(img, 'numpy'):
            # torch.Tensor - convert to numpy
            img = img.numpy()
        elif isinstance(img, np.ndarray):
            # Already numpy array
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            else:
                img = img.astype(np.float32)
            # Transpose from HWC to CHW if needed
            if len(img.shape) == 3 and img.shape[-1] in [1, 3]:
                img = np.transpose(img, (2, 0, 1))
            elif len(img.shape) == 2:
                # Grayscale image, add channel dimension
                img = np.expand_dims(img, axis=0)
        
        # Convert target to numpy if needed
        if hasattr(target, 'numpy'):
            target = target.numpy()
        elif hasattr(target, 'item'):
            target = target.item()
        
        # Ensure proper data types
        img = img.astype(np.float32)
        target = np.int64(target)
        
        return img, target