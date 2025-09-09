"""
MindSpore implementation of Dynamic CIFAR-10 dataset.

This module provides a backdoored CIFAR-10 dataset with dynamic triggers.
According to mindspore-note.md #5 and #8, we continue to use PyTorch datasets 
as base and wrap them with TorchDatasetWrapper for MindSpore compatibility.
Pre-trained PyTorch models are used for trigger generation.
"""

import torch
import numpy as np
import mindspore as ms
from torchvision import datasets
from torchvision import transforms

# Import PyTorch models for loading pre-trained trigger generators
# According to mindspore-note.md #8, we keep using PyTorch for pre-trained models
import models

# Set device context - using framework default per mindspore-exemptions.md #3
ms.set_context(mode=ms.PYNATIVE_MODE)

# Set device for PyTorch models
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def create_bd(netG, netM, inputs, targets, opt):
    """
    Create backdoor patterns and masks using pre-trained PyTorch models.
    
    Args:
        netG: Generator network for patterns (PyTorch model)
        netM: Generator network for masks (PyTorch model)
        inputs: Input images (torch tensor)
        targets: Target labels
        opt: Options from the checkpoint
    
    Returns:
        patterns: Generated patterns
        masks_output: Generated masks after thresholding
    """
    patterns = netG(inputs)
    masks_output = netM.threshold(netM(inputs))
    return patterns, masks_output


class DynamicCIFAR10(datasets.CIFAR10):
    """
    Dynamic backdoored CIFAR-10 dataset for MindSpore.
    
    This class inherits from torchvision's CIFAR10 dataset and adds dynamic backdoor triggers.
    The dataset will be wrapped with TorchDatasetWrapper when used with MindSpore's
    DatasetGenerator for proper format conversion.
    
    Note: The trigger generation uses pre-trained PyTorch models, following mindspore-note.md #8.
    
    Args:
        root: Root directory of dataset where CIFAR-10 exists or will be downloaded.
        train: If True, creates dataset from training set, otherwise from test set.
        transform: A function/transform that takes in a PIL image and returns a 
                  transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        download: If True, downloads the dataset from the internet if not available.
        poison_rate: Fraction of samples to poison with backdoor trigger.
        target_label: The target label for backdoored samples.
        **kwargs: Additional keyword arguments.
    """
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform,
                         target_transform=target_transform)

        # Load dynamic trigger model (PyTorch pre-trained models)
        # Use absolute path or path relative to CognitiveDistillation root
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ckpt_path = os.path.join(root_dir, 'trigger/all2one_cifar10_ckpt.pth.tar')
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        opt = state_dict["opt"]
        
        # Load generator for patterns (using PyTorch models)
        netG = models.dynamic_models.Generator(opt).to(device)
        netG.load_state_dict(state_dict["netG"])
        netG = netG.eval()
        
        # Load generator for masks (using PyTorch models)
        netM = models.dynamic_models.Generator(opt, out_channels=1).to(device)
        netM.load_state_dict(state_dict["netM"])
        netM = netM.eval()
        
        # Normalizer for preprocessing images before trigger generation
        normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465],
                                          [0.247, 0.243, 0.261])

        # Select backdoor index
        s = len(self)
        if not train:
            # For test set, poison samples that are not originally the target label
            idx = np.where(np.array(self.targets) != target_label)[0]
            self.poison_idx = np.random.choice(idx, size=int(s * poison_rate), replace=False)
        else:
            # For training set, randomly select samples to poison
            self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]

        # Add triggers to selected samples
        for i in self.poison_idx:
            x = self.data[i]
            y = self.targets[i]
            
            # Convert to torch tensor and normalize for processing
            x = torch.tensor(x).permute(2, 0, 1) / 255.0
            x_in = torch.stack([normalizer(x)]).to(device)
            
            # Generate pattern and mask using PyTorch models
            p, m = create_bd(netG, netM, x_in, y, opt)
            p = p[0, :, :, :].detach().cpu()
            m = m[0, :, :, :].detach().cpu()
            
            # Apply the backdoor pattern
            x_bd = x + (p - x) * m
            
            # Convert back to numpy array format expected by dataset
            x_bd = x_bd.permute(1, 2, 0).numpy() * 255
            x_bd = x_bd.astype(np.uint8)
            self.data[i] = x_bd

        # Convert targets to numpy array and update labels for poisoned samples
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Inject: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.5f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx), len(self.poison_idx)/len(self)))


# Note: This dataset class will be used through the TorchDatasetWrapper
# when integrated with MindSpore's DatasetGenerator, as specified in
# mindspore_impl/datasets/dataset.py. The wrapper handles all necessary
# tensor format conversions between PyTorch and MindSpore.