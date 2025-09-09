import numpy as np
from torchvision.datasets import CIFAR10 as TorchCIFAR10
from PIL import Image


class CIFAR10Wrapper(TorchCIFAR10):
    """Wrapper for torchvision CIFAR10 to work with PaddlePaddle DataLoader"""
    
    def __getitem__(self, index):
        """
        Override to handle data format conversion for PaddlePaddle.
        Returns numpy arrays that PaddlePaddle DataLoader can handle.
        """
        img, target = self.data[index], self.targets[index]
        
        # img is a numpy array of shape (32, 32, 3) 
        # Convert to PIL for transforms
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # Convert torch tensor to numpy array if necessary
        if hasattr(img, 'numpy'):  # torch tensor
            img = img.numpy()
        elif isinstance(img, Image.Image):  # PIL Image
            # Convert PIL to numpy array in CHW format
            img = np.array(img)
            if len(img.shape) == 3:  # HWC to CHW
                img = img.transpose(2, 0, 1)
            img = img.astype(np.float32) / 255.0
        
        return img, target