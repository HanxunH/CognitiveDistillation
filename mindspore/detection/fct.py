import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
import torch  # Need for torchvision transforms output conversion

# Set MindSpore context 
ms.set_context(mode=ms.PYNATIVE_MODE)


class FCT_Detection:
    """
    Feature Consistency Towards transformations detection method.
    
    This class does NOT inherit from nn.Cell as per exemption #7 in mindspore-exemptions.md
    to maintain the same calling interface as the PyTorch version.
    """
    
    def __init__(self, model, train_loader):
        super(FCT_Detection, self).__init__()
        # Feature consistency towards transformations
        self.model = model 
        self.train_loader = train_loader
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/utils/dataloader_bd.py
        # Per exemption #8: Can use torchvision.transforms for image augmentation
        self.transforms_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
        ])
        # Finetune with L_intra
        # Set model to training mode for finetuning
        self.model.set_train(True)
        self.finetune_l_intra()
        self.model.set_train(False)

    def finetune_l_intra(self):
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/finetune_attack_noTrans.py
        # Skip DataParallel check as per distributed code skip requirement
        self.model.get_features = True
        
        # Create optimizer - using MindSpore's SGD
        optimizer = nn.SGD(self.model.trainable_params(), learning_rate=0.01, momentum=0.9, weight_decay=5e-4)
        
        # Define loss function for gradient computation
        def loss_fn(images, labels):
            features, logits = self.model(images)
            features = features[-1]
            
            # Calculate intra-class loss
            centers = []
            for j in range(logits.shape[1]):
                # MindSpore where returns indices differently than PyTorch
                j_mask = ops.equal(labels, j)
                # Check if any samples belong to class j
                if not ops.any(j_mask):
                    continue
                    
                # Use boolean indexing to get features for class j
                j_features = features[j_mask]
                j_center = ops.mean(j_features, axis=0)
                centers.append(j_center)
            
            if len(centers) == 0:
                # If no centers computed, return zero loss
                return Tensor(0.0, dtype=ms.float32)
                
            centers = ops.stack(centers, axis=0)
            # Normalize centers
            centers = ops.L2Normalize(axis=1)(centers)
            similarity_matrix = ops.matmul(centers, ops.transpose(centers, (1, 0)))
            
            # Create eye mask
            eye_mask = ops.eye(similarity_matrix.shape[0], dtype=ms.bool_)
            # Set diagonal to 0
            similarity_matrix = ops.masked_fill(similarity_matrix, eye_mask, 0.0)
            loss = ops.mean(similarity_matrix)
            return loss
        
        # Create gradient function
        grad_fn = ops.value_and_grad(loss_fn, None, optimizer.parameters, has_aux=False)
        
        for epoch in range(10):
            pbar = tqdm(self.train_loader.create_tuple_iterator())
            for images, labels in pbar:
                # Compute loss and gradients
                loss_val, grads = grad_fn(images, labels)
                # Update parameters
                optimizer(grads)
                
                pbar.set_description("Loss {:.4f}".format(loss_val.asnumpy().item()))
        
        # Reset get_features flag
        self.model.get_features = False
    
    def transforms(self, images):
        """
        Apply transformations to images.
        Per exemption #8: Use torchvision.transforms and convert tensors at boundaries.
        """
        new_imgs = []
        # Convert MindSpore tensor to numpy for torchvision transforms
        images_np = images.asnumpy()
        
        for img in images_np:
            # torchvision transforms expect channel-first format
            # and values in [0, 1] range which should already be the case
            img_torch = torch.from_numpy(img)
            transformed = self.transforms_op(img_torch)
            # Convert back to numpy then to MindSpore tensor
            new_imgs.append(transformed.numpy())
        
        new_imgs = np.stack(new_imgs)
        return Tensor(new_imgs, dtype=ms.float32)
    
    def forward(self, model, images, labels):
        """
        Forward pass maintaining same interface as PyTorch version.
        
        Args:
            model: The model to use for feature extraction
            images: Input images tensor
            labels: Input labels (not used but kept for interface compatibility)
        
        Returns:
            Feature consistency scores
        """
        # Enable feature extraction
        self.model.get_features = True
        
        # Apply transformations
        aug_imgs = self.transforms(images)
        
        # Calculate feature consistency
        # No need for torch.no_grad() equivalent - MindSpore handles this differently
        self.model.set_train(False)
        
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/calculate_consistency.py
        features1, _ = model(images)
        features2, _ = model(aug_imgs)
        features1 = features1[-1]  # activations of last hidden layer
        features2 = features2[-1]  # activations of last hidden layer
        
        ### Calculate consistency ###
        feature_consistency = ops.mean(ops.pow(features1 - features2, 2), axis=1)
        
        # Reset get_features flag
        self.model.get_features = False
        
        return feature_consistency
    
    def __call__(self, model, images, labels=None):
        """
        Call method to maintain same interface as PyTorch version.
        """
        return self.forward(model, images, labels)