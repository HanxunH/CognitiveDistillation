import tensorflow as tf
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm


class FCT_Detection:
    """Detection class not inheriting from tf.keras.Model per exemption #7"""
    def __init__(self, model, train_loader):
        # Feature consistency towards transformations
        self.model = model 
        self.train_loader = train_loader
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/utils/dataloader_bd.py
        # Per exemption #8, we can use torchvision.transforms for image augmentation
        self.transforms_op = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            transforms.ToTensor(),
        ])
        # Finetune with L_intra
        # Set model to training mode
        self.model.trainable = True
        self.finetune_l_intra()
        # Set model back to evaluation mode
        self.model.trainable = False

    def finetune_l_intra(self):
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/finetune_attack_noTrans.py
        # Skip DataParallel check since TensorFlow handles device placement differently
        self.model.get_features = True

        # Use TensorFlow optimizer with same parameters as PyTorch
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        # Note: TensorFlow's weight_decay is handled differently, we'll apply it manually if needed

        for epoch in range(10):
            pbar = tqdm(self.train_loader)
            for images, labels in pbar:
                # Convert to TensorFlow tensors if needed
                if not isinstance(images, tf.Tensor):
                    images = tf.convert_to_tensor(images, dtype=tf.float32)
                if not isinstance(labels, tf.Tensor):
                    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
                
                # Ensure images are in channels-last format for TensorFlow
                if len(images.shape) == 4 and images.shape[1] in [1, 3]:  # Likely channels-first
                    images = tf.transpose(images, [0, 2, 3, 1])  # Convert to [b, h, w, c]
                
                # Use GradientTape for gradient computation (exemption #13)
                with tf.GradientTape() as tape:
                    # Features and Outputs
                    features, logits = self.model(images, training=True)
                    features = features[-1]
                    
                    # Calculate intra-class loss
                    centers = []
                    num_classes = logits.shape[1]
                    for j in range(num_classes):
                        # Find indices where label equals j
                        j_idx = tf.where(labels == j)
                        if tf.shape(j_idx)[0] == 0:
                            continue
                        # Gather features for class j
                        j_idx = tf.squeeze(j_idx, axis=1)
                        j_features = tf.gather(features, j_idx)
                        j_center = tf.reduce_mean(j_features, axis=0)
                        centers.append(j_center)
                    
                    if len(centers) > 0:
                        centers = tf.stack(centers, axis=0)
                        # Normalize centers
                        centers = tf.nn.l2_normalize(centers, axis=1)
                        similarity_matrix = tf.matmul(centers, centers, transpose_b=True)
                        # Create mask for diagonal elements
                        mask = tf.eye(tf.shape(similarity_matrix)[0], dtype=tf.bool)
                        # Set diagonal to 0
                        similarity_matrix = tf.where(mask, 0.0, similarity_matrix)
                        loss = tf.reduce_mean(similarity_matrix)
                    else:
                        # No valid centers found, skip this batch
                        continue
                    
                    # Add L2 regularization (weight decay)
                    weight_decay = 5e-4
                    l2_loss = 0.0
                    for var in self.model.trainable_variables:
                        if 'kernel' in var.name:  # Only apply to weights, not biases
                            l2_loss += tf.nn.l2_loss(var)
                    loss += weight_decay * l2_loss
                
                # Compute and apply gradients
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                pbar.set_description("Loss {:.4f}".format(float(loss)))

        # Reset get_features flag
        self.model.get_features = False

    def transforms(self, images):
        """Apply transformations to images using torchvision.transforms"""
        # Per exemption #8, we use torchvision.transforms and convert at boundaries
        import torch
        new_imgs = []
        
        # Handle batch of images - iterate over first dimension
        batch_size = images.shape[0]
        for i in range(batch_size):
            # Get single image
            img = images[i]
            
            # Convert TensorFlow tensor to numpy for torchvision
            if isinstance(img, tf.Tensor):
                img_np = img.numpy()
            else:
                img_np = img
            
            # Determine current format and convert to CHW if needed
            # torchvision.transforms expects CHW format
            if len(img_np.shape) == 3:
                # Check if it's HWC (height, width, channels)
                if img_np.shape[-1] in [1, 3]:  # Last dim is likely channels
                    # HWC -> CHW
                    img_np = np.transpose(img_np, (2, 0, 1))
                # If shape[0] is 1 or 3, it's already CHW
                elif img_np.shape[0] not in [1, 3]:
                    # Ambiguous case, but likely HWC if first dim is not 1 or 3
                    img_np = np.transpose(img_np, (2, 0, 1))
            
            # Convert numpy to torch tensor for transforms
            # ToPILImage expects torch tensors or uint8 numpy arrays
            img_torch = torch.from_numpy(img_np)
            
            # Apply transforms (returns torch.Tensor)
            transformed = self.transforms_op(img_torch)
            # Convert torch.Tensor back to numpy
            transformed_np = transformed.numpy()
            new_imgs.append(transformed_np)
        
        # Stack and convert to TensorFlow tensor
        new_imgs = np.stack(new_imgs)
        new_imgs = tf.convert_to_tensor(new_imgs, dtype=tf.float32)
        
        # Convert from CHW to HWC for TensorFlow
        if len(new_imgs.shape) == 4 and new_imgs.shape[1] in [1, 3]:
            new_imgs = tf.transpose(new_imgs, [0, 2, 3, 1])
        
        return new_imgs

    def __call__(self, model, images, labels):
        """Implement __call__ with same parameters as PyTorch forward()"""
        # Enable feature extraction
        self.model.get_features = True
        
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        
        # Ensure images are in channels-last format for TensorFlow
        if len(images.shape) == 4 and images.shape[1] in [1, 3]:  # Likely channels-first
            images = tf.transpose(images, [0, 2, 3, 1])  # Convert to [b, h, w, c]
        
        # Apply transformations
        aug_imgs = self.transforms(images)
        
        # Calculate feature consistency
        # https://github.com/SCLBD/Effective_backdoor_defense/blob/main/calculate_consistency.py
        features1, _ = model(images, training=False)
        features2, _ = model(aug_imgs, training=False)
        features1 = features1[-1]  # activations of last hidden layer
        features2 = features2[-1]  # activations of last hidden layer
        
        ### Calculate consistency ###
        feature_consistency = tf.reduce_mean(tf.square(features1 - features2), axis=1)
        
        # Reset get_features flag
        self.model.get_features = False
        
        return feature_consistency