import tensorflow as tf
import numpy as np


class STRIP_Detection:
    """Detection class not inheriting from tf.keras.Model per exemption #7"""
    def __init__(self, data, alpha=1.0, beta=1.0, n=100):
        # Convert data to TensorFlow tensor if it's a numpy array
        if isinstance(data, np.ndarray):
            self.data = tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            self.data = data
        self.alpha = alpha
        self.beta = beta
        self.n = n

    def _superimpose(self, background, overlay):
        """Superimpose overlay on background with alpha-beta blending"""
        # cv2.addWeighted(background, 1, overlay, 1, 0)
        imgs = self.alpha * background + self.beta * overlay
        imgs = tf.clip_by_value(imgs, 0, 1)
        return imgs

    def __call__(self, model, imgs, labels=None):
        """
        Implement __call__ with same parameters as PyTorch forward()
        
        Args:
            model: TensorFlow model for inference
            imgs: Input images tensor
            labels: Optional labels (not used in STRIP)
            
        Returns:
            numpy array of entropy values (H)
        """
        # Convert imgs to TensorFlow tensor if needed
        if isinstance(imgs, np.ndarray):
            imgs = tf.convert_to_tensor(imgs, dtype=tf.float32)
        
        # Handle channel format conversion if needed
        # Assuming imgs comes in as [b, c, h, w] from PyTorch dataset
        if len(imgs.shape) == 4 and imgs.shape[1] in [1, 3]:  # Likely channels-first
            imgs = tf.transpose(imgs, [0, 2, 3, 1])  # Convert to [b, h, w, c]
        
        # Similarly handle data tensor format
        data_tensor = self.data
        if len(data_tensor.shape) == 4 and data_tensor.shape[1] in [1, 3]:  # Likely channels-first
            data_tensor = tf.transpose(data_tensor, [0, 2, 3, 1])  # Convert to [b, h, w, c]
        
        # Generate random indices
        idx = np.random.randint(0, data_tensor.shape[0], size=self.n)
        
        H = []
        for i in range(imgs.shape[0]):  # For each image
            img = imgs[i]
            
            # Create n copies of the image
            x = tf.stack([img] * self.n, axis=0)
            
            # Superimpose with random samples
            x_list = []
            for j in range(self.n):
                x_0 = x[j]
                x_1 = data_tensor[idx[j]]
                x_2 = self._superimpose(x_0, x_1)
                x_list.append(x_2)
            
            # Stack all superimposed images
            x_batch = tf.stack(x_list, axis=0)
            
            # Get model predictions
            logits = model(x_batch, training=False)
            
            # Compute softmax probabilities
            p = tf.nn.softmax(logits, axis=1)
            
            # Compute entropy: H = -sum(p * log(p))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-8
            H_i = -tf.reduce_sum(p * tf.math.log(p + epsilon), axis=1)
            
            # Take mean entropy for this image
            H.append(tf.reduce_mean(H_i).numpy())
        
        # Return as numpy array to match original behavior
        return np.array(H)