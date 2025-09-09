import tensorflow as tf
import numpy as np


def total_variation_loss(img, weight=1):
    # img shape: [b, h, w, c] in TensorFlow (channels last)
    b, h, w, c = tf.shape(img)[0], tf.shape(img)[1], tf.shape(img)[2], tf.shape(img)[3]
    # Convert to float for shape calculation
    h_f = tf.cast(h, tf.float32)
    w_f = tf.cast(w, tf.float32)
    c_f = tf.cast(c, tf.float32)
    
    tv_h = tf.reduce_sum(tf.pow(img[:, 1:, :, :] - img[:, :-1, :, :], 2), axis=[1, 2, 3])
    tv_w = tf.reduce_sum(tf.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2), axis=[1, 2, 3])
    return weight * (tv_h + tv_w) / (c_f * h_f * w_f)


class CognitiveDistillation:
    """Detection class not inheriting from tf.keras.Model per exemption #7"""
    def __init__(self, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, norm_only=False):
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.lr = lr
        self.mask_channel = mask_channel
        self.get_features = False
        self._EPSILON = 1.e-6
        self.norm_only = norm_only

    def get_raw_mask(self, mask):
        mask = (tf.tanh(mask) + 1) / 2
        return mask

    def __call__(self, model, images, preprocessor=None, labels=None):
        """Implement __call__ with same parameters as PyTorch forward()"""
        if preprocessor is None:
            preprocessor = lambda x: x  # Identity function
            
        # Check image normalization
        if tf.reduce_min(images) < 0 or tf.reduce_max(images) > 1:
            # Use TypeError to match original raise() behavior per exemption #1
            raise TypeError('images should be normalized')
        
        # TensorFlow uses channels-last format by default, need to handle this
        # Assuming images come in as [b, c, h, w] from PyTorch dataset
        if len(images.shape) == 4 and images.shape[1] in [1, 3]:  # Likely channels-first
            images = tf.transpose(images, [0, 2, 3, 1])  # Convert to [b, h, w, c]
        
        b, h, w, c_img = images.shape
        
        # Create mask in channels-last format
        mask = tf.ones([b, h, w, self.mask_channel], dtype=tf.float32)
        mask_var = tf.Variable(mask, trainable=True)
        
        # TensorFlow optimizer
        optimizer = tf.optimizers.Adam(learning_rate=self.lr, beta_1=0.1, beta_2=0.1)
        
        # Get initial predictions
        if self.get_features:
            # For models that return features, we need to handle this case
            features, logits = model(preprocessor(images), training=False)
        else:
            # TensorFlow models expect channels-last format, which we already have
            logits = model(preprocessor(images), training=False)
        
        # Optimization loop
        for step in range(self.num_steps):
            with tf.GradientTape() as tape:
                mask = self.get_raw_mask(mask_var)
                
                # Generate random values for background
                rand_bg = tf.random.uniform([b, 1, 1, self.mask_channel], 0, 1, dtype=tf.float32)
                
                # Apply mask (in channels-last format)
                x_adv = images * mask + (1 - mask) * rand_bg
                
                # TensorFlow models expect channels-last format
                
                if self.get_features:
                    adv_fe, adv_logits = model(preprocessor(x_adv), training=False)
                    # L1 loss on features
                    if len(adv_fe[-2].shape) == 4:
                        # Feature maps case - need to handle channel ordering
                        loss = tf.reduce_mean(tf.abs(adv_fe[-2] - features[-2]), axis=[1, 2, 3])
                    else:
                        # FC features case
                        loss = tf.reduce_mean(tf.abs(adv_fe[-2] - features[-2]), axis=1)
                else:
                    adv_logits = model(preprocessor(x_adv), training=False)
                    # L1 loss on logits
                    loss = tf.reduce_mean(tf.abs(adv_logits - logits), axis=1)
                
                # Compute mask norm (convert mask to channels-first for norm calculation)
                mask_for_norm = tf.transpose(mask, [0, 3, 1, 2])
                if self.p == 1:
                    norm = tf.reduce_sum(tf.abs(mask_for_norm), axis=[1, 2, 3])
                elif self.p == 2:
                    norm = tf.sqrt(tf.reduce_sum(tf.square(mask_for_norm), axis=[1, 2, 3]))
                else:
                    norm = tf.pow(tf.reduce_sum(tf.pow(tf.abs(mask_for_norm), self.p), axis=[1, 2, 3]), 1.0/self.p)
                
                norm = norm * self.gamma
                
                # Total variation loss (mask is already in channels-last)
                tv_loss = total_variation_loss(mask, weight=self.beta)
                
                # Total loss
                loss_total = loss + norm + tv_loss
                loss_mean = tf.reduce_mean(loss_total)
            
            # Compute gradients and update
            gradients = tape.gradient(loss_mean, [mask_var])
            optimizer.apply_gradients(zip(gradients, [mask_var]))
        
        # Get final mask
        mask = self.get_raw_mask(mask_var)
        
        # Convert back to channels-first format for output
        mask = tf.transpose(mask, [0, 3, 1, 2])
        
        if self.norm_only:
            # Return L1 norm of mask
            return tf.reduce_sum(tf.abs(mask), axis=[1, 2, 3])
        
        return mask