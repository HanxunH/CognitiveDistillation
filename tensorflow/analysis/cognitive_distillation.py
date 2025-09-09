import tensorflow as tf
import numpy as np


def min_max_normalization(x):
    """Min-max normalization function for TensorFlow tensors or numpy arrays."""
    # Convert to TensorFlow tensor if numpy array
    if isinstance(x, np.ndarray):
        x = tf.convert_to_tensor(x)
    
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    norm = (x - x_min) / (x_max - x_min)
    return norm


class CognitiveDistillationAnalysis():
    def __init__(self, od_type='l1_norm', norm_only=False):
        self.od_type = od_type
        self.norm_only = norm_only
        self.mean = None
        self.std = None
        return

    def train(self, data):
        """Train the analysis model with data.
        
        Args:
            data: Input data as TensorFlow tensor or numpy array
        """
        # Convert to TensorFlow tensor if numpy array
        if isinstance(data, np.ndarray):
            data = tf.convert_to_tensor(data)
        
        if not self.norm_only:
            # Flatten spatial dimensions and compute L1 norm
            data_flat = tf.reshape(data, [tf.shape(data)[0], -1])
            data = tf.norm(data_flat, axis=1, ord=1)
        self.mean = tf.reduce_mean(data).numpy().item()
        self.std = tf.math.reduce_std(data).numpy().item()
        return

    def predict(self, data, t=1):
        """Predict labels based on trained statistics.
        
        Args:
            data: Input data as TensorFlow tensor or numpy array
            t: Threshold value
            
        Returns:
            numpy array of predictions
        """
        # Convert to TensorFlow tensor if numpy array
        if isinstance(data, np.ndarray):
            data = tf.convert_to_tensor(data)
            
        if not self.norm_only:
            # Flatten spatial dimensions and compute L1 norm
            data_flat = tf.reshape(data, [tf.shape(data)[0], -1])
            data = tf.norm(data_flat, axis=1, ord=1)
        p = (self.mean - data) / self.std
        p = tf.where((p > t) & (p > 0), 1, 0)
        return p.numpy()

    def analysis(self, data, is_test=False):
        """
        Analyze data and return anomaly scores.
        
        Args:
            data: Input data (TensorFlow tensor or numpy array) with shape b,c,h,w
                  data is the distilled mask or pattern extracted by CognitiveDistillation
            is_test: Whether this is test data (unused in implementation)
            
        Returns:
            numpy array of scores (lower scores indicate backdoor samples)
        """
        # Convert to TensorFlow tensor if numpy array
        if isinstance(data, np.ndarray):
            data = tf.convert_to_tensor(data)
            
        if self.norm_only:
            if len(data.shape) > 1:
                # Flatten spatial dimensions and compute L1 norm
                data_flat = tf.reshape(data, [tf.shape(data)[0], -1])
                data = tf.norm(data_flat, axis=1, ord=1)
            score = data
        else:
            # Flatten spatial dimensions and compute L1 norm
            score_flat = tf.reshape(data, [tf.shape(data)[0], -1])
            score = tf.norm(score_flat, axis=1, ord=1)
        score = min_max_normalization(score)
        return 1 - score.numpy()  # Lower for BD