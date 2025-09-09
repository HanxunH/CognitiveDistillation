import tensorflow as tf


class Feature_Detection:
    """Detection class not inheriting from tf.keras.Model per exemption #7"""
    def __init__(self):
        # Feature extraction for detections
        pass

    def __call__(self, model, images, labels):
        """
        Extract features from model's last hidden layer.
        
        Args:
            model: TensorFlow model that supports get_features flag
            images: Input images as TensorFlow tensor
            labels: Labels (not used in feature extraction but kept for interface consistency)
            
        Returns:
            features: Features from the last hidden layer as TensorFlow tensor
        """
        # Check if model is wrapped (e.g., in distributed training)
        # TensorFlow doesn't have exact equivalent to DataParallel, but we handle model wrapping
        if hasattr(model, 'module'):
            model.module.get_features = True
        else:
            model.get_features = True
        
        # Extract features without gradient computation
        # TensorFlow equivalent of torch.no_grad() context - use training=False
        features, _ = model(images, training=False)
        features = features[-1]  # activations of last hidden layer
        
        # Reset the get_features flag
        if hasattr(model, 'module'):
            model.module.get_features = False
        else:
            model.get_features = False
            
        return features