import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Feature_Detection:
    """
    Feature extraction for detections (AC and SS methods).
    
    This class does NOT inherit from nn.Cell as per exemption #7 in mindspore-exemptions.md
    to maintain the same calling interface as the PyTorch version.
    """
    
    def __init__(self):
        # Feature extraction for detections
        pass
    
    def __call__(self, model, images, labels):
        """
        Extract features from the model's last hidden layer.
        
        Args:
            model: MindSpore model with get_features capability
            images: Input images tensor
            labels: Input labels (not used but kept for interface compatibility)
            
        Returns:
            features: Features from the last hidden layer
        """
        # Handle DataParallel models (skip parallel processing as per requirements)
        if hasattr(model, 'module'):
            raise NotImplementedError("Skip")  # Skip parallel processing
        
        # Save training state and set to eval mode
        prev_training = model.training
        model.set_train(False)
        
        # Enable feature extraction mode
        model.get_features = True
        
        # Extract features without gradient computation
        features, _ = model(images)
        features = features[-1]  # activations of last hidden layer
        
        # Use stop_gradient to ensure no gradients flow through features
        features = ops.stop_gradient(features)
        
        # Disable feature extraction mode
        model.get_features = False
        
        # Restore training state
        model.set_train(prev_training)
        
        return features