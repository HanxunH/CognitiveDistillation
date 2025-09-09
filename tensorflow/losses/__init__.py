import mlconfig
import tensorflow as tf

# Register TensorFlow loss functions with mlconfig
mlconfig.register(tf.keras.losses.SparseCategoricalCrossentropy)

# Create an alias for CrossEntropyLoss to match PyTorch naming
class CrossEntropyLoss:
    """
    Wrapper for TensorFlow's SparseCategoricalCrossentropy to match PyTorch's CrossEntropyLoss interface.
    In PyTorch, CrossEntropyLoss expects logits and integer labels.
    In TensorFlow, SparseCategoricalCrossentropy with from_logits=True provides the same functionality.
    """
    def __init__(self, reduction='mean', **kwargs):
        # Map PyTorch reduction names to TensorFlow
        reduction_map = {
            'mean': tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            'sum': tf.keras.losses.Reduction.SUM,
            'none': tf.keras.losses.Reduction.NONE
        }
        tf_reduction = reduction_map.get(reduction, tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, 
            reduction=tf_reduction,
            **kwargs
        )
        self.reduction = reduction
    
    def __call__(self, logits, labels):
        """
        Compute cross entropy loss.
        
        Args:
            logits: Model outputs (logits), shape [batch_size, num_classes]
            labels: Integer labels, shape [batch_size]
        
        Returns:
            Loss value (scalar if reduction is 'mean' or 'sum', tensor if 'none')
        """
        return self.loss_fn(labels, logits)

# Register the wrapper
mlconfig.register(CrossEntropyLoss)