import logging
import os
import numpy as np
import tensorflow as tf
import json
import math
import tensorflow_impl.losses as losses
from scipy.spatial.distance import cdist

# Device configuration following TensorFlow convention
# TensorFlow automatically selects the best available device


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Note: TensorFlow doesn't have the same parameter group concept as PyTorch.
    This function returns configuration that can be used with custom training loops.
    """
    # Check if model has _get_layer_ids method
    if hasattr(model, '_get_layer_ids'):
        layer_ids = model._get_layer_ids()
    else:
        # Return a simple configuration if the method doesn't exist
        return [{
            "lr_scale": 1.0,
            "weight_decay": weight_decay,
            "params": model.trainable_variables
        }]
    
    layer_scales = list(layer_decay ** (len(layer_ids) - i - 1) for i in range(len(layer_ids)+1))
    param_groups = {}
    param_group_names = {}
    used = []
    
    # TensorFlow uses model.trainable_variables instead of named_parameters
    for var in model.trainable_variables:
        n = var.name
        for i, layer_name in enumerate(layer_ids):
            if layer_name in n and n not in used:
                this_scale = layer_scales[i]
                if layer_name not in param_groups:
                    param_groups[layer_name] = {
                        "lr_scale": this_scale,
                        "weight_decay": weight_decay,
                        "params": [var],
                    }
                    param_group_names[layer_name] = {
                        "lr_scale": this_scale,
                        "weight_decay": weight_decay,
                        "params": [n],
                    }
                else:
                    param_groups[layer_name]['params'].append(var)
                    param_group_names[layer_name]['params'].append(n)
                used.append(n)
    
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())


def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < configs.warmup_epochs:
        lr = configs.lr * epoch / configs.warmup_epochs
    elif 'lr_schdule' in configs and configs.lr_schdule == 'milestone':
        milestones = [int(s*configs.epochs) for s in [0.75, 0.95, 1.0]]
        if epoch < milestones[0]:
            lr = configs.lr
        elif epoch >= milestones[0] and epoch < milestones[1]:
            lr = configs.lr * 0.1
        else:
            lr = configs.lr * 0.01
    else:
        lr = configs.min_lr + (configs.lr - configs.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - configs.warmup_epochs) / (configs.epochs - configs.warmup_epochs)))
    
    # TensorFlow optimizers use learning_rate property
    if hasattr(optimizer, 'learning_rate'):
        if isinstance(optimizer.learning_rate, tf.Variable):
            optimizer.learning_rate.assign(lr)
        else:
            # For optimizers that don't support variable learning rates,
            # we need to handle this differently in the training loop
            optimizer.learning_rate = lr
    elif hasattr(optimizer, 'lr'):
        if isinstance(optimizer.lr, tf.Variable):
            optimizer.lr.assign(lr)
        else:
            optimizer.lr = lr
    
    return lr


def setup_logger(name, log_file, ddp=False, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    if not ddp:
        logger.addHandler(console_handler)
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              ' global_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = ' ' + key + '=' + value
        else:
            display += ' ' + str(key) + '=%.4f' % value
    display += ' time=%.2fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    
    batch_size = tf.shape(target)[0]
    
    # Get top k predictions
    _, pred = tf.nn.top_k(output, k=maxk, sorted=True)
    pred = tf.transpose(pred)
    
    # Ensure target has the same dtype as predictions
    if target.dtype != pred.dtype:
        target = tf.cast(target, pred.dtype)
    
    # Expand target for comparison
    target_reshaped = tf.reshape(target, [1, -1])
    target_expanded = tf.tile(target_reshaped, [maxk, 1])
    
    # Check if predictions match target
    correct = tf.equal(pred, target_expanded)
    
    res = []
    for k in topk:
        # Sum correct predictions in top k
        correct_k = tf.reduce_sum(tf.cast(correct[:k], tf.float32))
        # Compute accuracy
        acc_k = correct_k / tf.cast(batch_size, tf.float32)
        res.append(acc_k)
    
    return res


def count_parameters_in_MB(model):
    """Count the number of parameters in MB"""
    if hasattr(model, 'trainable_variables'):
        # TensorFlow/Keras model
        return sum(np.prod(v.shape.as_list()) for v in model.trainable_variables) / 1e6
    else:
        # Fallback for other model types
        return 0.0


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)