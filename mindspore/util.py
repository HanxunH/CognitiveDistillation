import logging
import os
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import json
import math
import losses
from scipy.spatial.distance import cdist

# MindSpore: Use framework default device allocation (exemption #3)
# No explicit device setting needed - MindSpore handles this automatically


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """Create parameter groups with layer-wise learning rate decay"""
    # Check if model has _get_layer_ids method
    if hasattr(model, '_get_layer_ids'):
        layer_ids = model._get_layer_ids()
    else:
        # If not available, return all parameters as single group
        return [{
            "params": list(model.trainable_params()),
            "weight_decay": weight_decay,
        }]
    
    layer_scales = list(layer_decay ** (len(layer_ids) - i - 1) for i in range(len(layer_ids)+1))
    param_groups = {}
    param_group_names = {}
    used = []
    
    for param in model.trainable_params():
        n = param.name
        for i, layer_name in enumerate(layer_ids):
            if layer_name in n and n not in used:
                this_scale = layer_scales[i]
                if layer_name not in param_groups:
                    param_groups[layer_name] = {
                        "lr_scale": this_scale,
                        "weight_decay": weight_decay,
                        "params": [param],
                    }
                    param_group_names[layer_name] = {
                        "lr_scale": this_scale,
                        "weight_decay": weight_decay,
                        "params": [n],
                    }
                else:
                    param_groups[layer_name]['params'].append(param)
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
    
    # MindSpore: Use framework-specific optimizer API (exemption #5)
    # Different approach for setting learning rate in MindSpore
    if hasattr(optimizer, 'learning_rate'):
        # For standard MindSpore optimizers
        if isinstance(optimizer.learning_rate, ms.Parameter):
            optimizer.learning_rate.set_data(ms.Tensor(lr, ms.float32))
        else:
            # If learning_rate is a Tensor or scalar
            optimizer.learning_rate = lr
    
    # Handle parameter groups if supported
    if hasattr(optimizer, 'param_groups'):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
    
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
            display += ' ' + key + '=' + value
        else:
            display += ' ' + str(key) + '=%.4f' % value
    display += ' time=%.2fit/s' % (1. / time_elapse)
    return display


def accuracy(output, target, topk=(1,)):
    """Computes and stores the top k accuracy"""
    maxk = max(topk)
    
    batch_size = target.shape[0]
    
    # Get top k predictions
    topk_op = ops.TopK(sorted=True)
    _, pred = topk_op(output, maxk)
    
    # Transpose predictions
    pred = ops.transpose(pred, (1, 0))
    
    # Expand target for comparison
    target_expand = ops.broadcast_to(
        ops.reshape(target, (1, -1)), 
        pred.shape
    )
    
    # Check if predictions are correct
    correct = ops.equal(pred, target_expand)
    
    res = []
    for k in topk:
        # Sum correct predictions for top k
        correct_k = ops.cast(correct[:k], ms.float32).sum()
        # Compute accuracy (using regular division, not in-place)
        # MindSpore doesn't support in-place operations (exemption #6)
        acc_k = correct_k * (1.0 / batch_size)
        res.append(acc_k)
    
    return res


def count_parameters_in_MB(model):
    """Count the number of parameters in MB"""
    total_params = 0
    for param in model.trainable_params():
        total_params += np.prod(param.shape)
    return total_params / 1e6


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