import logging
import os
import numpy as np
import paddle
import json
import math
# import losses
import paddle.nn.functional as F
from scipy.spatial.distance import cdist
from paddle.io import DataLoader

if paddle.device.cuda.device_count() > 0:
    device = paddle.device.set_device('gpu')
else:
    device = paddle.device.set_device('cpu')


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    layer_ids = model._get_layer_ids()
    layer_scales = list(layer_decay ** (len(layer_ids) - i - 1) for i in range(len(layer_ids)+1))
    param_groups = {}
    param_group_names = {}
    used = []
    for n, p in model.named_parameters():
        for i, layer_name in enumerate(layer_ids):
            if layer_name in n and n not in used:
                this_scale = layer_scales[i]
                if layer_name not in param_groups:
                    param_groups[layer_name] = {
                        "lr_scale": this_scale,
                        "weight_decay": weight_decay,
                        "params": [p],
                    }
                    param_group_names[layer_name] = {
                        "lr_scale": this_scale,
                        "weight_decay": weight_decay,
                        "params": [n],
                    }
                else:
                    param_groups[layer_name]['params'].append(p)
                    param_group_names[layer_name]['params'].append(n)
                used.append(n)
    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    return list(param_groups.values())


def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < configs.warmup_epochs:
        lr = configs.lr * epoch / configs.warmup_epochs
    elif 'lr_schdule' in configs and configs.lr_schdule == 'milestone':
        milestones = [int(s*configs.epochs)for s in [0.75, 0.95, 1.0]]
        if epoch < milestones[0]:
            lr = configs.lr
        elif epoch >= milestones[0] and epoch < milestones[1]:
            lr = configs.lr * 0.1
        else:
            lr = configs.lr * 0.01
    else:
        lr = configs.min_lr + (configs.lr - configs.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - configs.warmup_epochs) / (configs.epochs - configs.warmup_epochs)))
    
    # PaddlePaddle uses a different optimizer API
    optimizer.set_lr(lr)
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
    maxk = max(topk)

    batch_size = target.shape[0]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred == target.reshape([1, -1]).expand_as(pred)

    res = []
    for k in topk:
        correct_k = correct[:k].astype('float32').sum()
        res.append(correct_k * (1/batch_size))
    return res


def count_parameters_in_MB(model):
    return sum(np.prod(v.shape) for name, v in model.named_parameters())/1e6


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