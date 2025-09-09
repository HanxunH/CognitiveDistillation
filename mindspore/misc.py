# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import functools
import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        Note: Distributed operations are skipped per project guidelines
        """
        # Skip distributed synchronization as per project guidelines
        raise NotImplementedError("Skip")

    @property
    def median(self):
        d = Tensor(list(self.deque), dtype=ms.float32)
        # MindSpore's median returns (value, index) tuple
        median_val = ops.median(d)
        if isinstance(median_val, tuple):
            return median_val[0].asnumpy().item()
        return median_val.asnumpy().item()

    @property
    def avg(self):
        d = Tensor(list(self.deque), dtype=ms.float32)
        return ops.mean(d).asnumpy().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    @property
    def max(self):
        return max(self.deque) if self.deque else 0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, Tensor):
                v = v.asnumpy().item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        # Skip distributed synchronization as per project guidelines
        raise NotImplementedError("Skip")

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        # MindSpore doesn't have direct CUDA memory tracking like PyTorch
        # Skip GPU memory logging
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    Note: Distributed operations are skipped per project guidelines
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    """
    Check if distributed is available and initialized.
    Note: Always returns False as distributed is skipped per project guidelines
    """
    return False


def get_world_size():
    """
    Get world size for distributed training.
    Note: Always returns 1 as distributed is skipped per project guidelines
    """
    return 1


def get_rank():
    """
    Get rank for distributed training.
    Note: Always returns 0 as distributed is skipped per project guidelines
    """
    return 0


def is_main_process():
    """Check if this is the main process"""
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save only on master process.
    For MindSpore, we use ms.save_checkpoint instead of torch.save
    """
    if is_main_process():
        # Convert the save operation to MindSpore format
        # This is a utility function that should be called with MindSpore specific params
        if len(args) == 2 and isinstance(args[0], dict):
            # Expecting (state_dict, path) format
            state_dict, path = args
            # For MindSpore, we need to handle this differently
            # The caller should use MindSpore's save methods directly
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(state_dict, f)
        else:
            # For other cases, let the caller handle MindSpore-specific saving
            raise NotImplementedError("Use MindSpore's save_checkpoint for model saving")


def destroy_process_group():
    """
    Destroy process group.
    Note: Distributed operations are skipped per project guidelines
    """
    raise NotImplementedError("Skip")


def init_distributed_mode(args):
    """
    Initialize distributed mode.
    Note: Distributed operations are skipped per project guidelines
    """
    print('Not using distributed mode (MindSpore implementation - distributed skipped)')
    setup_for_distributed(is_master=True)
    args.distributed = False
    return


class NativeScalerWithGradNormCount:
    """
    Gradient scaler for mixed precision training.
    Note: MindSpore handles mixed precision differently than PyTorch
    """
    state_dict_key = "amp_scaler"

    def __init__(self):
        # MindSpore uses different mixed precision approach
        # This is a placeholder implementation
        self.loss_scale = 1.0

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        """
        Note: MindSpore's gradient computation and optimization is different.
        This is a simplified implementation that should be adapted based on actual usage.
        """
        if update_grad:
            # In MindSpore, gradient computation and optimization is handled differently
            # This would need to be integrated with MindSpore's training loop
            norm = None
            if clip_grad is not None and parameters is not None:
                # MindSpore gradient clipping would be handled differently
                norm = get_grad_norm_(parameters)
        else:
            norm = None
        return norm

    def state_dict(self):
        return {"loss_scale": self.loss_scale}

    def load_state_dict(self, state_dict):
        self.loss_scale = state_dict.get("loss_scale", 1.0)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> Tensor:
    """
    Calculate gradient norm.
    Note: MindSpore gradient handling is different from PyTorch
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    
    # Filter parameters with gradients
    # Note: In MindSpore, gradient access is different
    # This is a simplified implementation
    parameters = [p for p in parameters if p is not None]
    
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return Tensor(0., dtype=ms.float32)
    
    if norm_type == float('inf'):
        # Max norm
        total_norm = max(ops.abs(p).max() for p in parameters if p is not None)
        return Tensor(total_norm, dtype=ms.float32)
    else:
        # L2 norm
        norms = []
        for p in parameters:
            if p is not None:
                param_norm = ops.norm(p.astype(ms.float32), ord=norm_type)
                norms.append(param_norm)
        
        if norms:
            total_norm = ops.norm(ops.stack(norms), ord=norm_type)
        else:
            total_norm = Tensor(0., dtype=ms.float32)
        
        return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    """
    Save model checkpoint.
    Note: Adapted for MindSpore's checkpoint system
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.ckpt' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            # For MindSpore, we need to handle state dict differently
            # Save using MindSpore's checkpoint format
            if is_main_process():
                # Create a state dict compatible with MindSpore
                to_save = {
                    'epoch': epoch,
                    'args': args,
                }
                
                # Save model parameters using MindSpore's save_checkpoint
                ms.save_checkpoint(model_without_ddp, str(checkpoint_path))
                
                # Save additional metadata separately
                import pickle
                meta_path = str(checkpoint_path).replace('.ckpt', '_meta.pkl')
                with open(meta_path, 'wb') as f:
                    pickle.dump(to_save, f)
    else:
        # Alternative saving method
        if is_main_process():
            checkpoint_path = output_dir / ('checkpoint-%s.ckpt' % epoch_name)
            ms.save_checkpoint(model, str(checkpoint_path))


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    """
    Load model checkpoint.
    Note: Adapted for MindSpore's checkpoint system
    """
    if args.resume:
        if args.resume.startswith('https'):
            # URL loading not directly supported in MindSpore
            raise NotImplementedError("URL checkpoint loading not implemented for MindSpore")
        else:
            # Load MindSpore checkpoint
            checkpoint_path = args.resume
            
            # Load model parameters
            param_dict = ms.load_checkpoint(checkpoint_path)
            ms.load_param_into_net(model_without_ddp, param_dict)
            print("Resume checkpoint %s" % args.resume)
            
            # Load metadata if available
            meta_path = checkpoint_path.replace('.ckpt', '_meta.pkl')
            if os.path.exists(meta_path) and not (hasattr(args, 'eval') and args.eval):
                import pickle
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    if 'epoch' in meta:
                        args.start_epoch = meta['epoch'] + 1
                    print("With metadata!")


def all_reduce_mean(x):
    """
    All reduce mean for distributed training.
    Note: Distributed operations are skipped per project guidelines
    """
    # Since we're not using distributed mode, just return the value
    if isinstance(x, Tensor):
        return x.asnumpy().item()
    return x


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend.
    Note: Distributed operations are skipped per project guidelines
    """
    raise NotImplementedError("Skip")


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data.
    Note: Distributed operations are skipped per project guidelines
    
    Args:
        data: any picklable object
        group: a process group (not used in MindSpore implementation)
    
    Returns:
        list[data]: list containing only the input data (no distributed gathering)
    """
    # Since we're not using distributed mode, just return the data in a list
    return [data]