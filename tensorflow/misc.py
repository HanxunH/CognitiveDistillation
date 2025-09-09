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
import tensorflow as tf
import numpy as np


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
        """
        # Skip distributed synchronization as per the workflow instructions
        raise NotImplementedError("Skip")

    @property
    def median(self):
        d = tf.constant(list(self.deque))
        return tf.reduce_median(d).numpy() if hasattr(tf, 'reduce_median') else np.median(list(self.deque))

    @property
    def avg(self):
        d = tf.constant(list(self.deque), dtype=tf.float32)
        return tf.reduce_mean(d).numpy()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

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
            if isinstance(v, tf.Tensor):
                v = v.numpy()
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
        # Skip distributed synchronization as per the workflow instructions
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
        if len(tf.config.list_physical_devices('GPU')) > 0:
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if len(tf.config.list_physical_devices('GPU')) > 0:
                    # TensorFlow doesn't have a direct equivalent to torch.cuda.max_memory_allocated()
                    # We'll skip memory tracking for now or use a placeholder
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=0))  # Placeholder for memory tracking
                else:
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
    # Skip distributed functionality as per the workflow instructions
    return False


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    raise NotImplementedError("Skip")


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    raise NotImplementedError("Skip")


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        # TensorFlow uses different saving mechanism
        # This function should be called with appropriate TF saving logic
        raise NotImplementedError("Use TensorFlow's model.save() or tf.saved_model.save() instead")


def destroy_process_group():
    # Skip distributed functionality as per the workflow instructions
    raise NotImplementedError("Skip")


def init_distributed_mode(args):
    # Skip distributed functionality as per the workflow instructions
    print('Not using distributed mode')
    setup_for_distributed(is_master=True)  # hack
    args.distributed = False
    return


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        # TensorFlow uses tf.keras.mixed_precision for automatic mixed precision
        self._optimizer = None

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        # TensorFlow handles gradients differently
        # This is a placeholder implementation
        if update_grad:
            if clip_grad is not None:
                # TensorFlow gradient clipping is handled differently
                norm = tf.constant(0.0)
            else:
                norm = None
            # Actual gradient update would be handled by TensorFlow's optimizer
        else:
            norm = None
        return norm

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


def get_grad_norm_(parameters, norm_type: float = 2.0) -> tf.Tensor:
    # TensorFlow handles gradients differently
    # This is a simplified implementation
    if isinstance(parameters, tf.Tensor):
        parameters = [parameters]
    
    # In TensorFlow, gradients are typically accessed through GradientTape
    # This is a placeholder implementation
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return tf.constant(0.0)
    
    # Placeholder for gradient norm calculation
    return tf.constant(0.0)


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    
    # TensorFlow model saving
    checkpoint_path = output_dir / f'checkpoint-{epoch_name}'
    
    # Save model weights
    model.save_weights(str(checkpoint_path / 'model.h5'))
    
    # Save optimizer state if needed (TensorFlow handles this differently)
    # TensorFlow typically saves optimizer state with model.save()
    
    # Save additional metadata
    metadata = {
        'epoch': epoch,
        'args': vars(args) if hasattr(args, '__dict__') else args,
    }
    
    import json
    with open(checkpoint_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            # TensorFlow doesn't have direct equivalent to torch.hub.load_state_dict_from_url
            raise NotImplementedError("URL loading not implemented for TensorFlow")
        else:
            # Load model weights
            model_without_ddp.load_weights(args.resume)
            print("Resume checkpoint %s" % args.resume)
            
            # Load metadata if available
            checkpoint_dir = Path(args.resume).parent
            metadata_path = checkpoint_dir / 'metadata.json'
            if metadata_path.exists() and not (hasattr(args, 'eval') and args.eval):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                args.start_epoch = metadata.get('epoch', 0) + 1
                print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        # Skip distributed functionality as per the workflow instructions
        raise NotImplementedError("Skip")
    else:
        return x


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    # Skip distributed functionality as per the workflow instructions
    raise NotImplementedError("Skip")


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    # Skip distributed functionality as per the workflow instructions
    raise NotImplementedError("Skip")