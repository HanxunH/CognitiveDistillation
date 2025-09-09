import argparse
import mlconfig
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
import time
import sys
import os
import numpy as np
from collections import OrderedDict

# Add mindspore_impl to path
sys.path.append('/root/CognitiveDistillation/mindspore_impl')

import models
import datasets
import losses
import util
import misc
from exp_mgmt import ExperimentManager

# Set MindSpore context
ms.set_context(mode=ms.PYNATIVE_MODE)
if ms.get_context("device_target") == "GPU":
    ms.set_context(device_target="GPU")
elif ms.get_context("device_target") == "CPU":
    ms.set_context(device_target="CPU")
else:
    # Default to CPU if not specified
    ms.set_context(device_target="CPU")

parser = argparse.ArgumentParser(description='CognitiveDistillation')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--ddp', action='store_true', default=False)
# distributed training parameters - kept for compatibility but will raise NotImplementedError
parser.add_argument('--dist_eval', action='store_true', default=False)
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist_url', default='env://',
                    help='url used to set up distributed training')


def save_model():
    # Save model
    exp.save_state(model_without_ddp, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')


def epoch_exp_stats():
    # Set epoch level experiment tracking
    # Track Training Loss, this is used by ABL
    stats = {}
    model.set_train(False)
    train_loss_list, correct_list = [], []
    
    for data in no_shuffle_loader:
        images, labels = data
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, dtype=ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, dtype=ms.int32)
            
        logits = model(images)
        
        # Calculate cross entropy loss per sample
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        loss = loss_fn(logits, labels)
        
        # Get predictions
        predicted = ops.argmax(logits, axis=1)
        correct = ops.equal(predicted, labels)
        
        # Convert to numpy and extend lists
        train_loss_list.extend(loss.asnumpy().tolist())
        correct_list.extend(correct.asnumpy().tolist())

    stats['samplewise_train_loss'] = train_loss_list
    stats['samplewise_correct'] = correct_list
    return stats


def evaluate(model, loader):
    model.set_train(False)
    metric_logger = misc.MetricLogger(delimiter=" ")
    
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, dtype=ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, dtype=ms.int32)
            
        logits = model(images)
        
        # Calculate cross entropy loss
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        loss = loss_fn(logits, labels)

        # Calculate acc
        acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        # Update Meters
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.asnumpy().item())
        metric_logger.update(acc=acc.asnumpy().item(), n=batch_size)
        metric_logger.update(acc5=acc5.asnumpy().item(), n=batch_size)

    # Skip distributed synchronization
    if hasattr(metric_logger, 'synchronize_between_processes'):
        try:
            metric_logger.synchronize_between_processes()
        except NotImplementedError:
            pass  # Expected for distributed operations
    
    payload = (metric_logger.meters['loss'].avg,
               metric_logger.meters['acc'].avg,
               metric_logger.meters['acc5'].avg)
    return payload


def bd_evaluate(model, loader, data):
    bd_idx = data.poison_test_set.poison_idx
    model.set_train(False)
    pred_list, label_list = [], []
    
    for i, batch_data in enumerate(loader):
        # Prepare batch data
        images, labels = batch_data
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, dtype=ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, dtype=ms.int32)
            
        logits = model(images)
        predicted = ops.argmax(logits, axis=1)
        
        pred_list.append(predicted.asnumpy())
        label_list.append(labels.asnumpy())
    
    pred_list = np.concatenate(pred_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    asr = (pred_list[bd_idx] == label_list[bd_idx]).sum().item() / len(bd_idx)
    return asr


def train(epoch):
    global global_step
    # Track exp stats
    if isinstance(criterion, nn.SoftmaxCrossEntropyWithLogits):
        epoch_stats = epoch_exp_stats()
    else:
        epoch_stats = {}

    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # Skip distributed sampler epoch setting
    if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
        raise NotImplementedError("Skip")  # Distributed training not supported
    
    # Training
    model.set_train(True)
    
    for i, data in enumerate(train_loader):
        start = time.time()
        # Adjust LR
        if 'accum_iter' in exp.config:
            if (global_step+1) % exp.config.accum_iter == 0:
                util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        else:
            util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        
        images, labels = data
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, dtype=ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, dtype=ms.int32)
        
        model.set_train(True)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Calculate gradients and update
        loss_scalar = loss.asnumpy().item()
        grads = grad_fn(logits, labels)
        optimizer(grads)
        
        # Calculate acc
        acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        
        # Update Meters
        batch_size = images.shape[0]
        metric_logger.update(loss=loss_scalar)
        metric_logger.update(acc=acc.asnumpy().item(), n=batch_size)
        metric_logger.update(acc5=acc5.asnumpy().item(), n=batch_size)
        
        # Log results
        end = time.time()
        time_used = end - start
        lr = optimizer.learning_rate.asnumpy().item() if hasattr(optimizer, 'learning_rate') else exp.config.optimizer.lr
        metric_logger.update(lr=lr)
        
        if global_step % exp.config.log_frequency == 0:
            # Skip distributed all_reduce operations
            if misc.get_rank() == 0:  # Will return 0 for non-distributed
                # Skip distributed synchronization
                payload = {
                    "acc": acc.asnumpy().item(),
                    "acc_avg": metric_logger.meters['acc'].avg,
                    "acc5_avg": metric_logger.meters['acc5'].avg,
                    "loss": loss_scalar,
                    "loss_avg": metric_logger.meters['loss'].avg,
                    "lr": lr,
                }
                display = util.log_display(epoch=epoch,
                                         global_step=global_step,
                                         time_elapse=time_used,
                                         **payload)
                logger.info(display)
        # Update Global Step
        global_step += 1

    epoch_stats['global_step'] = global_step
    epoch_stats['train_acc'] = metric_logger.meters['acc'].avg
    epoch_stats['train_acc5'] = metric_logger.meters['acc5'].avg
    epoch_stats['train_loss'] = metric_logger.meters['loss'].avg

    return epoch_stats


def main():
    # Set Global Vars
    global criterion, model, optimizer, model_without_ddp
    global train_loader, test_loader, data
    global poison_test_loader, no_shuffle_loader
    global logger, start_epoch, global_step, best_acc
    global grad_fn

    # Set up Experiments
    logger = exp.logger
    config = exp.config
    # Prepare Data
    data = config.dataset(exp)
    
    if args.ddp:
        # Skip all distributed training setup
        raise NotImplementedError("Skip")  # Distributed training not supported
    else:
        # Non-distributed data loading
        loader = data.get_loader(train_shuffle=True)
        train_loader, test_loader, poison_test_loader = loader
        no_shuffle_loader, _, _ = data.get_loader(train_shuffle=False)

    # Save poison idx (only on rank 0 or in non-distributed mode)
    if misc.get_rank() == 0:
        if hasattr(data.train_set, 'noisy_idx'):
            noisy_idx = data.train_set.noisy_idx
            filename = os.path.join(exp.exp_path, 'train_noisy_idx.npy')
            with open(filename, 'wb') as f:
                np.save(f, noisy_idx)
        elif hasattr(data.train_set, 'poison_idx'):
            poison_idx = data.train_set.poison_idx
            filename = os.path.join(exp.exp_path, 'train_poison_idx.npy')
            with open(filename, 'wb') as f:
                np.save(f, poison_idx)
        if hasattr(data.poison_test_set, 'noisy_idx'):
            noisy_idx = data.poison_test_set.noisy_idx
            filename = os.path.join(exp.exp_path, 'bd_test_noisy_idx.npy')
            with open(filename, 'wb') as f:
                np.save(f, noisy_idx)
        elif hasattr(data.poison_test_set, 'poison_idx'):
            poison_idx = data.poison_test_set.poison_idx
            filename = os.path.join(exp.exp_path, 'bd_test_poison_idx.npy')
            with open(filename, 'wb') as f:
                np.save(f, poison_idx)

    # Prepare Model
    model = config.model()
    optimizer = config.optimizer(model.trainable_params())

    if 'pretrain_weight' in exp.config:
        # Skip pretrained weight loading with special layer decay handling
        raise NotImplementedError("Skip")  # Pretrained weight loading with layer decay not implemented

    if misc.get_rank() == 0:
        print(model)

    # Prepare Objective Loss function
    criterion = config.criterion()
    
    # Create gradient function for training
    def forward_fn(inputs, labels):
        logits = model(inputs)
        loss = criterion(logits, labels)
        return loss
    
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)
    
    # Define a custom training step
    def train_step(inputs, labels):
        loss, grads = grad_fn(inputs, labels)
        optimizer(grads)
        return loss
    
    start_epoch = 0
    global_step = 0
    best_acc = 0

    # Resume: Load models
    if args.load_model:
        exp_stats = exp.load_epoch_stats()
        start_epoch = exp_stats['epoch'] + 1
        global_step = exp_stats['global_step'] + 1
        model = exp.load_state(model, 'model_state_dict')
        optimizer = exp.load_state(optimizer, 'optimizer_state_dict')

    if args.ddp:
        # Skip DDP wrapper
        raise NotImplementedError("Skip")  # Distributed training not supported
    
    model_without_ddp = model

    # Train Loops
    for epoch in range(start_epoch, exp.config.epochs):
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.set_train(True)
        stats = train(epoch)

        # Epoch Eval Function
        if misc.get_rank() == 0:
            logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        model.set_train(False)
        eval_loss, eval_acc, eval_acc5 = evaluate(model, test_loader)
        if eval_acc > best_acc:
            best_acc = eval_acc
        stats['eval_loss'] = eval_loss
        stats['eval_acc'] = eval_acc
        stats['eval_acc5'] = eval_acc5
        stats['best_acc'] = best_acc
        if misc.get_rank() == 0:
            payload = 'Eval Loss: %.4f Eval Acc: %.4f Best Acc: %.4f' % \
                      (stats['eval_loss'], stats['eval_acc'], best_acc)
            logger.info('\033[33m'+payload+'\033[0m')
        # Backdoor Evaluation
        asr = bd_evaluate(model, poison_test_loader, data)
        if misc.get_rank() == 0:
            stats['eval_asr'] = asr
            payload = 'Model Backdoor Attack success rate %.4f' % (asr)
            logger.info('\033[33m'+payload+'\033[0m')

        # Save Model
        if misc.get_rank() == 0:
            exp.save_epoch_stats(epoch=epoch, exp_stats=stats)
            save_model()
    return


if __name__ == '__main__':
    global exp
    args = parser.parse_args()
    
    if args.ddp:
        # Skip distributed initialization
        raise NotImplementedError("Skip")  # Distributed training not supported
    else:
        ms.set_seed(args.seed)
        np.random.seed(args.seed)

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("MindSpore Version: %s" % (ms.__version__))
        logger.info("Python Version: %s" % (sys.version))
        for arg in vars(args):
            logger.info("%s: %s" % (arg, getattr(args, arg)))
        for key in experiment.config:
            logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    if misc.get_rank() == 0:
        payload = "Running Cost %.2f Days" % cost
        logger.info(payload)