import argparse
import mlconfig
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore.train import Model
from mindspore.nn import WithLossCell, TrainOneStepCell
import time
import sys
import os
import numpy as np

# Add mindspore_impl to path
sys.path.append('/root/CognitiveDistillation/mindspore_impl')

import models
import datasets
import losses
import optimizers
import util
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
parser.add_argument('--seed', default=0, type=int)
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)


def save_model():
    # Save model
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    exp.save_state(scheduler, 'scheduler_state_dict')


def epoch_exp_stats():
    # Set epoch level experiment tracking
    # Track Training Loss, this is used by ABL
    stats = {}
    model.set_train(False)
    train_loss_list, correct_list = [], []
    
    for images, labels in no_shuffle_loader:
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, ms.int32)
        
        logits = model(images)
        
        # Compute cross entropy loss
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        loss = loss_fn(logits, labels)
        
        # Get predictions
        predicted = ops.argmax(logits, 1)
        correct = ops.equal(predicted, labels)
        
        # Convert to numpy and store
        train_loss_list += loss.asnumpy().tolist()
        correct_list += correct.asnumpy().tolist()
    
    stats['samplewise_train_loss'] = train_loss_list
    stats['samplewise_correct'] = correct_list
    return stats


def evaluate(target_model, epoch, loader):
    target_model.set_train(False)
    # Training Evaluations
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()
    loss_list, correct_list = [], []
    
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, ms.int32)
        
        batch_size = images.shape[0]
        logits = target_model(images)
        
        # Compute cross entropy loss
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        loss = loss_fn(logits, labels)
        loss_list += loss.asnumpy().tolist()
        loss_mean = loss.mean().asnumpy().item()
        
        # Calculate acc
        acc = util.accuracy(logits, labels, topk=(1,))[0].asnumpy().item()
        
        # Update Meters
        loss_meters.update(loss_mean, batch_size)
        acc_meters.update(acc, batch_size)
        
        # Get predictions
        predicted = ops.argmax(logits, 1)
        correct = ops.equal(predicted, labels)
        correct_list += correct.asnumpy().tolist()
    
    return loss_meters.avg, acc_meters.avg, loss_list, correct_list


def bd_evaluate(target_model, epoch, loader, data):
    # Handle TorchDatasetWrapper - access the underlying dataset
    poison_test_set = data.poison_test_set
    if hasattr(poison_test_set, 'poison_idx'):
        bd_idx = poison_test_set.poison_idx
    elif hasattr(poison_test_set, 'dataset') and hasattr(poison_test_set.dataset, 'poison_idx'):
        bd_idx = poison_test_set.dataset.poison_idx
    else:
        # If no poison indices found, skip backdoor evaluation
        return 0.0
    
    target_model.set_train(False)
    pred_list, label_list = [], []
    
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, ms.int32)
        
        logits = target_model(images)
        predicted = ops.argmax(logits, 1)
        
        pred_list.append(predicted.asnumpy())
        label_list.append(labels.asnumpy())
    
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    asr = (pred_list[bd_idx] == label_list[bd_idx]).sum().item() / len(bd_idx)
    return asr


def train(epoch):
    global global_step, best_acc
    # Track exp stats
    if isinstance(criterion, nn.SoftmaxCrossEntropyWithLogits):
        epoch_stats = epoch_exp_stats()
    else:
        epoch_stats = {}
    
    # Set Meters
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()
    
    # Training
    model.set_train(True)
    
    for i, data in enumerate(train_loader):
        start = time.time()
        # Prepare batch data
        images, labels = data
        
        # Convert to MindSpore tensors if needed
        if not isinstance(images, ms.Tensor):
            images = ms.Tensor(images, ms.float32)
        if not isinstance(labels, ms.Tensor):
            labels = ms.Tensor(labels, ms.int32)
        
        batch_size = images.shape[0]
        
        # Forward pass and compute loss
        if isinstance(criterion, nn.SoftmaxCrossEntropyWithLogits):
            # Define forward function for standard cross entropy loss
            def forward_fn():
                logits = model(images)
                loss = criterion(logits, labels)
                return loss, logits
            
            # Compute gradients using value_and_grad
            grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, logits), grads = grad_fn()
        else:
            # For custom criterion
            def forward_fn():
                logits, loss = criterion(model, images, labels)
                return loss, logits
            
            grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, logits), grads = grad_fn()
        
        # Update parameters
        optimizer(grads)
        
        # Calculate acc
        loss_val = loss.asnumpy().item()
        acc = util.accuracy(logits, labels, topk=(1,))[0].asnumpy().item()
        
        # Update Meters
        loss_meters.update(loss_val, batch_size)
        acc_meters.update(acc, batch_size)
        
        # Log results
        end = time.time()
        time_used = end - start
        if global_step % exp.config.log_frequency == 0:
            # Get current learning rate
            if hasattr(optimizer, 'learning_rate'):
                if isinstance(optimizer.learning_rate, ms.Parameter):
                    current_lr = optimizer.learning_rate.data.asnumpy().item()
                else:
                    current_lr = optimizer.learning_rate
            else:
                current_lr = exp.config.lr  # Use default from config
            
            payload = {
                "acc_avg": acc_meters.avg,
                "loss_avg": loss_meters.avg,
                "lr": current_lr
            }
            display = util.log_display(epoch=epoch,
                                      global_step=global_step,
                                      time_elapse=time_used,
                                      **payload)
            logger.info(display)
        # Update Global Step
        global_step += 1
    
    epoch_stats['global_step'] = global_step
    
    return epoch_stats


def main():
    # Set Global Vars
    global criterion, model, optimizer, scheduler, gcam
    global train_loader, test_loader, data
    global poison_test_loader, no_shuffle_loader
    global logger, start_epoch, global_step, best_acc
    # Set up Experiments
    logger = exp.logger
    config = exp.config
    # Prepare Data
    data = config.dataset(exp)
    loader = data.get_loader(train_shuffle=True)
    train_loader, test_loader, poison_test_loader = loader
    no_shuffle_loader, _, _ = data.get_loader(train_shuffle=False)
    
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
    print(model)
    
    # Prepare Objective Loss function
    criterion = config.criterion()
    start_epoch = 0
    global_step = 0
    best_acc = 0
    
    # Resume: Load models
    exp_stats = exp.load_epoch_stats()
    if exp_stats is not None:
        # start_epoch = exp_stats['epoch'] + 1
        global_step = exp_stats.get('global_step', 0) + 1
    else:
        exp_stats = {'global_step': 0}
    
    model = exp.load_state(model, 'model_state_dict')
    optimizer = exp.load_state(optimizer, 'optimizer_state_dict')
    
    if args.data_parallel:
        # Skip DataParallel support per requirements
        logger.info("DataParallel not supported in MindSpore implementation - skipping")
    
    # Epoch Eval Function
    logger.info("="*20 + "Eval Epoch %d" % (exp.config.epochs) + "="*20)
    model.set_train(False)
    eval_loss, eval_acc, ll, cl = evaluate(model, exp.config.epochs, test_loader)
    if eval_acc > best_acc:
        best_acc = eval_acc
    payload = 'Eval Loss: %.4f Eval Acc: %.4f Best Acc: %.4f' % \
        (eval_loss, eval_acc, best_acc)
    logger.info('\033[33m'+payload+'\033[0m')
    exp_stats['eval_acc'] = eval_acc
    exp_stats['best_acc'] = best_acc
    exp_stats['epoch'] = exp.config.epochs
    exp_stats['samplewise_eval_loss'] = ll
    exp_stats['samplewise_eval_correct'] = cl
    
    # Epoch Backdoor Eval
    if poison_test_loader is not None:
        asr = bd_evaluate(model, exp.config.epochs, poison_test_loader, data)
        payload = 'Model Backdoor Attack success rate %.4f' % (asr)
        logger.info('\033[33m'+payload+'\033[0m')
        exp_stats['eval_asr'] = asr
    
    return


if __name__ == '__main__':
    global exp
    args = parser.parse_args()
    ms.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                  exp_path=args.exp_path,
                                  config_file_path=config_filename,
                                  eval_mode=True)
    logger = experiment.logger
    logger.info("MindSpore Version: %s" % (ms.__version__))
    logger.info("Python Version: %s" % (sys.version))
    
    # Log device info
    device_target = ms.get_context("device_target")
    logger.info("Device Target: %s" % device_target)
    
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    for key in experiment.config:
        logger.info("%s: %s" % (key, experiment.config[key]))
    start = time.time()
    exp = experiment
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)