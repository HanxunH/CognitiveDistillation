import argparse
import mlconfig
import torch
import time
import models
import datasets
import losses
import torch.nn.functional as F
import util
import os
import sys
import numpy as np
import misc
from exp_mgmt import ExperimentManager
from collections import OrderedDict
from timm.models.layers import trunc_normal_
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='CognitiveDistillation')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--ddp', action='store_true', default=False)
# distributed training parameters
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


@torch.no_grad()
def epoch_exp_stats():
    # Set epoch level experiment tracking
    # Track Training Loss, this is used by ABL
    stats = {}
    model.eval()
    train_loss_list, correct_list = [], []
    for images, labels in no_shuffle_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels, reduction='none')
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels)
        train_loss_list += loss.detach().cpu().tolist()
        correct_list += correct.detach().cpu().tolist()
    stats['samplewise_train_loss'] = train_loss_list
    stats['samplewise_correct'] = correct_list
    return stats


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter=" ")
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        # Calculate acc
        acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        # Update Meters
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc.item(), n=batch_size)
        metric_logger.update(acc5=acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    payload = (metric_logger.meters['loss'].avg,
               metric_logger.meters['acc'].avg,
               metric_logger.meters['acc5'].avg)
    return payload


@torch.no_grad()
def bd_evaluate(model, loader, data):
    bd_idx = data.poison_test_set.poison_idx
    model.eval()
    pred_list, label_list = [], []
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        _, predicted = torch.max(logits.data, 1)
        pred_list.append(predicted.detach().cpu())
        label_list.append(labels.detach().cpu())
    pred_list, label_list = torch.cat(pred_list, dim=0), torch.cat(label_list, dim=0)
    asr = (pred_list[bd_idx] == label_list[bd_idx]).sum().item() / len(bd_idx)
    return asr


def train(epoch):
    global global_step
    # Track exp stats
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        epoch_stats = epoch_exp_stats()
    else:
        epoch_stats = {}

    # Set Meters
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    train_loader.sampler.set_epoch(epoch)
    # Training
    model.train()
    model.zero_grad()
    optimizer.zero_grad()
    for i, data in enumerate(train_loader):
        start = time.time()
        # Adjust LR
        if 'accum_iter' in exp.config:
            if (global_step+1) % exp.config.accum_iter == 0:
                util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        else:
            util.adjust_learning_rate(optimizer, i / len(train_loader) + epoch, exp.config)
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        # Optimize
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        # Calculate acc
        acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
        loss = loss.item()
        # Update Meters
        batch_size = images.shape[0]
        metric_logger.update(loss=loss)
        metric_logger.update(acc=acc.item(), n=batch_size)
        metric_logger.update(acc5=acc5.item(), n=batch_size)
        # Log results
        end = time.time()
        time_used = end - start
        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)
        if global_step % exp.config.log_frequency == 0:
            loss = misc.all_reduce_mean(loss)
            acc = misc.all_reduce_mean(acc)
            acc5 = misc.all_reduce_mean(acc5)
            lr = misc.all_reduce_mean(lr)
            metric_logger.synchronize_between_processes()
            payload = {
                "acc": acc,
                "acc_avg": metric_logger.meters['acc'].avg,
                "acc5_avg": metric_logger.meters['acc5'].avg,
                "loss": loss,
                "loss_avg": metric_logger.meters['loss'].avg,
                "lr": lr,
            }
            if misc.get_rank() == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=global_step,
                                           time_elapse=time_used,
                                           **payload)
                logger.info(display)
        # Update Global Step
        global_step += 1

    epoch_stats['global_step'] = global_step
    epoch_stats['train_acc'] = metric_logger.meters['acc'].avg,
    epoch_stats['train_acc5'] = metric_logger.meters['acc5'].avg,
    epoch_stats['train_loss'] = metric_logger.meters['loss'].avg,

    return epoch_stats


def main():
    # Set Global Vars
    global criterion, model, optimizer, model_without_ddp
    global train_loader, test_loader, data
    global poison_test_loader, no_shuffle_loader
    global logger, start_epoch, global_step, best_acc

    # Set up Experiments
    logger = exp.logger
    config = exp.config
    # Prepare Data
    data = config.dataset(exp)
    if args.ddp:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            data.train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(data.test_set) % num_tasks != 0 or len(data.poison_test_set) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(data.test_set, num_replicas=num_tasks,
                                                              rank=global_rank, shuffle=True)
            sampler_bd_val = torch.utils.data.DistributedSampler(data.poison_test_set, num_replicas=num_tasks,
                                                                 rank=global_rank, shuffle=True)
            # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(data.test_set)
            sampler_bd_val = torch.utils.data.SequentialSampler(data.poison_test_set)
    else:
        sampler_train = torch.utils.data.RandomSampler(data.train_set)
        sampler_val = torch.utils.data.SequentialSampler(data.test_set)

    loader = data.get_loader(train_shuffle=True, train_sampler=sampler_train, test_sampler=sampler_val)
    train_loader, test_loader, poison_test_loader = loader
    no_shuffle_loader, _, _ = data.get_loader(train_shuffle=False,  train_sampler=sampler_train,
                                              test_sampler=sampler_val, sampler_bd_val=sampler_bd_val)

    # Save poison idx
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
    model = config.model().to(device)
    optimizer = config.optimizer(model.parameters())

    if 'pretrain_weight' in exp.config:
        state_dict = torch.load(exp.config.pretrain_weight)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('encoder.') or k.startswith('module.encoder.'):
                remove_l = 8 if k.startswith('encoder.') else len('module.encoder.')
                name = k[remove_l:]  # remove encoder.
                if name.startswith('fc'):
                    # Ignore FC
                    continue
                else:
                    new_state_dict[name] = v
        if misc.get_rank() == 0:
            print(new_state_dict.keys())
        msg = model.load_state_dict(new_state_dict, strict=False)
        if misc.get_rank() == 0:
            logger.info(msg)
        # Adjust Layer Decays
        param_groups = util.param_groups_lrd(model, weight_decay=exp.config.optimizer.weight_decay,
                                             layer_decay=exp.config.layer_decay)
        optimizer = config.optimizer(param_groups)
        trunc_normal_(model.fc.weight, std=2e-5)

    if misc.get_rank() == 0:
        print(model)

    # Prepare Objective Loss function
    criterion = config.criterion()
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
        if misc.get_rank() == 0:
            logger.info('DDP')
        if 'sync_bn' in exp.config and exp.config.sync_bn:
            if misc.get_rank() == 0:
                logger.info('Sync Batch Norm')
            sync_bn_network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(sync_bn_network, broadcast_buffers=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=True)
        model_without_ddp = model.module

    # Train Loops
    for epoch in range(start_epoch, exp.config.epochs):
        # Epoch Train Func
        if misc.get_rank() == 0:
            logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.train()
        stats = train(epoch)

        # Epoch Eval Function
        if misc.get_rank() == 0:
            logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        model.eval()
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
        misc.init_distributed_mode(args)
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
    else:
        torch.manual_seed(args.seed)

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    if misc.get_rank() == 0:
        logger = experiment.logger
        logger.info("PyTorch Version: %s" % (torch.__version__))
        logger.info("Python Version: %s" % (sys.version))
        if torch.cuda.is_available():
            device_list = [torch.cuda.get_device_name(i)
                           for i in range(0, torch.cuda.device_count())]
            logger.info("GPU List: %s" % (device_list))
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
