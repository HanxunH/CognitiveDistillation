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
from exp_mgmt import ExperimentManager
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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
        train_loss_list += loss.detach().cpu().numpy().tolist()
        correct_list += correct.detach().cpu().numpy().tolist()

    stats['samplewise_train_loss'] = train_loss_list
    stats['samplewise_correct'] = correct_list
    return stats


@torch.no_grad()
def evaluate(target_model, epoch, loader):
    target_model.eval()
    # Training Evaluations
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()
    loss_list, correct_list = [], []
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        logits = target_model(images)
        loss = F.cross_entropy(logits, labels, reduction='none')
        loss_list += loss.detach().cpu().numpy().tolist()
        loss = loss.mean().item()
        # Calculate acc
        acc = util.accuracy(logits, labels, topk=(1,))[0].item()

        # Update Meters
        loss_meters.update(loss, batch_size)
        acc_meters.update(acc, batch_size)

        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels)
        correct_list += correct.detach().cpu().numpy().tolist()

    return loss_meters.avg, acc_meters.avg, loss_list, correct_list


@torch.no_grad()
def bd_evaluate(target_model, epoch, loader, data):
    bd_idx = data.poison_test_set.poison_idx
    target_model.eval()
    pred_list, label_list = [], []
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = target_model(images)
        _, predicted = torch.max(logits.data, 1)
        pred_list.append(predicted.detach().cpu())
        label_list.append(labels.detach().cpu())
    pred_list = torch.cat(pred_list)
    label_list = torch.cat(label_list)
    asr = (pred_list[bd_idx] == label_list[bd_idx]).sum().item() / len(bd_idx)
    return asr


def train(epoch):
    global global_step, best_acc
    # Track exp stats
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        epoch_stats = epoch_exp_stats()
    else:
        epoch_stats = {}
    # Set Meters
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()

    # Training
    model.train()
    for i, data in enumerate(train_loader):
        start = time.time()
        # Prepare batch data
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        batch_size = images.shape[0]
        model.zero_grad()
        optimizer.zero_grad()

        # Objective function
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            logits = model(images)
            loss = criterion(logits, labels)
        else:
            logits, loss = criterion(model, images, labels)

        # Optimize
        loss.backward()
        optimizer.step()
        # Calculate acc
        loss = loss.item()
        acc = util.accuracy(logits, labels, topk=(1,))[0].item()

        # Update Meters
        loss_meters.update(loss, batch_size)
        acc_meters.update(acc, batch_size)

        # Log results
        end = time.time()
        time_used = end - start
        if global_step % exp.config.log_frequency == 0:
            payload = {
                "acc_avg": acc_meters.avg,
                "loss_avg": loss_meters.avg,
                "lr": optimizer.param_groups[0]['lr']
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
    model = config.model().to(device)
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
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
        scheduler = exp.load_state(scheduler, 'scheduler_state_dict')

    if args.data_parallel:
        model = torch.nn.DataParallel(model).to(device)
        logger.info("Using torch.nn.DataParallel")

    # Train Loops
    for epoch in range(start_epoch, exp.config.epochs):
        # Epoch Train Func
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.train()
        stats = train(epoch)
        scheduler.step()

        # Epoch Eval Function
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        model.eval()
        eval_loss, eval_acc, ll, cl = evaluate(model, epoch, test_loader)
        if eval_acc > best_acc:
            best_acc = eval_acc
        payload = 'Eval Loss: %.4f Eval Acc: %.4f Best Acc: %.4f' % \
            (eval_loss, eval_acc, best_acc)
        logger.info('\033[33m'+payload+'\033[0m')
        stats['eval_acc'] = eval_acc
        stats['best_acc'] = best_acc
        stats['epoch'] = epoch
        stats['samplewise_eval_loss'] = ll
        stats['samplewise_eval_correct'] = cl

        # Epoch Backdoor Eval
        if poison_test_loader is not None:
            asr = bd_evaluate(model, epoch, poison_test_loader, data)
            payload = 'Model Backdoor Attack success rate %.4f' % (asr)
            logger.info('\033[33m'+payload+'\033[0m')
            stats['eval_asr'] = asr

        # Save Model
        exp.save_epoch_stats(epoch=epoch, exp_stats=stats)
        save_model()
    return


if __name__ == '__main__':
    global exp
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
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
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)
