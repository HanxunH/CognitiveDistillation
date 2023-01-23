import argparse
import mlconfig
import torch
import torch.nn.functional as F
import time
import models
import datasets
import losses
import util
import os
import sys
import json
import numpy as np
import copy
import analysis
from exp_mgmt import ExperimentManager
from torchvision import transforms
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser(description='HiddenVision')
# General Options
parser.add_argument('--seed', type=int, default=0, help='seed')
# Experiment Options
parser.add_argument('--exp_name', default='test_exp', type=str)
parser.add_argument('--exp_path', default='experiments/test', type=str)
parser.add_argument('--exp_config', default='configs/test', type=str)
parser.add_argument('--data_parallel', action='store_true', default=False)
# CD Parameters
parser.add_argument('--logits_train_mask_filename',
                    default='cd_train_mask_p=1_c=1_gamma=0.010000_beta=1.000000_steps=100_step_size=0.100000.pt')
parser.add_argument('--logits_test_mask_filename',
                    default='cd_bd_test_mask_p=1_c=1_gamma=0.010000_beta=1.000000_steps=100_step_size=0.100000.pt')
parser.add_argument('--fe_train_mask_filename',
                    default='cd_fe_train_mask_p=1_c=1_gamma=0.001000_beta=10.000000_steps=100_step_size=0.050000.pt')
parser.add_argument('--fe_test_mask_filename',
                    default='cd_fe_bd_test_mask_p=1_c=1_gamma=0.001000_beta=10.000000_steps=100_step_size=0.050000.pt')
parser.add_argument('--norm_only', action='store_true', default=False)
# Unlearning Options
parser.add_argument('--method', type=str, default="CD", help='Detection Method')
parser.add_argument('--finetune_epochs', type=int, default=20, help='Number of unlearning steps')
parser.add_argument('--unlearn_epochs', type=int, default=5, help='Number of unlearning steps')
parser.add_argument('--unlearn_lr', type=float, default=5e-4, help='Learning rate for unlearning')
parser.add_argument('--unlearn_precent', type=int, default=0.025, help='Number of unlearning samples')
parser.add_argument('--safe_precent', type=int, default=0.7, help='Number of safe training samples')


@torch.no_grad()
def evaluate(target_model, loader):
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

    return acc_meters.avg


@torch.no_grad()
def bd_evaluate(target_model, loader, data):
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


def load_train_loss(exp):
    loss_list = []
    for e in range(exp.config.epochs):
        stats = exp.load_epoch_stats(e)
        loss = np.array(stats['samplewise_train_loss'])
        loss_list.append(loss)
    return np.array(loss_list)


def min_max_normalization(x):
    x_min = torch.min(x)
    x_max = torch.max(x)
    norm = (x - x_min) / (x_max - x_min)
    norm = torch.clamp(norm, 0, 1)
    return norm


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
    _, test_loader, poison_test_loader = loader

    # Prepare Model
    model = config.model().to(device)
    # Resume: Load models
    model = exp.load_state(model, 'model_state_dict')

    # Load poison_idx/clean_idx
    if 'CIFAR10' in config.dataset.train_d_type:
        # CIFAR10
        num_of_classes = 10
        train_poison_idx = np.load(os.path.join(exp.exp_path, 'train_poison_idx.npy'))
        train_clean_idx = np.setxor1d(train_poison_idx, range(len(data.train_set.data)))
    elif 'ImageNet' in config.dataset.train_d_type:
        # Subset ImageNet (ISSBA/BadNet)
        num_of_classes = 200
        train_poison_idx = np.load(os.path.join(exp.exp_path, 'train_poison_idx.npy'))
        train_clean_idx = np.setxor1d(train_poison_idx, range(len(data.train_set.samples)))
    elif 'GTSRB' in config.dataset.train_d_type:
        num_of_classes = 43
        train_poison_idx = np.load(os.path.join(exp.exp_path, 'train_poison_idx.npy'))
        train_clean_idx = np.setxor1d(train_poison_idx, range(len(data.train_set)))
    else:
        raise('Not Impelmented')

    # Load data
    if 'CD' in args.method:
        if 'FE' in args.method:
            train_filename = args.fe_train_mask_filename
            test_filename = args.fe_test_mask_filename
        else:
            train_filename = args.logits_train_mask_filename
            test_filename = args.logits_test_mask_filename
    elif args.method == 'STRIP':
        train_filename = 'train_STRIP_entropy.pt'
        test_filename = 'bd_test_STRIP_entropy.pt'
    elif args.method == 'SS' or args.method == 'AC' or args.method == 'LID':
        train_filename = 'train_features.pt'
        test_filename = 'bd_test_features.pt'
    elif args.method == 'ABL':
        loss_list = load_train_loss(exp)
        train_filename = None
        test_filename = None
    elif args.method == 'Frequency':
        train_results = data.train_set
        test_results = data.poison_test_set
        train_filename = None
        test_filename = None
    elif args.method == 'FCT':
        train_filename = 'train_fct.pt'
        test_filename = 'bd_test_fct.pt'
    else:
        raise('Unknown method')

    # Handle detection results
    if train_filename is not None and test_filename is not None:
        train_filename = os.path.join(exp.exp_path, train_filename)
        test_filename = os.path.join(exp.exp_path, test_filename)
        train_results = torch.load(train_filename)
        test_results = torch.load(test_filename)
        if args.method == 'SS' or args.method == 'AC' or args.method == 'LID':
            train_results = train_results.flatten(start_dim=1)
            test_results = test_results.flatten(start_dim=1)

    # Run detection analysis
    if args.method == 'ABL':
        loss_list = load_train_loss(exp)
        detector = analysis.ABLAnalysis()
        train_scores = detector.analysis(loss_list)
    elif args.method == 'STRIP':
        # STRIP already extracted the H, lower for bd, use 1 - score
        train_scores = 1 - min_max_normalization(train_results.detach().cpu()).numpy()
    elif args.method in ['AC', 'SS']:
        # Need test prediction as targets
        model = config.model().to(device)
        model = exp.load_state(model, 'model_state_dict')
        if args.data_parallel:
            model = torch.nn.DataParallel(model).to(device)
            logger.info("Using torch.nn.DataParallel")
        loader = data.get_loader(train_shuffle=False)
        _, _, bd_test_loader = loader
        train_cls_idx = [np.where(np.array(data.train_set.targets) == i)[0] for i in range(num_of_classes)]
        if args.method == 'AC':
            detector = analysis.ACAnalysis()
        elif args.method == 'SS':
            detector = analysis.SSAnalysis()
        train_scores = detector.analysis(train_results, data.train_set.targets, train_cls_idx)
    elif args.method == 'Frequency':
        detector = analysis.FrequencyAnalysis()
        train_scores = detector.analysis(train_results)
    elif args.method == 'FCT':
        # FCT already extracted the consistency score
        train_scores = train_results.detach().cpu().numpy()
    elif 'LID' in args.method:
        detector = analysis.LIDAnalysis()
        train_scores = detector.analysis(train_results)
    elif 'CD' in args.method:
        detector = analysis.CognitiveDistillationAnalysis(od_type=args.method, norm_only=args.norm_only)
        detector.train(train_results)
        train_scores = detector.analysis(train_results, is_test=False)
    else:
        raise('Unknown Method')

    assert train_scores.shape == (len(data.train_set), )

    # Select unlearning images
    sorted_idx = torch.argsort(torch.from_numpy(train_scores))  # Higher are anomolies
    target_imgs = []
    target_labels = []
    unlearn_idx_split = int(len(data.train_set) * args.unlearn_precent)
    for i in sorted_idx[-unlearn_idx_split:]:
        if 'ImageNet' in config.dataset.train_d_type:
            img, label = data.train_set.samples[i]
        elif 'GTSRB' in config.dataset.train_d_type:
            img, label = data.train_set._samples[i]
        else:
            img, label = data.train_set.data[i], data.train_set.targets[i]
        target_imgs.append(img)
        target_labels.append(label)
    bd_in_unlearn = np.intersect1d(train_poison_idx, sorted_idx[-unlearn_idx_split:])

    # Select safe images
    safe_imgs = []
    safe_labels = []
    safe_idx_split = int(len(data.train_set) * args.safe_precent)

    for i in sorted_idx[:safe_idx_split]:
        if 'ImageNet' in config.dataset.train_d_type:
            img, label = data.train_set.samples[i]
        elif 'GTSRB' in config.dataset.train_d_type:
            img, label = data.train_set._samples[i]
        else:
            img, label = data.train_set.data[i], data.train_set.targets[i]
        safe_imgs.append(img)
        safe_labels.append(label)

    bd_in_safe = np.intersect1d(train_poison_idx, sorted_idx[:safe_idx_split])
    logger.info('Unlearn samples: %d, Safe samples: %d' % (len(target_imgs), len(safe_imgs)))
    logger.info('BD in Unlearn samples: %d, BD in  Safe samples: %d' % (len(bd_in_unlearn), len(bd_in_safe)))

    # Build Loader
    unlearn_data = copy.deepcopy(data.train_set)
    safe_data = copy.deepcopy(data.train_set)

    if 'ImageNet' in config.dataset.train_d_type:
        unlearn_data.samples = list(zip(target_imgs, target_labels))
        safe_data.samples = list(zip(safe_imgs, safe_labels))
    elif 'GTSRB' in config.dataset.train_d_type:
        unlearn_data._samples = list(zip(target_imgs, target_labels))
        safe_data._samples = list(zip(safe_imgs, safe_labels))
    else:
        unlearn_data.data = target_imgs
        unlearn_data.targets = target_labels
        safe_data.data = safe_imgs
        safe_data.targets = safe_labels
        safe_data.transform = datasets.utils.transform_options['CIFAR10_ABL']['train_transform']
        safe_data.transform = transforms.Compose(safe_data.transform)

    unlearn_loader = DataLoader(dataset=unlearn_data, pin_memory=False,
                                batch_size=128, drop_last=False,
                                num_workers=4, shuffle=True)

    safe_loader = DataLoader(dataset=safe_data, pin_memory=False,
                             batch_size=128, drop_last=False,
                             num_workers=4, shuffle=True)

    logger.info("="*20 + "Before Unlearning" + "="*20)
    model.eval()
    ca = evaluate(model, test_loader)
    asr = bd_evaluate(model, poison_test_loader, data)
    payload = 'Clean Acc (CA): %.4f Attack Success Rate (ASR): %.4f' % (ca, asr)

    # Save results
    stats = {
        'ca': ca,
        'asr': asr,
        'bd_in_unlearn': len(bd_in_unlearn) / len(target_imgs),
        'bd_in_safe': len(bd_in_safe) / len(safe_imgs),
        'normalized_bd_in_safe': len(bd_in_safe) / len(train_poison_idx),
    }
    filename = 'unlearn_finetune_with_{:s}_epoch_{:d}.json'.format(args.method, 0)
    filename = os.path.join(exp.stas_eval_path, filename)
    with open(filename, 'w') as outfile:
        json.dump(stats, outfile)
    logger.info('\033[33m'+payload+'\033[0m')

    # Unlearning and Finetune
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4, momentum=0.9, nesterov=True)

    logger.info("="*20 + "Finetune" + "="*20)
    for step in range(0, args.finetune_epochs):
        if step < args.finetune_epochs * 0.5:
            lr = 0.01
        else:
            lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train()

        for images, labels in safe_loader:
            optimizer.zero_grad()
            model.zero_grad()
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Eval each Epoch
        model.eval()
        ca = evaluate(model, test_loader)
        asr = bd_evaluate(model, poison_test_loader, data)
        payload = 'Epoch %d Clean Acc (CA): %.4f Attack Success Rate (ASR): %.4f' % (step, ca, asr)
        logger.info('\033[33m'+payload+'\033[0m')

    logger.info("="*20 + "Unlearning" + "="*20)
    for step in range(0, args.unlearn_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
        model.train()
        for images, labels in safe_loader:
            optimizer.zero_grad()
            model.zero_grad()
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        for param_group in optimizer.param_groups:
            param_group['lr'] = 1.e-5
        model.train()
        for images, labels in unlearn_loader:
            optimizer.zero_grad()
            model.zero_grad()
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = - F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Eval each Epoch
        model.eval()
        ca = evaluate(model, test_loader)
        asr = bd_evaluate(model, poison_test_loader, data)
        payload = 'Epoch %d Clean Acc (CA): %.4f Attack Success Rate (ASR): %.4f' % (step, ca, asr)
        logger.info('\033[33m'+payload+'\033[0m')

        # Save results
        stats = {
            'ca': ca,
            'asr': asr,
            'step': step
        }
        filename = 'unlearn_finetune_with_{:s}_epoch_{:d}.json'.format(args.method, step+1)
        filename = os.path.join(exp.stas_eval_path, filename)
        with open(filename, 'w') as outfile:
            json.dump(stats, outfile)


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
