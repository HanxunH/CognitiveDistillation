import argparse
import mlconfig
import tensorflow as tf
import random
import numpy as np
import datasets
import time
import json
import os
import sys
import os.path
# Add the tensorflow_impl directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util
import models
import analysis
from exp_mgmt import ExperimentManager
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pprint import pformat
from scipy.stats import skew
from sklearn.metrics import confusion_matrix

# TensorFlow device configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
    except RuntimeError as e:
        print(e)


parser = argparse.ArgumentParser(description='CognitiveDistillation')
# Experiment Options
parser.add_argument('--exp_name', default='rn18', type=str)
parser.add_argument('--exp_path', default='experiments', type=str)
parser.add_argument('--exp_config', default='configs', type=str)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--method', type=str, default="CD")

# CD Parameters
parser.add_argument('--logits_train_mask_filename',
                    default='cd_train_mask_p=1_c=1_gamma=0.010000_beta=1.000000_steps=100_step_size=0.100000.npy')
parser.add_argument('--logits_test_mask_filename',
                    default='cd_bd_test_mask_p=1_c=1_gamma=0.010000_beta=1.000000_steps=100_step_size=0.100000.npy')
parser.add_argument('--fe_train_mask_filename',
                    default='cd_fe_train_mask_p=1_c=1_gamma=0.001000_beta=10.000000_steps=100_step_size=0.050000.npy')
parser.add_argument('--fe_test_mask_filename',
                    default='cd_fe_bd_test_mask_p=1_c=1_gamma=0.001000_beta=10.000000_steps=100_step_size=0.050000.npy')
parser.add_argument('--norm_only', action='store_true', default=False)


def min_max_normalization(x):
    x_min = tf.reduce_min(x)
    x_max = tf.reduce_max(x)
    norm = (x - x_min) / (x_max - x_min)
    norm = tf.clip_by_value(norm, 0, 1)
    return norm


def extract_predictions(model, loader):
    pred_list = []
    for images, _ in loader:
        # Convert to TensorFlow tensor if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images)
        pred = tf.argmax(model(images, training=False), axis=1)
        pred_list.append(pred)
    return tf.concat(pred_list, axis=0)


def load_train_loss(exp):
    loss_list = []
    for e in range(exp.config.epochs):
        stats = exp.load_epoch_stats(e)
        loss = np.array(stats['samplewise_train_loss'])
        loss_list.append(loss)
    return np.array(loss_list)


def main(exp):
    # Setup model and exp to run
    config = exp.config
    data = config.dataset(exp)

    # Load poison_idx/clean_idx
    if 'CIFAR10' in config.dataset.train_d_type:
        # CIFAR10
        num_of_classes = 10
        train_poison_idx = np.load(os.path.join(exp.exp_path, 'train_poison_idx.npy'))
        train_clean_idx = np.setxor1d(train_poison_idx, range(len(data.train_set.data)))
        bd_test_poison_idx = np.load(os.path.join(exp.exp_path, 'bd_test_poison_idx.npy'))
        bd_test_clean_idx = np.setxor1d(bd_test_poison_idx, range(len(data.poison_test_set.data)))
    elif 'ImageNet' in config.dataset.train_d_type:
        # Subset ImageNet (ISSBA/BadNet)
        num_of_classes = 200
        train_poison_idx = np.load(os.path.join(exp.exp_path, 'train_poison_idx.npy'))
        train_clean_idx = np.setxor1d(train_poison_idx, range(len(data.train_set.samples)))
        bd_test_poison_idx = np.load(os.path.join(exp.exp_path, 'bd_test_poison_idx.npy'))
        bd_test_clean_idx = np.setxor1d(bd_test_poison_idx, range(len(data.poison_test_set.samples)))
        data.train_set.targets = np.array([target for path, target in data.train_set.samples])
        data.test_set.targets = np.array([target for path, target in data.test_set.samples])
        data.poison_test_set.targets = np.array([target for path, target in data.poison_test_set.samples])
    elif 'GTSRB' in config.dataset.train_d_type:
        num_of_classes = 43
        train_poison_idx = np.load(os.path.join(exp.exp_path, 'train_poison_idx.npy'))
        train_clean_idx = np.setxor1d(train_poison_idx, range(len(data.train_set)))
        bd_test_poison_idx = np.load(os.path.join(exp.exp_path, 'bd_test_poison_idx.npy'))
        bd_test_clean_idx = np.setxor1d(bd_test_poison_idx, range(len(data.poison_test_set)))
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
        train_filename = 'train_STRIP_entropy.npy'
        test_filename = 'bd_test_STRIP_entropy.npy'
    elif args.method == 'SS' or args.method == 'AC' or args.method == 'LID':
        train_filename = 'train_features.npy'
        test_filename = 'bd_test_features.npy'
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
        train_filename = 'train_fct.npy'
        test_filename = 'bd_test_fct.npy'
    else:
        raise('Unknown method')

    # Handle detection results
    if train_filename is not None and test_filename is not None:
        train_filename = os.path.join(exp.exp_path, train_filename)
        test_filename = os.path.join(exp.exp_path, test_filename)
        train_results = np.load(train_filename)
        test_results = np.load(test_filename)
        if args.method == 'SS' or args.method == 'AC' or args.method == 'LID':
            # Flatten from the second dimension onward
            original_shape = train_results.shape
            train_results = train_results.reshape(original_shape[0], -1)
            original_shape = test_results.shape
            test_results = test_results.reshape(original_shape[0], -1)

    # Run analysis
    if args.method == 'ABL':
        loss_list = load_train_loss(exp)
        detector = analysis.ABLAnalysis()
        train_scores = detector.analysis(loss_list)
        test_scores = None
    elif args.method == 'STRIP':
        # STRIP already extracted the H, lower for bd, use 1 - score
        train_results_tf = tf.convert_to_tensor(train_results)
        test_results_tf = tf.convert_to_tensor(test_results)
        train_scores = 1 - min_max_normalization(train_results_tf).numpy()
        test_scores = 1 - min_max_normalization(test_results_tf).numpy()
    elif args.method == 'FCT':
        # FCT already extracted the consistency score
        train_scores = train_results
        test_scores = test_results
    elif args.method in ['AC', 'SS']:
        # Need test prediction as targets
        model = config.model()
        model = exp.load_state(model, 'model_state_dict')
        if args.data_parallel:
            # Skip parallel processing as per instructions
            raise NotImplementedError("Skip")
        loader = data.get_loader(train_shuffle=False)
        _, _, bd_test_loader = loader
        y_test_pred = extract_predictions(model, bd_test_loader).numpy()
        train_cls_idx = [np.where(np.array(data.train_set.targets) == i)[0] for i in range(num_of_classes)]
        test_cls_idx = [np.where(y_test_pred == i)[0] for i in range(num_of_classes)]
        if args.method == 'AC':
            detector = analysis.ACAnalysis()
        elif args.method == 'SS':
            detector = analysis.SSAnalysis()
        train_scores = detector.analysis(train_results, data.train_set.targets, train_cls_idx)
        test_scores = detector.analysis(test_results, y_test_pred, test_cls_idx)
    elif args.method == 'Frequency':
        detector = analysis.FrequencyAnalysis()
        train_scores = detector.analysis(train_results)
        test_scores = detector.analysis(test_results)
    elif 'LID' in args.method:
        detector = analysis.LIDAnalysis()
        train_scores = detector.analysis(train_results)
        test_scores = detector.analysis(test_results)
    elif 'CD' in args.method:
        detector = analysis.CognitiveDistillationAnalysis(od_type=args.method, norm_only=args.norm_only)
        detector.train(train_results)
        train_scores = detector.analysis(train_results, is_test=False)
        test_scores = detector.analysis(test_results, is_test=True)
    else:
        raise('Unknown Method')

    assert train_scores.shape == (len(data.train_set), )

    # Calculate metrics
    results = {}
    display_results = {}

    # Training set analysis
    train_y = np.zeros(len(data.train_set))
    train_y[train_poison_idx] = 1
    train_y[train_clean_idx] = 0

    fpr, tpr, _ = roc_curve(train_y, train_scores, pos_label=1)
    precision, recall, _ = precision_recall_curve(train_y, train_scores, pos_label=1)
    roc_auc = roc_auc_score(train_y, train_scores)
    pr_auc = auc(recall, precision)
    map = average_precision_score(train_y, train_scores, pos_label=1)
    results['train_fpr_list'] = fpr.tolist()
    results['train_tpr_list'] = tpr.tolist()
    results['train_roc_auc'] = roc_auc
    results['train_map'] = map
    results['train_pr_auc'] = pr_auc
    display_results['train_roc_auc'] = roc_auc
    display_results['train_map'] = map
    display_results['train_pr_auc'] = pr_auc

    # BD test set analysis
    if test_scores is not None:
        assert test_scores.shape == (len(data.poison_test_set), )
        test_y = np.zeros(len(data.poison_test_set))
        test_y[bd_test_poison_idx] = 1
        test_y[bd_test_clean_idx] = 0

        fpr, tpr, _ = roc_curve(test_y, test_scores, pos_label=1)
        precision, recall, _ = precision_recall_curve(test_y, test_scores, pos_label=1)
        roc_auc = roc_auc_score(test_y, test_scores)
        pr_auc = auc(recall, precision)
        map = average_precision_score(test_y, test_scores, pos_label=1)
        results['test_fpr_list'] = fpr.tolist()
        results['test_tpr_list'] = tpr.tolist()
        results['test_roc_auc'] = roc_auc
        results['test_map'] = map
        results['test_pr_auc'] = pr_auc
        display_results['test_roc_auc'] = roc_auc
        display_results['test_map'] = map
        display_results['test_pr_auc'] = pr_auc

        # Use standard deviation for testing
        # Assume have access to 1% clean images
        s = len(train_results) * 0.1
        s = int(s)
        threshold = 0.5
        if args.method in ['AC', 'SS']:
            train_cls_idx = [np.where(np.array(data.train_set.targets)[train_clean_idx][:s] == i)[0]
                             for i in range(num_of_classes)]
            detector.train(train_results[train_clean_idx][:s],
                           data.train_set.targets[train_clean_idx][:s],
                           train_cls_idx)
            y_pred = detector.predict(test_results, y_test_pred, test_cls_idx, t=threshold)
        elif args.method == 'STRIP':
            # STRIP already extracted the H, lower for bd
            mean, std = np.mean(train_results[train_clean_idx][:s]), np.std(train_results[train_clean_idx][:s])
            p = (mean - test_results) / std
            y_pred = np.where((p > threshold) & (p > 0), 1, 0)
        elif args.method == 'Frequency':
            y_pred = detector.predict(test_results, t=threshold)
        elif args.method == 'FCT':
             # FCT already extracted the consistency score
            mean, std = np.mean(train_results[train_clean_idx][s:]), np.std(train_results[train_clean_idx][s:])
            p = (test_results - mean) / std
            y_pred = np.where((p > threshold) & (p > 0), 1, 0)
        else:
            detector.train(train_results[train_clean_idx][:s])
            y_pred = detector.predict(test_scores, t=threshold)
        cm = confusion_matrix(test_y, y_pred)
        print(cm)
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        results['test_fpr'] = FPR[1]
        results['test_tpr'] = TPR[1]
        display_results['test_fpr'] = FPR[1]
        display_results['test_tpr'] = TPR[1]

    # Save results
    payload = pformat(display_results, width=1)
    payload = '%s Results:\n' % args.method + payload
    print('\033[94m'+payload+'\033[0m')
    filename = 'detection_results_%s.json' % args.method
    filename = os.path.join(exp.exp_path, filename)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    return


if __name__ == '__main__':
    args = parser.parse_args()

    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    exp = ExperimentManager(exp_name=args.exp_name, exp_path=args.exp_path,
                            config_file_path=config_filename)
    logger = exp.logger
    logger.info("TensorFlow Version: %s" % (tf.__version__))

    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        device_list = [device.name for device in tf.config.experimental.list_physical_devices('GPU')]
        logger.info("GPU List: %s" % (device_list))

    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    for key in exp.config:
        logger.info("%s: %s" % (key, exp.config[key]))

    start = time.time()
    main(exp)
    end = time.time()

    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days" % cost
    logger.info(payload)