import argparse
import mlconfig
import tensorflow as tf
import time
import tensorflow_impl.models as models
import tensorflow_impl.datasets as datasets
import tensorflow_impl.losses as losses
import tensorflow_impl.util as util
import os
import sys
import numpy as np
from tensorflow_impl.exp_mgmt import ExperimentManager

# TensorFlow device configuration
if len(tf.config.list_physical_devices('GPU')) > 0:
    # Enable memory growth for GPU
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass
    device = '/GPU:0'
else:
    device = '/CPU:0'

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
    """Set epoch level experiment tracking - Track Training Loss, used by ABL"""
    stats = {}
    train_loss_list, correct_list = [], []
    
    for images, labels in no_shuffle_loader:
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels, dtype=tf.int64)
        
        with tf.device(device):
            logits = model(images, training=False)
            # Use sparse categorical crossentropy for per-sample loss (reduction='none' equivalent)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            predicted = tf.argmax(logits, axis=1, output_type=tf.int64)
            correct = tf.equal(predicted, labels)
            
            train_loss_list += loss.numpy().tolist()
            correct_list += correct.numpy().tolist()

    stats['samplewise_train_loss'] = train_loss_list
    stats['samplewise_correct'] = correct_list
    return stats


def evaluate(target_model, epoch, loader):
    """Evaluate model performance on given loader"""
    # Training Evaluations
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()
    loss_list, correct_list = [], []
    
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels, dtype=tf.int64)
        
        batch_size = tf.shape(images)[0]
        
        with tf.device(device):
            logits = target_model(images, training=False)
            # Use sparse categorical crossentropy for per-sample loss (reduction='none' equivalent)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            loss_list += loss.numpy().tolist()
            loss_mean = tf.reduce_mean(loss).numpy()
            
            # Calculate acc
            acc = util.accuracy(logits, labels, topk=(1,))[0].numpy()

            # Update Meters
            loss_meters.update(loss_mean, batch_size.numpy())
            acc_meters.update(acc, batch_size.numpy())

            predicted = tf.argmax(logits, axis=1, output_type=tf.int64)
            correct = tf.equal(predicted, labels)
            correct_list += correct.numpy().tolist()

    return loss_meters.avg, acc_meters.avg, loss_list, correct_list


def bd_evaluate(target_model, epoch, loader, data):
    """Evaluate backdoor attack success rate"""
    bd_idx = data.poison_test_set.poison_idx
    pred_list, label_list = [], []
    
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels, dtype=tf.int64)
        
        with tf.device(device):
            logits = target_model(images, training=False)
            predicted = tf.argmax(logits, axis=1, output_type=tf.int64)
            pred_list.append(predicted.numpy())
            label_list.append(labels.numpy())
    
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    asr = (pred_list[bd_idx] == label_list[bd_idx]).sum() / len(bd_idx)
    return asr


def train(epoch):
    """Train function - kept for compatibility but not used in evaluation"""
    global global_step, best_acc
    # Track exp stats
    if isinstance(criterion, (tf.keras.losses.SparseCategoricalCrossentropy, losses.CrossEntropyLoss)):
        epoch_stats = epoch_exp_stats()
    else:
        epoch_stats = {}
    # Set Meters
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()

    # Training
    for i, data in enumerate(train_loader):
        start = time.time()
        # Prepare batch data
        images, labels = data
        
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images, dtype=tf.float32)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels, dtype=tf.int64)
        
        batch_size = tf.shape(images)[0]

        # Objective function
        with tf.GradientTape() as tape:
            if isinstance(criterion, (tf.keras.losses.SparseCategoricalCrossentropy, losses.CrossEntropyLoss)):
                logits = model(images, training=True)
                loss = criterion(labels, logits)
            else:
                logits, loss = criterion(model, images, labels)

        # Optimize
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Calculate acc
        loss_value = loss.numpy()
        acc = util.accuracy(logits, labels, topk=(1,))[0].numpy()

        # Update Meters
        loss_meters.update(loss_value, batch_size.numpy())
        acc_meters.update(acc, batch_size.numpy())

        # Log results
        end = time.time()
        time_used = end - start
        if global_step % exp.config.log_frequency == 0:
            # Get learning rate from optimizer
            lr = optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                lr = lr.numpy()
            
            payload = {
                "acc_avg": acc_meters.avg,
                "loss_avg": loss_meters.avg,
                "lr": lr
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
    with tf.device(device):
        model = config.model()
        optimizer = config.optimizer()
    print(model.summary() if hasattr(model, 'summary') else str(model))
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
        exp_stats = {}
        global_step = 1
    
    model = exp.load_state(model, 'model_state_dict')
    optimizer = exp.load_state(optimizer, 'optimizer_state_dict')

    if args.data_parallel:
        # Skip parallel processing as per project requirements
        raise NotImplementedError("Skip")

    # Epoch Eval Function
    logger.info("="*20 + "Eval Epoch %d" % (exp.config.epochs) + "="*20)
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
    tf.random.set_seed(args.seed)
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    logger = experiment.logger
    logger.info("TensorFlow Version: %s" % (tf.__version__))
    logger.info("Python Version: %s" % (sys.version))
    
    # GPU information
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        device_list = [gpu.name for gpu in gpu_devices]
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