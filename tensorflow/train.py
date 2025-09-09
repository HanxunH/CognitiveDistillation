import argparse
import mlconfig
import tensorflow as tf
import time
import os
import sys
import numpy as np

# Add parent directory to path to allow tensorflow_impl imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow_impl.models as models
import tensorflow_impl.datasets as datasets
import tensorflow_impl.losses as losses
import tensorflow_impl.util as util
import tensorflow_impl.optimizers as optimizers
from tensorflow_impl.exp_mgmt import ExperimentManager

# Configure TensorFlow
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
    """Save model, optimizer and scheduler states"""
    exp.save_state(model, 'model_state_dict')
    exp.save_state(optimizer, 'optimizer_state_dict')
    exp.save_state(scheduler, 'scheduler_state_dict')


def epoch_exp_stats():
    """
    Set epoch level experiment tracking
    Track Training Loss, this is used by ABL
    """
    stats = {}
    train_loss_list, correct_list = [], []
    
    for images, labels in no_shuffle_loader:
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images.numpy() if hasattr(images, 'numpy') else images)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels.numpy() if hasattr(labels, 'numpy') else labels)
        
        with device_context:
            logits = model(images, training=False)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            predicted = tf.argmax(logits, axis=1)
            correct = tf.equal(predicted, tf.cast(labels, predicted.dtype))
            
            train_loss_list.extend(loss.numpy().tolist())
            correct_list.extend(correct.numpy().tolist())
    
    stats['samplewise_train_loss'] = train_loss_list
    stats['samplewise_correct'] = correct_list
    return stats


def evaluate(target_model, epoch, loader):
    """Evaluate model on a given loader"""
    # Set up metrics
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()
    loss_list, correct_list = [], []
    
    for i, data in enumerate(loader):
        # Prepare batch data
        images, labels = data
        
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images.numpy() if hasattr(images, 'numpy') else images)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels.numpy() if hasattr(labels, 'numpy') else labels)
        
        batch_size = tf.shape(images)[0]
        
        with device_context:
            # Forward pass
            logits = target_model(images, training=False)
            
            # Calculate loss
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
            loss_list.extend(loss.numpy().tolist())
            loss_mean = tf.reduce_mean(loss).numpy()
            
            # Calculate accuracy
            acc = util.accuracy(logits, labels, topk=(1,))[0]
            if isinstance(acc, tf.Tensor):
                acc = acc.numpy()
            
            # Update meters
            loss_meters.update(loss_mean, batch_size.numpy())
            acc_meters.update(acc, batch_size.numpy())
            
            # Track predictions
            predicted = tf.argmax(logits, axis=1)
            correct = tf.equal(predicted, tf.cast(labels, predicted.dtype))
            correct_list.extend(correct.numpy().tolist())
    
    return loss_meters.avg, acc_meters.avg, loss_list, correct_list


def bd_evaluate(target_model, epoch, loader, data):
    """Evaluate backdoor attack success rate"""
    bd_idx = data.poison_test_set.poison_idx
    pred_list, label_list = [], []
    
    for i, batch_data in enumerate(loader):
        # Prepare batch data
        images, labels = batch_data
        
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images.numpy() if hasattr(images, 'numpy') else images)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels.numpy() if hasattr(labels, 'numpy') else labels)
        
        with device_context:
            logits = target_model(images, training=False)
            predicted = tf.argmax(logits, axis=1)
            
            pred_list.append(predicted.numpy())
            label_list.append(labels.numpy())
    
    # Concatenate all predictions and labels
    pred_list = np.concatenate(pred_list)
    label_list = np.concatenate(label_list)
    
    # Calculate ASR (Attack Success Rate)
    asr = (pred_list[bd_idx] == label_list[bd_idx]).sum().item() / len(bd_idx)
    return asr


def train(epoch):
    """Train for one epoch"""
    global global_step, best_acc
    
    # Track exp stats
    if isinstance(criterion, losses.CrossEntropyLoss):
        epoch_stats = epoch_exp_stats()
    else:
        epoch_stats = {}
    
    # Set Meters
    loss_meters = util.AverageMeter()
    acc_meters = util.AverageMeter()
    
    # Training loop
    for i, data in enumerate(train_loader):
        start = time.time()
        
        # Prepare batch data
        images, labels = data
        
        # Convert to TensorFlow tensors if needed
        if not isinstance(images, tf.Tensor):
            images = tf.convert_to_tensor(images.numpy() if hasattr(images, 'numpy') else images)
        if not isinstance(labels, tf.Tensor):
            labels = tf.convert_to_tensor(labels.numpy() if hasattr(labels, 'numpy') else labels)
        
        batch_size = tf.shape(images)[0]
        
        with device_context:
            with tf.GradientTape() as tape:
                # Forward pass
                if isinstance(criterion, losses.CrossEntropyLoss):
                    logits = model(images, training=True)
                    loss = criterion(logits, labels)
                else:
                    # For custom criterion that takes model as input
                    logits, loss = criterion(model, images, labels)
                
                # Calculate mean loss if needed
                if len(loss.shape) > 0:
                    loss = tf.reduce_mean(loss)
            
            # Compute gradients and update weights
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Calculate accuracy
        loss_value = loss.numpy()
        acc = util.accuracy(logits, labels, topk=(1,))[0]
        if isinstance(acc, tf.Tensor):
            acc = acc.numpy()
        
        # Update Meters
        loss_meters.update(loss_value, batch_size.numpy())
        acc_meters.update(acc, batch_size.numpy())
        
        # Log results
        end = time.time()
        time_used = end - start
        if global_step % exp.config.log_frequency == 0:
            # Get current learning rate
            if hasattr(optimizer, 'learning_rate'):
                if callable(optimizer.learning_rate):
                    current_lr = optimizer.learning_rate(global_step).numpy()
                elif isinstance(optimizer.learning_rate, tf.Variable):
                    current_lr = optimizer.learning_rate.numpy()
                else:
                    current_lr = optimizer.learning_rate
            elif hasattr(optimizer, 'lr'):
                if callable(optimizer.lr):
                    current_lr = optimizer.lr(global_step).numpy()
                elif isinstance(optimizer.lr, tf.Variable):
                    current_lr = optimizer.lr.numpy()
                else:
                    current_lr = optimizer.lr
            else:
                current_lr = 0.001  # Default fallback
            
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
    global criterion, model, optimizer, scheduler, device_context
    global train_loader, test_loader, data
    global poison_test_loader, no_shuffle_loader
    global logger, start_epoch, global_step, best_acc
    
    # Set device context
    device_context = tf.device(device)
    
    # Set up Experiments
    logger = exp.logger
    config = exp.config
    
    # Prepare Data
    data = config.dataset(exp)
    loader = data.get_loader(train_shuffle=True)
    train_loader, test_loader, poison_test_loader = loader
    no_shuffle_loader, _, _ = data.get_loader(train_shuffle=False)
    
    # Save poison indices if available
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
    with device_context:
        model = config.model()
    
    # Prepare optimizer
    # The config.optimizer returns a callable that expects model.parameters() in PyTorch
    # For TensorFlow, we need to handle this differently
    optimizer_fn = config.optimizer
    if callable(optimizer_fn):
        # Check if it's a function that expects parameters
        import inspect
        sig = inspect.signature(optimizer_fn)
        if 'params' in sig.parameters or len(sig.parameters) > 0:
            # Call with model's trainable variables
            optimizer = optimizer_fn(model.trainable_variables)
        else:
            # Call without parameters
            optimizer = optimizer_fn()
    else:
        optimizer = optimizer_fn
    
    # If optimizer has a __call__ method (our custom wrappers), call it with the model
    if hasattr(optimizer, '__call__') and hasattr(optimizer, 'params'):
        optimizer = optimizer(model)
    
    # Prepare scheduler (custom implementation needed for TensorFlow)
    scheduler = config.scheduler(optimizer)
    
    print(model.summary() if hasattr(model, 'summary') else model)
    
    # Prepare Objective Loss function
    criterion = config.criterion()
    
    start_epoch = 0
    global_step = 0
    best_acc = 0
    
    # Resume: Load models
    if args.load_model:
        exp_stats = exp.load_epoch_stats()
        if exp_stats:
            start_epoch = exp_stats['epoch'] + 1
            global_step = exp_stats['global_step'] + 1
            model = exp.load_state(model, 'model_state_dict')
            optimizer = exp.load_state(optimizer, 'optimizer_state_dict')
            scheduler = exp.load_state(scheduler, 'scheduler_state_dict')
    
    if args.data_parallel:
        # Skip distributed training as per instructions
        raise NotImplementedError("Skip")
        logger.info("Data parallel not supported in TensorFlow implementation")
    
    # Train Loops
    for epoch in range(start_epoch, exp.config.epochs):
        # Epoch Train Func
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        stats = train(epoch)
        
        # Step scheduler
        if hasattr(scheduler, 'step'):
            scheduler.step()
        elif hasattr(scheduler, '__call__'):
            # For custom schedulers
            current_lr = scheduler(epoch)
            if hasattr(optimizer, 'learning_rate'):
                if isinstance(optimizer.learning_rate, tf.Variable):
                    optimizer.learning_rate.assign(current_lr)
                else:
                    optimizer.learning_rate = current_lr
            elif hasattr(optimizer, 'lr'):
                if isinstance(optimizer.lr, tf.Variable):
                    optimizer.lr.assign(current_lr)
                else:
                    optimizer.lr = current_lr
        
        # Epoch Eval Function
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
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
    
    # Set random seed
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup Experiment
    config_filename = os.path.join(args.exp_config, args.exp_name+'.yaml')
    experiment = ExperimentManager(exp_name=args.exp_name,
                                   exp_path=args.exp_path,
                                   config_file_path=config_filename)
    logger = experiment.logger
    logger.info("TensorFlow Version: %s" % (tf.__version__))
    logger.info("Python Version: %s" % (sys.version))
    
    if len(tf.config.list_physical_devices('GPU')) > 0:
        device_list = [device.name for device in tf.config.list_physical_devices('GPU')]
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