import os
import tensorflow_impl.util as util
import datetime
import shutil
import mlconfig
import tensorflow as tf
import json
import tensorflow_impl.misc as misc
from collections import OrderedDict

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


class ExperimentManager():
    def __init__(self, exp_name, exp_path, config_file_path, eval_mode=False):
        if exp_name == '' or exp_name is None:
            exp_name = 'exp_at' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_path = os.path.join(exp_path, exp_name)
        checkpoint_path = os.path.join(exp_path, 'checkpoints')
        log_filepath = os.path.join(exp_path, exp_name) + ".log"
        stas_hist_path = os.path.join(exp_path, 'stats')
        stas_eval_path = os.path.join(exp_path, 'stats_eval')

        if misc.get_rank() == 0 and not eval_mode:
            util.build_dirs(exp_path)
            util.build_dirs(checkpoint_path)
            util.build_dirs(stas_hist_path)
            util.build_dirs(stas_eval_path)

        if config_file_path is not None:
            dst = os.path.join(exp_path, exp_name+'.yaml')
            if dst != config_file_path and misc.get_rank() == 0 and not eval_mode:
                shutil.copyfile(config_file_path, dst)
            config = mlconfig.load(config_file_path)
            config.set_immutable()
        else:
            config = None

        self.exp_name = exp_name
        self.exp_path = exp_path
        self.checkpoint_path = checkpoint_path
        self.log_filepath = log_filepath
        self.stas_hist_path = stas_hist_path
        self.stas_eval_path = stas_eval_path
        self.config = config
        self.logger = None
        self.eval_mode = eval_mode
        if misc.get_rank() == 0:
            self.logger = util.setup_logger(name=self.exp_path, log_file=self.log_filepath,
                                            ddp=misc.get_world_size() > 1)

    def save_eval_stats(self, exp_stats, name):
        filename = '%s_exp_stats_eval.json' % name
        filename = os.path.join(self.stas_eval_path, filename)
        with open(filename, 'w') as outfile:
            json.dump(exp_stats, outfile)
        return

    def load_eval_stats(self, name):
        filename = '%s_exp_stats_eval.json' % name
        filename = os.path.join(self.stas_eval_path, filename)
        if os.path.exists(filename):
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                return data
        else:
            return None

    def save_epoch_stats(self, epoch, exp_stats):
        filename = 'exp_stats_epoch_%d.json' % epoch
        filename = os.path.join(self.stas_hist_path, filename)
        with open(filename, 'w') as outfile:
            json.dump(exp_stats, outfile)
        return

    def load_epoch_stats(self, epoch=None):
        if epoch is not None:
            filename = 'exp_stats_epoch_%d.json' % epoch
            filename = os.path.join(self.stas_hist_path, filename)
            with open(filename, 'r') as json_file:
                data = json.load(json_file)
                return data
        else:
            epoch = self.config.epochs
            filename = 'exp_stats_epoch_%d.json' % epoch
            filename = os.path.join(self.stas_hist_path, filename)
            while not os.path.exists(filename) and epoch >= 0:
                epoch -= 1
                filename = 'exp_stats_epoch_%d.json' % epoch
                filename = os.path.join(self.stas_hist_path, filename)

            if not os.path.exists(filename):
                return None

            with open(filename, 'rb') as json_file:
                data = json.load(json_file)
                return data
        return None

    def save_state(self, target, name):
        """Save model or optimizer state in TensorFlow format"""
        filename = os.path.join(self.checkpoint_path, name)
        
        if hasattr(target, 'save_weights'):
            # For Keras models
            target.save_weights(filename + '.h5')
            if misc.get_rank() == 0:
                self.logger.info('%s saved at %s.h5' % (name, filename))
        elif hasattr(target, 'get_config'):
            # For optimizers and other configurable objects
            config = target.get_config()
            weights = target.get_weights() if hasattr(target, 'get_weights') else None
            state = {'config': config}
            if weights is not None:
                # Convert numpy arrays to lists for JSON serialization
                state['weights'] = [w.tolist() if hasattr(w, 'tolist') else w for w in weights]
            
            with open(filename + '.json', 'w') as f:
                json.dump(state, f)
            if misc.get_rank() == 0:
                self.logger.info('%s saved at %s.json' % (name, filename))
        else:
            # For schedulers or other custom objects, save their state as JSON
            if hasattr(target, '__dict__'):
                state = {k: v for k, v in target.__dict__.items() 
                        if not k.startswith('_') and not callable(v)}
                with open(filename + '.json', 'w') as f:
                    json.dump(state, f, default=str)
                if misc.get_rank() == 0:
                    self.logger.info('%s saved at %s.json' % (name, filename))
        return

    def load_state(self, target, name, strict=True):
        """Load model or optimizer state in TensorFlow format"""
        filename = os.path.join(self.checkpoint_path, name)
        
        if hasattr(target, 'load_weights'):
            # For Keras models
            if os.path.exists(filename + '.h5'):
                target.load_weights(filename + '.h5')
                if misc.get_rank() == 0 and not self.eval_mode:
                    self.logger.info('%s loaded from %s.h5' % (name, filename))
        elif hasattr(target, 'get_config'):
            # For optimizers and other configurable objects
            if os.path.exists(filename + '.json'):
                with open(filename + '.json', 'r') as f:
                    state = json.load(f)
                if 'weights' in state and hasattr(target, 'set_weights'):
                    import numpy as np
                    weights = [np.array(w) for w in state['weights']]
                    target.set_weights(weights)
                if misc.get_rank() == 0 and not self.eval_mode:
                    self.logger.info('%s loaded from %s.json' % (name, filename))
        else:
            # For schedulers or other custom objects
            if os.path.exists(filename + '.json'):
                with open(filename + '.json', 'r') as f:
                    state = json.load(f)
                for k, v in state.items():
                    if hasattr(target, k):
                        setattr(target, k, v)
                if misc.get_rank() == 0 and not self.eval_mode:
                    self.logger.info('%s loaded from %s.json' % (name, filename))
        
        return target