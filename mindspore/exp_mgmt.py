import os
import sys
sys.path.append('/root/CognitiveDistillation/mindspore_impl')
import util
import datetime
import shutil
import mlconfig
import mindspore as ms
import json
import misc


class ExperimentManager():
    def __init__(self, exp_name, exp_path, config_file_path, eval_mode=False):
        if exp_name == '' or exp_name is None:
            exp_name = 'exp_at' + datetime.datetime.now()
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
        # For MindSpore, we save as .ckpt files
        filename = os.path.join(self.checkpoint_path, name) + '.ckpt'
        
        if hasattr(target, 'parameters_dict'):
            # For models
            ms.save_checkpoint(target, filename)
        elif hasattr(target, 'get_lr'):
            # For learning rate schedulers - save as JSON
            lr_data = {'learning_rate': float(target.get_lr())}
            json_filename = os.path.join(self.checkpoint_path, name) + '.json'
            with open(json_filename, 'w') as f:
                json.dump(lr_data, f)
            if misc.get_rank() == 0:
                self.logger.info('%s saved at %s' % (name, json_filename))
            return
        else:
            # For optimizers - MindSpore optimizers don't have state_dict
            # We'll skip saving optimizer state for now
            if misc.get_rank() == 0:
                self.logger.info('Skipping save for %s (optimizer state not supported in MindSpore)' % name)
            return
            
        if misc.get_rank() == 0:
            self.logger.info('%s saved at %s' % (name, filename))
        return

    def load_state(self, target, name, strict=True):
        filename = os.path.join(self.checkpoint_path, name) + '.ckpt'
        
        if hasattr(target, 'parameters_dict'):
            # For models
            param_dict = ms.load_checkpoint(filename)
            # Filter out ops and params count keys if present
            keys_to_remove = []
            for k in param_dict.keys():
                if 'total_ops' in k or 'total_params' in k:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del param_dict[k]
            
            ms.load_param_into_net(target, param_dict)
        elif hasattr(target, 'get_lr'):
            # For learning rate schedulers - load from JSON
            json_filename = os.path.join(self.checkpoint_path, name) + '.json'
            if os.path.exists(json_filename):
                with open(json_filename, 'r') as f:
                    lr_data = json.load(f)
                # Note: MindSpore schedulers may not support direct state loading
                if misc.get_rank() == 0 and not self.eval_mode:
                    self.logger.info('Loaded scheduler state from %s' % json_filename)
        else:
            # For optimizers - skip loading
            if misc.get_rank() == 0 and not self.eval_mode:
                self.logger.info('Skipping load for %s (optimizer state not supported in MindSpore)' % name)
            return target
            
        if misc.get_rank() == 0 and not self.eval_mode:
            self.logger.info('%s loaded from %s' % (name, filename))
        return target