import os
import util
import datetime
import shutil
import mlconfig
import torch
import json
import misc
from collections import OrderedDict
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


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
        if isinstance(target, torch.nn.DataParallel):
            target = target.module
        filename = os.path.join(self.checkpoint_path, name) + '.pt'
        torch.save(target.state_dict(), filename)
        if misc.get_rank() == 0:
            self.logger.info('%s saved at %s' % (name, filename))
        return

    def load_state(self, target, name, strict=True):
        filename = os.path.join(self.checkpoint_path, name) + '.pt'
        d = torch.load(filename, map_location=device)
        keys = []
        for k, v in d.items():
            if 'total_ops' in k or 'total_params' in k:
                keys.append(k)
        for k in keys:
            del d[k]
        target.load_state_dict(d)
        if misc.get_rank() == 0 and not self.eval_mode:
            self.logger.info('%s loaded from %s' % (name, filename))
        return target
