from abc import ABC

from core.log import *
from core.checkpoint import *


class BaseAgent(ABC):
    """ Restore & save """
    def restore(self):
        """ Restore the latest parameter recorded by ckpt_manager

        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
            ckpt: An instance of tf.train.Checkpoint
            ckpt_path: The directory in which to write checkpoints
            name: optional name for print
        """
        restore(self._ckpt_manager, self._ckpt, self._ckpt_path, self._model_name)
        self.env_step = self._env_step.numpy()
        self.train_step = self._train_step.numpy()

    def save(self, print_terminal_info=False):
        """ Save Model
        
        Args:
            ckpt_manager: An instance of tf.train.CheckpointManager
        """
        self._env_step.assign(self.env_step)
        self._train_step.assign(self.train_step)
        save(self._ckpt_manager, print_terminal_info=print_terminal_info)

    """ Logging """
    def save_config(self, config):
        save_config(self._root_dir, self._model_name, config)

    def log(self, step, prefix=None, print_terminal_info=True):
        prefix = prefix or self.name
        log(self._logger, self._writer, self._model_name, prefix=prefix, 
            step=step, print_terminal_info=print_terminal_info)

    def log_stats(self, stats, print_terminal_info=True):
        log_stats(self._logger, stats, print_terminal_info=print_terminal_info)

    def set_summary_step(self, step):
        set_summary_step(step)

    def scalar_summary(self, stats, prefix=None, step=None):
        prefix = prefix or self.name
        scalar_summary(self._writer, stats, prefix=prefix, step=step)

    def histogram_summary(self, stats, prefix=None, step=None):
        prefix = prefix or self.name
        histogram_summary(self._writer, stats, prefix=prefix, step=step)

    def graph_summary(self, sum_type, *args, step=None):
        """
        Args:
            sum_type str: either "video" or "image"
            args: Args passed to summary function defined in utility.graph,
                of which the first must be a str to specify the tag in Tensorboard
            
        """
        assert isinstance(args[0], str), f'args[0] is expected to be a name string, but got "{args[0]}"'
        args = list(args)
        args[0] = f'{self.name}/{args[0]}'
        graph_summary(self._writer, sum_type, args, step=step)

    def store(self, **kwargs):
        store(self._logger, **kwargs)

    def get_raw_item(self, key):
        return get_raw_item(self._logger, key)

    def get_item(self, key, mean=True, std=False, min=False, max=False):
        return get_item(self._logger, key, mean=mean, std=std, min=min, max=max)

    def get_stats(self, mean=True, std=False, min=False, max=False):
        return get_stats(self._logger, mean=mean, std=std, min=min, max=max)

    def print_construction_complete(self):
        pwc(f'{self._model_name.upper()} is constructed...', color='cyan')
