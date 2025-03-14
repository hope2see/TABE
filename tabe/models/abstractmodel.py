import os
import torch.nn as nn
from tabe.utils.misc_util import experiment_sig, get_loss_func


class AbstractModel(object):
    def __init__(self, configs, name):
        self.configs = configs
        self.name = name
        self.criterion = get_loss_func(self.configs.loss)

    def _get_checkpoint_path(self):
        path = os.path.join(self.configs.checkpoints, experiment_sig()) 
        path = os.path.join(path, self.name) 
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _get_result_path(self):
        path = os.path.join("./result", experiment_sig()) 
        path = os.path.join(path, self.name) 
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load_saved_model(self):
        pass

    # @abstractmethod
    def train(self):
        raise NotImplementedError

    # @abstractmethod
    def test(self):
        raise NotImplementedError

    # @abstractmethod
    def proceed_onestep(self, training: bool = False):
        raise NotImplementedError

