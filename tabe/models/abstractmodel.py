import os
import numpy as np
import torch 
from tabe.utils.misc_util import experiment_sig, get_loss_func


class AbstractModel(object):

    # static variable shared by all child instances. 
    # __current_phase = 0  # 0: base_train(including val),  1: ensemble_train,  2: test 
    __max_saved_result = 1000


    # @staticmethod
    # def notify_toal_test_steps(total_test_steps: int):
    #     """
    #     total_test_steps is not given when test use 'LIVE' data. 
    #     """
    #     AbstractModel.__total_test_steps = total_test_steps


    def __init__(self, configs, device, name):
        self.configs = configs
        self.device = device
        self.name = name
        self.criterion = get_loss_func(self.configs.loss)
        self.prev_prediction = None
        self.need_to_store_result = False
        self.predictions = None
        self.num_results = 0


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


    def train(self):
        raise NotImplementedError


    def test(self):
        raise NotImplementedError


    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        raise NotImplementedError


    def _in_first_step(self):
        return self.prev_prediction is None


    def _update_prev_prediction(self, cur_prediction):
        prev_pred = self.prev_prediction
        self.prev_prediction = cur_prediction
        return prev_pred


    def _store_result(self, pred, loss):
        # if self.num_results == 0:
        #     max_stored_results = AbstractModel.__max_saved_result if AbstractModel.__total_test_steps is None \
        #                         else AbstractModel.__total_test_steps
        #     self.predictions = np.full((max_stored_results), np.nan, dtype=np.float32)
        #     # self.losses = np.full((max_saved_results), np.nan, dtype=np.float32)

        if self.num_results >= len(self.predictions):
            # TODO : dump the existing results, and start saving
            assert False, "Num reults exceeds the limit of saving!"
        else:
            self.predictions[self.num_results] = pred 
            # self.losses[self.num_results] = loss
            self.num_results += 1


    def prepare_result_storage(self, total_steps=None):
        max_stored_results = AbstractModel.__max_saved_result if total_steps is None else total_steps
        self.predictions = np.full((max_stored_results), np.nan, dtype=np.float32)        
        self.num_results = 0
        self.need_to_store_result = True


    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """
        batch_y.shape : (batch_size, label_len, num_features)

        Returns the prediction at the current timestep, and the loss of prediction made at the previous timestep.
        """
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1
        # assert batch_y.shape[1] == self.configs.label_len
        # We assume that batch_y has true values of the previous prediction. 
        # This assumption is satisfied only in case that pred_len == label_len, 
        # and proceed_onestep() is invoked sequentially step by step.
        
        cur_pred = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)

        prev_truth = batch_y[0, -1, -1]
        prev_pred = self._update_prev_prediction(cur_pred)
        if prev_pred is None:
            prev_loss = None
        else:
            prev_loss = self.criterion(torch.tensor(prev_pred), prev_truth).item()

        if self.need_to_store_result:
            self._store_result(cur_pred, prev_loss)

        return cur_pred, prev_loss
