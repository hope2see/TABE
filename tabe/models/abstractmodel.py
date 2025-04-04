import os
import numpy as np
import torch
from tabe.utils.losses import get_loss_func
from tabe.utils.distributions import get_quantile_value, get_cumul_prob
from tabe.data_provider.dataset_tabe import Dataset_TABE


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


    def __init__(self, configs, name):
        self.configs = configs
        self.device = configs.device
        self.name = name
        self.criterion = get_loss_func(self.configs.loss)

        # NOTE : 
        # batch_y (which has 'label' or 'ground truth') is provided 1 timestep behind batch_x (feature data).
        # In order to compute a 'deviation' for a prediction, we have to wait 1 timestep to get the label. 
        # So, the first prediction made in proceed_onestep() is saved temporarily in self.cur_prediction, 
        # and then all the actual results (predictions, deviations, dv_quantiles, etc) are computed and saved 
        # when the next proceed_onestep() is called with the batch_y (which has the label/truth) for the prediction. 
        self.next_truth_pred = None # prediction for the next target 
        self.next_prob_ascending = None

        # store results for analysis ----------------
        self._result_storage_prepared = False
        self.num_results = 0
        self.predictions = None 
        self.deviations = None
        self.prob_ascendings = None
        self.dv_quantiles = None # deviation quantiles. shape=[num_results, 2] (left_quantile, right_quantile)
        # --------------------------------------------
        # cached actual results
        self._cached_result_predictions = None 
        self._cached_result_deviations = None 
        self._cached_result_prob_ascendings = None 
        self._cached_result_dv_quantiles = None 
        # --------------------------------------------


    def _get_checkpoint_path(self):
        path = os.path.join(self.configs.checkpoints, self.name) 
        if not os.path.exists(path):
            os.makedirs(path)
        return path


    def _get_result_path(self):
        path = os.path.join(self.configs.result_dir, self.name) 
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
        return self.next_truth_pred is None


    # def _compute_result_and_store(self, truth, truth_pred, prob_ascending, next_truth_pred):
    #     assert self._result_storage_prepared

    #     if self.num_results >= len(self.predictions):
    #         # TODO : dump the existing results, and start saving
    #         assert False, "Num reults exceeds the limit of saving!"

    #     next_prob_ascending = None
    #     deviation = truth - truth_pred
    #     self.predictions[self.num_results] = truth_pred 
    #     self.deviations[self.num_results] = deviation
    #     self.prob_ascendings[self.num_results] = prob_ascending

    #     if self.num_results < self.configs.prob_stat_win: 
    #         self.dv_quantiles[self.num_results, :] = (np.nan, np.nan)
    #     else:
    #         self.dv_quantiles[self.num_results, :] = get_quantile_value(\
    #             self.deviations[self.num_results-self.configs.prob_stat_win : self.num_results], self.configs.quantile)

    #         # the probability of ascending ==  the probability that next_truth_pred's deviation will be more (or equal to) next_truth_pred. 
    #         deviation_to_be_equal = next_truth_pred - truth
    #         next_prob_ascending = get_cumul_prob(self.deviations[self.num_results-self.configs.prob_stat_win : self.num_results], \
    #                             deviation_to_be_equal, left_tail=False)

    #     self.num_results += 1

    #     # return the probability of next target being ascending
    #     return next_prob_ascending


    def _compute_probability_of_ascending(self, cur_truth):
        """
        return the probability of next target being higher (or equal to) the current target.
        """
        # 'flat' means next_truth is same as prev_truth. So,
        # the probability of ascending
        #  ==  the probability that next_truth_pred's deviation will be more (or equal to) next_truth_pred. 
        deviation_to_be_equal = self.next_truth_pred - cur_truth
        return get_cumul_prob(self.deviations[self.num_results-self.configs.prob_stat_win : self.num_results], \
                              deviation_to_be_equal, left_tail=False)



    def prepare_result_storage(self, total_steps=None):
        result_storage_size = AbstractModel.__max_saved_result if total_steps is None else total_steps-1
        self.predictions = np.full((result_storage_size), np.nan, dtype=np.float32)   
        self.deviations = np.full((result_storage_size), np.nan, dtype=np.float32)   
        self.prob_ascendings = np.full((result_storage_size), np.nan, dtype=np.float32)   
        self.dv_quantiles = np.full((result_storage_size, 2), np.nan, dtype=np.float32)
        self.num_results = 0
        self._result_storage_prepared = True


    def result_predictions(self):
        actual_num_results = self.num_results - self.configs.warm_up_length
        if (self._cached_result_predictions is None) or (len(self._cached_result_predictions) < actual_num_results):
            self._cached_result_predictions = self.predictions[-actual_num_results:]
        return self._cached_result_predictions


    def result_deviations(self):
        actual_num_results = self.num_results - self.configs.warm_up_length
        if (self._cached_result_deviations is None) or (len(self._cached_result_deviations) < actual_num_results):
            self._cached_result_deviations = self.deviations[-actual_num_results:]
        return self._cached_result_deviations


    def result_prob_ascendings(self):
        actual_num_results = self.num_results - self.configs.warm_up_length
        if (self._cached_result_prob_ascendings is None) or (len(self._cached_result_prob_ascendings) < actual_num_results):
            self._cached_result_prob_ascendings = self.prob_ascendings[-actual_num_results:]
        return self._cached_result_prob_ascendings


    def result_dv_quantiles(self):
        actual_num_results = self.num_results - self.configs.warm_up_length
        if (self._cached_result_dv_quantiles is None) or (len(self._cached_result_dv_quantiles) < actual_num_results):
            self._cached_result_dv_quantiles = self.dv_quantiles[-actual_num_results:]
        return self._cached_result_dv_quantiles


    def invert_result(self, dataset: Dataset_TABE):
        assert self._result_storage_prepared 
        self.predictions = dataset.inverse_transform(self.predictions)
        self.deviations = dataset.get_labels() - self.predictions
        self.dv_quantiles[:, 0] = dataset.inverse_transform(self.dv_quantiles[:, 0])
        self.dv_quantiles[:, 1] = dataset.inverse_transform(self.dv_quantiles[:, 1])


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

        assert self._result_storage_prepared

        truth = batch_y[0, -1, -1]
        truth_pred = self.next_truth_pred
        loss = None if truth_pred is None else self.criterion(torch.tensor(truth_pred), truth).item()

        next_truth_pred = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)        
        next_prob_ascending = None

        if truth_pred is not None:
            prob_ascending = self.next_prob_ascending            

            next_prob_ascending = None
            deviation = truth - truth_pred

            if self.num_results >= len(self.predictions): 
                # TODO : dump the existing results, and start saving
                assert False, "Num reults exceeds the result storage size!"
            
            self.predictions[self.num_results] = truth_pred 
            self.deviations[self.num_results] = deviation
            self.prob_ascendings[self.num_results] = prob_ascending

            if self.num_results+1 < self.configs.prob_stat_win: 
                self.dv_quantiles[self.num_results, :] = (np.nan, np.nan)
            else:
                self.dv_quantiles[self.num_results, :] = get_quantile_value(\
                    self.deviations[self.num_results+1-self.configs.prob_stat_win : self.num_results+1], self.configs.quantile)

                # the probability of ascending ==  the probability that next_truth_pred's deviation will be more (or equal to) next_truth_pred. 
                deviation_to_be_equal = next_truth_pred - truth
                next_prob_ascending = get_cumul_prob(self.deviations[self.num_results+1-self.configs.prob_stat_win : self.num_results+1], \
                                    deviation_to_be_equal, left_tail=False)
            self.num_results += 1

        self.next_truth_pred = next_truth_pred
        self.next_prob_ascending = next_prob_ascending
        
        return next_truth_pred, loss, next_prob_ascending
