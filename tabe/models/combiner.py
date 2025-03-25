# NOTE
# Which one would be better? 
# 1) Basemodels are not trained once it they have been trained in the first training (and validation) process.
# 2) Basemodels are trained continuously as a new input comes in.


import time
import numpy as np
import pandas as pd
import torch

from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK
from sklearn.metrics import mean_absolute_error as MAE

from utils.tools import adjust_learning_rate, visual
from utils.metrics import metric

from tabe.data_provider.dataset_loader import get_data_provider
from tabe.models.abstractmodel import AbstractModel
import tabe.utils.report as report
from tabe.utils.mem_util import MemUtil
from tabe.utils.logger import logger
from tabe.utils.misc_util import OptimTracker
import tabe.utils.weighting as weighting


import warnings
warnings.filterwarnings('ignore')


_mem_util = MemUtil(rss_mem=False, python_mem=False)


class CombinerModel(AbstractModel):

    # Maximum lookback-window size for computing weights of base models
    MIN_LOOKBACK_WIN = 1
    MAX_LOOKBACK_WIN = 15 

    # Period for HPO (HyperParameter Optimization)
    # The most recent period of this length is used for HPO. 
    # For the first HPO, the ensemble-training period can be used. 
    # But, to provide continuous adaptive HPO feature, 
    # the latest input should be used (or added) for HPO. 
    # And, ever-growing input size is not practically acceptable. 
    # Thus, we use fixed-size evaluation period for HPO.
    # |<-  MAX_LOOKBACK_WIN   ->|<-     HPO_EVAL_PEROID      ->|
    # [0,1,..              ,t-1][t,t+1,... ,t+hpo_eval_period-1]
    HPO_EVAL_PEROID = 15 # Probably, the more, the better. But, too mush time cost. 
    HPO_PERIOD = MAX_LOOKBACK_WIN + HPO_EVAL_PEROID


    def __init__(self, configs, basemodels):
        super().__init__(configs, "Combiner")
        self.basemodels = basemodels
        self.basemodel_weights = None # lastly applied weights for basemodels
        self.basemodel_losses = None # the last basemodel_losses. Shape = (len(basemodels), HPO_EVALUATION_PEROID)
        self.truths = None # the last 'y' values. Shape = (HPO_EVALUATION_PEROID)
        self.hpo_policy = self.configs.hpo_policy
        if self.hpo_policy == 0 : # No HPO
            self.hp_dict = {
                'lookback_window':self.configs.lookback_win, # for weighting 
                'discount_factor':self.configs.discount_factor,
                'avg_method':self.configs.avg_method,
                'weighting_method':self.configs.weighting_method,
                'scaling_factor':self.configs.scaling_factor,
                'smoothing_factor':self.configs.smoothing_factor, 
                'max_models': self.configs.max_models
                }
        else:
            self.hp_space = {
                # 'cool_start': hp.quniform('cool_start', 0, num_comps-1, 1),
                'lookback_window': hp.quniform('lookback_window', self.MIN_LOOKBACK_WIN, self.MAX_LOOKBACK_WIN, 1),
                'discount_factor':hp.uniform('discount_factor', 1.0, 2.0),
                'avg_method': hp.choice('avg_method', [weighting.AvgMethod.MEAN, weighting.AvgMethod.MEAN_SQUARED]),
                'weighting_method': hp.choice('weighting_method', 
                                        [weighting.WeightingMethod.INVERTED, weighting.WeightingMethod.SQUARED_INVERTED, weighting.WeightingMethod.SOFTMAX]),
                'scaling_factor':hp.choice('scaling_factor', [10, 30, 50, 100]),
                'smoothing_factor': hp.uniform('smoothing_factor', 0.0, 0.2),
                'max_models': hp.quniform('max_models', 1, len(basemodels), 1),
            }
            self.hp_dict = None
            self.hpo_counter = 0 # used for counting timesteps for Adaptive HPO


    def compute_basemodel_weights(self, hp_dict, model_losses, prev_model_weights):        
        lookback_window = int(hp_dict['lookback_window']) if 'lookback_window' in hp_dict else self.configs.lookback_window
        discount_factor = hp_dict['discount_factor'] if 'discount_factor' in hp_dict else self.configs.discount_factor
        avg_method = hp_dict['avg_method'] if 'avg_method' in hp_dict else self.configs.avg_method
        weighting_method = hp_dict['weighting_method'] if 'weighting_method' in hp_dict else self.configs.weighting_method
        scaling_factor = hp_dict['scaling_factor'] if 'scaling_factor' in hp_dict else self.configs.scaling_factor
        smoothing_factor = hp_dict['smoothing_factor'] if 'smoothing_factor' in hp_dict else self.configs.smoothing_factor

        num_all_models = model_losses.shape[0]
        if 'max_models' in hp_dict: 
            max_models = min(int(hp_dict['max_models']), num_all_models)
        else:
            max_models = num_all_models

        basemodel_weights = weighting.compute_model_weights(model_losses, prev_model_weights, 
                                    lookback_window=lookback_window, 
                                     discount_factor=discount_factor,
                                    avg_method=avg_method, 
                                    weighting_method=weighting_method,
                                    softmax_scaling_factor=scaling_factor, 
                                    smoothing_factor=smoothing_factor, 
                                    max_models=max_models)

        return basemodel_weights


    def _optimize_HP(self, use_BOA=True, max_evals=100):

        # Objective function (loss function) for hyper-parameter optimization
        # Loss == Mean of the lossees in all timesteps in the period [lookback_window, len(y)]
        def _evaluate_hp(hp_dict):
            lookback_window = int(hp_dict['lookback_window'])
            losses = []
            basemodel_weights = None
            t = self.MAX_LOOKBACK_WIN # We should start from the same timestep to evaluate the same period.
            while t < len(self.truths):
                basemodel_weights = self.compute_basemodel_weights(
                    hp_dict, self.basemodel_losses[:, t-lookback_window : t], basemodel_weights)
                next_y_hat = np.dot(basemodel_weights, self.basemodel_losses[:, t:t+1])
                next_y = self.truths[t]
                losses.append(np.abs(next_y_hat - next_y))
                t += 1
            mean_loss = np.mean(losses)
            var_loss = np.var(losses)
            self.optim_tracker(mean_loss, hp_dict)

            return {
                'loss': mean_loss,         
                'loss_variance': var_loss, 
                'status': STATUS_OK
            }

        time_now = time.time()

        trials = Trials()
        algo = tpe.suggest if use_BOA else rand.suggest
        self.optim_tracker = OptimTracker(use_early_stop=False, verbose=True, save_to_file=False)
        self.best_hp = fmin(_evaluate_hp, self.hp_space, algo=algo, max_evals=max_evals, 
                    trials=trials, rstate=np.random.default_rng(1), verbose=True)
            
        spent_time = (time.time() - time_now) 
        logger.info(f'Combiner._optimize_HP() : {spent_time:.4f}sec elapsed. min_loss={self.optim_tracker.val_loss_min}')
        report.print_dict(self.best_hp, '[ Combiner HP ]')

        return self.best_hp, trials


    def train_basemodels(self):
        if self.configs.is_training:
            logger.info('Training base models ==================================\n')
            for basemodel in self.basemodels:
                # TODO : Better to optimize the the reduncancy of the same proceudures in training base models. 
                logger.info(f'Training {basemodel.name} ...')
                basemodel.train()
        else:
            logger.info('Loading trained base models ======================\n')
            for basemodel in self.basemodels:
                basemodel.load_saved_model()


    def train(self, train_dataset=None, train_loader=None):
        if train_dataset is None :
            train_dataset, train_loader = get_data_provider(self.configs, flag='ensemble_train', step_by_step=True)

        time_now = time.time()

        basemodel_preds = np.empty((len(self.basemodels), len(train_loader)))
        basemodel_losses = np.empty((len(self.basemodels), len(train_loader)))

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            for m, basemodel in enumerate(self.basemodels):
                basemodel_losses[m, t], basemodel_preds[m, t] = basemodel.proceed_onestep(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, 
                    training=True) # NOTE : Base models ARE trained in ensemble_training period.
            _mem_util.print_memory_usage()

        spent_time = (time.time() - time_now) 
        logger.info(f"CombinerModel.train() : {spent_time:.4f} sec elapsed for getting base models' predictions")

        if self.hpo_policy == 0: # No HPO
            lookback_window = int(self.hp_dict['lookback_window'])
            assert basemodel_losses.shape[1] >= lookback_window, \
                    f'basemodel_losses.shape[1]) {basemodel_losses.shape[1]} is shorter than lookback_window ({lookback_window})'
            self.basemodel_losses = basemodel_losses[:, -lookback_window:]
            self.truths = train_dataset.data_y[-lookback_window:, -1]
        else:
            assert self.HPO_PERIOD <= len(train_loader), \
                    f'length of train data ({len(train_loader)}) should be longer than HPO_PERIOD({self.HPO_PERIOD})'     
            self.basemodel_losses = basemodel_losses[:, -self.HPO_PERIOD:]
            self.truths = train_dataset.data_y[-self.HPO_PERIOD:, -1]
            self.hp_dict, trials = self._optimize_HP(max_evals=self.configs.max_hpo_eval)
            report.plot_hpo_result(trials, "HyperParameter Optimization for Combiner",
                                self._get_result_path()+"/hpo_result.pdf")
        

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = True):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        basemodel_preds = np.empty((len(self.basemodels)))
        basemodel_losses = np.empty((len(self.basemodels), 1))

        for m, basemodel in enumerate(self.basemodels):
            basemodel_preds[m], basemodel_losses[m, 0] = basemodel.proceed_onestep(
                batch_x, batch_y, batch_x_mark, batch_y_mark, training) 

        basemodel_losses = np.concatenate((self.basemodel_losses, basemodel_losses), axis=1)
        lookback_window = int(self.hp_dict['lookback_window'])
        basemodel_weights = self.compute_basemodel_weights(
            self.hp_dict, basemodel_losses[:, -lookback_window:], self.basemodel_weights)
        self.basemodel_weights = basemodel_weights

        # Adaptive HPO 
        if self.configs.hpo_policy == 2:
            assert self.HPO_PERIOD <= basemodel_losses.shape[1], \
                        f'length of basemodel data ({basemodel_losses.shape[1]}) should be longer than HPO_PERIOD({self.HPO_PERIOD})'            
            self.basemodel_losses = basemodel_losses[:, -self.HPO_PERIOD:]
            self.truths = np.concatenate((self.truths, batch_y[-1:, -1, -1]))[-self.HPO_PERIOD:]
            self.hpo_counter += 1
            if self.hpo_counter == self.configs.hpo_interval:
                self.hp_dict, _ = self._optimize_HP(max_evals=self.configs.max_hpo_eval)
                self.hpo_counter = 0                

        y_hat = np.dot(basemodel_weights, basemodel_preds)
        return y_hat, basemodel_preds


    # 
    # NOTE:
    # In test() func, 'Adaptive HPO' is not applied. 
    #
    def test(self):
        time_now = time.time()

        test_data, test_loader = get_data_provider(self.configs, flag='test', step_by_step=True)
        y = test_data.data_y[self.configs.seq_len:, -1]
        need_to_invert_data = True if (test_data.scale and self.configs.inverse) else False

        # prepare the forecasted values of base models in test period
        basemodel_preds = np.empty((len(self.basemodels), len(test_loader)))
        basemodel_losses = np.empty((len(self.basemodels), len(test_loader)))
        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for m, basemodel in enumerate(self.basemodels):
                basemodel_preds[m, t] = basemodel.proceed_onestep(
                    batch_x, batch_y, batch_x_mark, batch_y_mark, training=True)      
                
        # compute CombinerModel's predictions
        y_hat = np.empty_like(y)
        weights_hist = np.empty((len(y), len(self.basemodels)))

        # concatenate the last vali losses to test losses. 
        basemodel_losses = np.concatenate((self.basemodel_losses, basemodel_losses), axis=1)

        lookback_window = int(self.hp_dict['lookback_window'])
        basemodel_weights = None
        for t in range(len(y)):
            basemodel_weights = self.compute_basemodel_weights(
                self.hp_dict, basemodel_losses[:, -lookback_window:], basemodel_weights)
            y_hat[t] = np.dot(basemodel_weights, basemodel_preds[:, t:t+1])
            weights_hist[t] = basemodel_weights

        spent_time = (time.time() - time_now) 
        logger.info(f'CombinerModel.test() : {spent_time:.4f}sec elapsed for testing')

        report.plot_weights(weights_hist, "Base Model Weights",
                           self._get_result_path() + "/basemodel_weights.pdf")

        if need_to_invert_data:
            n_features = test_data.data_y.shape[1]
            data_y = np.zeros((len(y), n_features))
            data_y_hat = np.zeros((len(y), n_features))
            data_y[:, -1] = y
            data_y_hat[:, -1] = y_hat
            y = test_data.inverse_transform(data_y)[:, -1]
            y_hat = test_data.inverse_transform(data_y_hat)[:, -1]

        losses = self.criterion(torch.tensor(y_hat), y)

        logger.info(f"CombinerModel.test() : Loss ----- ")
        logger.info(f"max={np.max(losses):.6f}, mean={np.mean(losses):.6f}, min={np.min(losses):.6f}, var={np.var(losses):.6f})")

        report.plot_forecast(y, y_hat, "Combiner Forecast", 
                            self._get_result_path() + "/combiner_forecast.pdf")

        return y, y_hat, losses

