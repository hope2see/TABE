
import os
import io 
import time
import numpy as np
import torch

from scipy.stats import norm

import pyro
import pyro.contrib.gp as gp

from hyperopt import hp, tpe, rand, fmin, Trials, STATUS_OK

# from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE

from tabe.models.abstractmodel import AbstractModel
from tabe.models.timemoe import TimeMoE
from tabe.utils.mem_util import MemUtil
from tabe.utils.misc_util import logger, OptimTracker
import tabe.utils.weighting as weighting
import tabe.utils.report as report

smoke_test = "CI" in os.environ  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.9.1')
pyro.set_rng_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)


_mem_util = MemUtil(rss_mem=False, python_mem=False)


class AdjusterModel(AbstractModel):

    def __init__(self, configs, name='Adjuster'):
        super().__init__(configs, name)
        self.y_hat = None
        self.y_hat_cbm = None # the last predictions of combiner. shape = (HPO_EVALUATION_PEROID)
        self.truths = None # the last 'y' values. shape = (HPO_EVALUATION_PEROID)
        self.credibility = 0.5 # relative credibility against combiner model. Initially neutral value 0.5
    

    def _predict_next(self, truth=None, y_hat=None, y_hat_cbm=None, user_param=None):
        LOOKBACK_WIN_FOR_SD = 30 # lookback window size for computing std_dev of deviations 

        # truths[-1] and y_hat_cbm[-1] must have the lastest 'past' data, not next timestep's data.
        cbm_deviations = truth - y_hat_cbm 

        exp_deviation = self.do_predict_next(cbm_deviations, user_param)

        my_deviations = truth - y_hat
        stddev_window = min(LOOKBACK_WIN_FOR_SD, len(my_deviations))
        sd_adj = my_deviations[-stddev_window:].std() if stddev_window > 1 else 0.0
        sd_cbm = cbm_deviations[-stddev_window:].std() if stddev_window > 1 else 0.0

        logger.info(f"Adjuster._predict_next(): exp_dev={exp_deviation.item():.4f}, sd_adj={sd_adj.item():.4f}, sd_cbm={sd_cbm.item():.4f}")
        return exp_deviation, sd_adj


    def do_train(self, y, y_hat_cbm):
        raise NotImplementedError


    def do_predict_next(self, cbm_deviations, user_param):
        raise NotImplementedError


    def train(self, y, y_hat_cbm):
        assert len(y) == len(y_hat_cbm)

        y_hat = self.do_train(y, y_hat_cbm)

        self.truths = y
        self.y_hat = y_hat
        self.y_hat_cbm = y_hat_cbm
        return y_hat


    def proceed_onestep(self, y, y_hat_cbm, training: bool = False):
        # assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        # estimate the next deviation with the last deviation 
        pred_deviation, devi_stddev = self._predict_next(self.truths, self.y_hat, self.y_hat_cbm)

        # get credibility (or weight over combiner) of Adjuster model 
        eval_period = min(len(self.y_hat), self.configs.lookback_win)
        if eval_period > 0:
            my_loss = np.abs(self.y_hat[-eval_period:] - self.truths[-eval_period:])   
            cbm_loss = np.abs(self.y_hat_cbm[-eval_period:] - self.truths[-eval_period:])
            model_losses = np.array([my_loss, cbm_loss])
            prev_weights = np.array([self.credibility, 1.0-self.credibility])
            weights = weighting.compute_model_weights(model_losses, prev_weights, 
                                    lookback_window=eval_period, 
                                    discount_factor=self.configs.discount_factor, 
                                    avg_method=self.configs.avg_method, 
                                    weighting_method=self.configs.weighting_method,
                                    softmax_scaling_factor=self.configs.scaling_factor, 
                                    smoothing_factor=self.configs.smoothing_factor)
            self.credibility = weights[0]

            logger.debug("Adj.predict : Adj Losses : " + "[" + ", ".join(f'{l:.5f}' for l in my_loss) + "]")
            logger.debug("Adj.predict : Cbm Losses : " + "[" + ", ".join(f'{l:.5f}' for l in cbm_loss) + "]")
            logger.info(f'Adj.predict : adjuster_credibility = {self.credibility:.5f}')

        y_hat = y_hat_cbm + pred_deviation.item()
                     
        # adjust combinerModel's prediction by adding expected deviation 
        y_hat = y_hat_cbm + (pred_deviation.item() * self.credibility)

        self.y_hat_cbm = np.concatenate((self.y_hat_cbm, np.array([y_hat_cbm])))
        self.y_hat = np.concatenate((self.y_hat, np.array([y_hat])))
        self.truths = np.concatenate((self.truths, np.array([y])))

        # logger.info(f'Adj.predict : final_pred={final_pred:.5f}, y_hat={y_hat:.5f}, y_hat_cbm={y_hat_cbm:.5f}')
        return y_hat


class GpmAdjuster(AdjusterModel):

    # lookback-window size for fitting gaussian process model
    # MIN_LOOKBACK_WIN = 10
    # MAX_LOOKBACK_WIN = 80
    UNLIMITED_LOOKBACK_WIN = -1

    # Period for HPO (HyperParameter Optimization)
    # The most recent period of this length is used for HPO. 
    # For the first HPO, the ensemble-training period can be used. 
    # But, to provide continuous adaptive HPO feature, 
    # the latest input should be used (or added) for HPO. 
    # And, ever-growing input size is not practically acceptable. 
    # Thus, we use fixed-size evaluation period for HPO.
    #
    # |<-  MAX_LOOKBACK_WIN   ->|<-     HPO_EVAL_PEROID      ->|
    # [0,1,..              ,t-1][t,t+1,... ,t+hpo_eval_period-1]
    HPO_EVAL_PEROID = 20 # Probably, the more, the better. But, too mush time cost. 

    def __init__(self, configs):
        super().__init__(configs, 'GpmAdjuster')
        self.gpm_noise = self.configs.gpm_noise
        self._set_gpm_kernel()
        self.gpm = None # Gaussain Process Model 
        self.hpo_policy = self.configs.hpo_policy
        if self.hpo_policy == 0 :
            self.hp_dict = {
                'gpm_lookback_win':self.configs.gpm_lookback_win,
                # 'lookback_window':self.configs.lookback_win, # for weighting 
                # 'discount_factor':self.configs.discount_factor,
                # 'avg_method':self.configs.avg_method,
                # 'weighting_method':self.configs.weighting_method,
                # 'scaling_factor':self.configs.scaling_factor,
                # 'smoothing_factor':self.configs.smoothing_factor
                }
        else:
            self.hp_space = {
                'gpm_lookback_win': hp.choice('gpm_lookback_win', [5, 10, 30, 50, 100, self.UNLIMITED_LOOKBACK_WIN]), 
                # 'lookback_window': hp.quniform('lookback_window', 1, 10, 1),
                # 'discount_factor':hp.uniform('discount_factor', 1.0, 2.0),
                # 'avg_method': hp.choice('avg_method', [weighting.AvgMethod.MEAN, weighting.AvgMethod.MEAN_SQUARED]),
                # 'weighting_method': hp.choice('weighting_method', 
                #                         [weighting.WeightingMethod.INVERTED, weighting.WeightingMethod.SQUARED_INVERTED, weighting.WeightingMethod.SOFTMAX]),
                # 'scaling_factor':hp.choice('scaling_factor', [10, 30, 50, 100]),
                # 'smoothing_factor': hp.uniform('smoothing_factor', 0.0, 0.2),
            }
            self.hp_dict = None
            self.hpo_counter = 0 # used for counting timesteps for Adaptive HPO


    def _set_gpm_kernel(self):
        kernels = {
            # name      : (class, need_lengthscale)
            'RBF'       : (gp.kernels.RBF, True),
            'Matern32'  : (gp.kernels.Matern32, True),
            'Matern52'  : (gp.kernels.Matern52, True),
            'Linear'    : (gp.kernels.Linear, False),
            'Brownian'  : (gp.kernels.Brownian, False),
        }
        input_dim = 1
        # TODO ; optimize the HPs of GP
        variance = torch.tensor(1) 
        lengthscale = torch.tensor(1.5)
        kernel_name = self.configs.gpm_kernel
        kernel_class = kernels[kernel_name][0]
        need_lengthscale = kernels[kernel_name][1]
        if need_lengthscale:
            self.gpm_kernel = kernel_class(input_dim=input_dim, variance=variance, lengthscale=lengthscale)
        else:
            self.gpm_kernel = kernel_class(input_dim=input_dim, variance=variance)


    def _train_new_gpmodel(self, hp_dict, y):
        pyro.clear_param_store() # NOTE : Need to do everytime? 

        gpm_lookback_win = int(hp_dict['gpm_lookback_win'])
        if gpm_lookback_win != self.UNLIMITED_LOOKBACK_WIN and len(y) > gpm_lookback_win+1:
            y = y[-(gpm_lookback_win+1):]
        X = y[:-1]
        y = y[1:]
        gpm = gp.models.GPRegression(X, y, self.gpm_kernel, 
                                            noise=torch.tensor(self.gpm_noise), mean_function=None, jitter=1e-6)
        gpm.set_data(X, y)
        self.optimizer = torch.optim.Adam(gpm.parameters(), lr=0.005)
        self.loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        
        optim_tracker = OptimTracker(use_early_stop=True, patience=self.configs.max_gp_opt_patience, verbose=False, save_to_file=False)
        num_batch = 10
        for step in range(1, self.configs.max_gp_opt_steps, num_batch):
            loss = gp.util.train(gpm, self.optimizer, self.loss_fn, num_steps=num_batch)
            mean_loss = np.mean(loss).item() / len(y) # mean loss for one-step
            optim_tracker(mean_loss, None)
            if optim_tracker.early_stop:
                break
        logger.debug(f"Adj.HPO: when lb_win_size={gpm_lookback_win}, after training {step} times, loss={mean_loss:.4f}")
        
        return gpm
 

    def _forward_onestep(self, hp_dict, gpm, y):
        assert y.shape[0]==1, "Allowed to add only one observation.."
        pyro.clear_param_store() 

        # incorporate new observation(s)
        X = torch.cat([gpm.X, gpm.y[-1:]]) # Add the last y to the end of 'X's
        y = torch.cat([gpm.y, y]) # Add new observation to the end of 'y's

        gpm_lookback_win = int(hp_dict['gpm_lookback_win'])
        if gpm_lookback_win != self.UNLIMITED_LOOKBACK_WIN and len(y) > gpm_lookback_win+1:
            X = X[-gpm_lookback_win:]
            y = y[-gpm_lookback_win:]
        gpm.set_data(X, y)

        optim_tracker = OptimTracker(use_early_stop=True, patience=self.configs.max_gp_opt_patience, verbose=False, save_to_file=False)
        num_batch = 10
        for step in range(1, self.configs.max_gp_opt_steps, num_batch):
            loss = gp.util.train(gpm, self.optimizer, self.loss_fn, num_steps=num_batch)
            mean_loss = np.mean(loss).item() / len(y) # mean loss for one timestep
            optim_tracker(mean_loss, None)
            if optim_tracker.early_stop:
                break                
        logger.debug(f"Adj.HPO: when lb_win_size={gpm_lookback_win}, after training  {step} times, loss={mean_loss:.4f}")
        return gpm


    def _optimize_HP(self, 
                     search_alg=0, # 0 : ad-hoc, 1 : BOA, 2 : random
                     max_evals=10):
        
        # Objective function (loss function) for hyper-parameter optimization
        # Loss == Mean of the lossees in all timesteps in the period [gpm_lookback_win, len(y)]
        def _evaluate_hp(hp_dict):
            deviations = (self.truths - self.y_hat_cbm)[:-1] # exclude the last one, becuause it is the target to predict
            assert len(deviations) > self.HPO_EVAL_PEROID
            gpm = None            
            losses = []
            for t in range(len(deviations) - self.HPO_EVAL_PEROID, len(deviations)):
                if gpm is None:
                    gpm = self._train_new_gpmodel(hp_dict, torch.Tensor(deviations[:t]))
                else:
                    gpm = self._forward_onestep(hp_dict, gpm, torch.tensor([deviations[t]]))
                exp_deviation, _ = self._predict_next(self.truths, self.y_hat, self.y_hat_cbm, user_param=gpm)
                next_y_hat = self.y_hat_cbm[t] + exp_deviation
                next_y = self.truths[t]
                losses.append(np.abs(next_y_hat - next_y)) # MAE!  # Need to use the loss metric of configuration 
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

        self.optim_tracker = OptimTracker(use_early_stop=False, patience=self.configs.max_gp_opt_patience, verbose=True, save_to_file=False)
        trials = Trials()
        algo = tpe.suggest if search_alg == 1 else rand.suggest
        self.best_hp = fmin(_evaluate_hp, self.hp_space, algo=algo, max_evals=max_evals, 
                trials=trials, rstate=np.random.default_rng(1), verbose=True)

        spent_time = (time.time() - time_now) 

        logger.info(f'Adjuster._optimize_HP() : {spent_time:.4f}sec elapsed. min_loss={self.optim_tracker.val_loss_min:.5f}')            
        report.print_dict(self.best_hp, '[ Adjuster HP ]')
        return self.best_hp, trials


    def do_train(self, y, y_hat_cbm):
        assert self.HPO_EVAL_PEROID <= len(y), \
                    f'length of train data ({len(y)}) should be longer than HPO_EVAL_PEROID({self.HPO_EVAL_PEROID})'     

        if self.hpo_policy != 0: 
            self.hp_dict, trials = self._optimize_HP(search_alg=1, max_evals=self.configs.max_hpo_eval)
            if trials is not None:
                report.plot_hpo_result(trials, "HyperParameter Optimization for Adjuster",
                                self._get_result_path()+"/hpo_result.pdf")

        deviations = y - y_hat_cbm
        self.gpm = self._train_new_gpmodel(self.hp_dict, torch.Tensor(deviations))
        y_hat = np.copy(y_hat_cbm)
        return y_hat


    def do_predict_next(self, cbm_deviations, user_param):
        gpm = self.gpm if user_param is None else user_param  
        last_deviation = gpm.y[-1:] 
        with torch.no_grad():
            exp_deviation, cov = gpm(last_deviation, full_cov=False, noiseless=False)

        # NOTE 
        # It seems not good to estimate stddev estimation by GP
        # sd = cov.diag().sqrt()  
        # report.plot_gpmodel(gpm, filepath="./temp/gpmodel_analysis.png")

        return exp_deviation


    def proceed_onestep(self, y, y_hat_cbm, training: bool = False):
        y_hat = super().proceed_onestep(y, y_hat_cbm, training)

        if training:
            true_deviation = torch.tensor([y.item() - y_hat_cbm])
            self.gpm = self._forward_onestep(self.hp_dict, self.gpm, true_deviation)

        # Adaptive HPO 
        if self.configs.hpo_policy == 2: 
            self.hpo_counter += 1
            if self.hpo_counter == self.configs.hpo_interval:
                self.hp_dict, _ = self._optimize_HP(search_alg=1, max_evals=self.configs.max_hpo_eval)
                self.hpo_counter = 0                

        return y_hat


class TimeMoeAdjuster(AdjusterModel):

    def __init__(self, configs, device):
        super().__init__(configs, 'GpmAdjuster')
        self.moe = TimeMoE(configs, device)

    def do_predict_next(self, cbm_deviations, user_param):
        LOOKBACK_WIN_FOR_PRED = 30 # lookback window size for predicting next
        if len(cbm_deviations) < LOOKBACK_WIN_FOR_PRED: 
            exp_deviation = torch.tensor(0)
        else:
            preds = self.moe.predict(torch.tensor([cbm_deviations], dtype=self.moe.model.dtype), context_len=LOOKBACK_WIN_FOR_PRED)
            exp_deviation = preds[0]
        return exp_deviation

    def do_train(self, y, y_hat_cbm):
        deviations = y - y_hat_cbm

        # TODO 
        # Fine-tune model ! 

        # estimate adjusted_prediction 
        y_hat = np.empty_like(y)
        MIN_LOOKBACK_WIN = 10 
        for t in range(0, len(deviations)):
            if t < MIN_LOOKBACK_WIN: 
                y_hat[t] = y_hat_cbm[t]
            else:
                pred_deviation, _ = self._predict_next(y[:t], y_hat[:t], y_hat_cbm[:t])
                y_hat[t] = y_hat_cbm[t] + pred_deviation

        return y_hat