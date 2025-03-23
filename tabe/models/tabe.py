
# Gaussian Process Model 
# ref) https://pyro.ai/examples/gp.html

import os
import io 
import time
import numpy as np
import torch

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from tabe.data_provider.dataset_loader import get_data_provider
from tabe.models.abstractmodel import AbstractModel
from tabe.models.timemoe import TimeMoE
from tabe.utils.mem_util import MemUtil
from tabe.utils.misc_util import logger, OptimTracker
import tabe.utils.weighting as weighting
import tabe.utils.report as report


_mem_util = MemUtil(rss_mem=False, python_mem=False)


class TabeModel(AbstractModel):

    def __init__(self, configs, combiner_model, adjuster_model=None):
        super().__init__(configs, "Tabe")
        self.combiner_model = combiner_model # must've been trained already
        self.adjuster_model = adjuster_model # Model used to adjust. TimeMoE
        self.y_hat = None
        self.truths = None # the last 'y' values. shape = (HPO_EVALUATION_PEROID)
        self.test_set = None

        # scalers used to normalize the training data while performing test (proceed_onestep)
        self.use_scaler = configs.use_batch_norm 
        self.scaler_x = StandardScaler() 
        self.scaler_y = StandardScaler() 
        self.scaler_x_m = StandardScaler() 
        self.scaler_y_m = StandardScaler() 
 

    def train(self):
        self.combiner_model.train()

        train_dataset, train_loader = get_data_provider(self.configs, flag='ensemble_train', step_by_step=True)
        y = train_dataset.data_y[self.configs.seq_len:, -1] # the next timestep truth [-1] is excluded
        assert len(y) == len(train_loader)

        y_hat_cbm = np.empty_like(y)
        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            y_hat_t, _ = self.combiner_model.proceed_onestep(
                batch_x, batch_y, batch_x_mark, batch_y_mark)
            y_hat_cbm[t] = y_hat_t
            # _mem_util.print_memory_usage()

        if self.adjuster_model is None:
            y_hat = y_hat_cbm
        else:
            y_hat = self.adjuster_model.train(y, y_hat_cbm)

        self.truths = y
        self.y_hat = y_hat
        self.y_hat_cbm = y_hat_cbm


    def _normalize_batch(self, batch_list, inverse=False):
        scaler_list = [self.scaler_x, self.scaler_y, self.scaler_x_m, self.scaler_y_m]
        for i, batch in enumerate(batch_list):
            scaler = scaler_list[i]
            batch_np = batch.squeeze(0).numpy() # batch.shape = (1, seq_len, feature_dim)
            if not inverse:
                scaled_batch_np = scaler.fit_transform(batch_np) 
            else:
                scaled_batch_np = scaler.inverse_transform(batch_np) 
            batch_list[i] = torch.from_numpy(scaled_batch_np).unsqueeze(0)
        return batch_list


    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        # truth at the next timestep
        y = batch_y[0, -1, -1] 

        if self.use_scaler:
            batch_x, batch_y, batch_x_mark, batch_y_mark = \
                self._normalize_batch([batch_x, batch_y, batch_x_mark, batch_y_mark], inverse=False)
            
        # get combiner model's predition
        y_hat_cbm, y_hat_bsm = self.combiner_model.proceed_onestep(
            batch_x, batch_y, batch_x_mark, batch_y_mark, training)                

        if self.use_scaler:
            batch_x, batch_y, batch_x_mark, batch_y_mark = \
                self._normalize_batch([batch_x, batch_y, batch_x_mark, batch_y_mark], inverse=True)

        if self.adjuster_model is None:
            y_hat_adj = y_hat_cbm
        else:
            # get next prediction of Adjuster 
            y_hat_adj = self.adjuster_model.proceed_onestep(y, y_hat_cbm, training)

        y_hat_tabe = y_hat_adj  

        self.y_hat_cbm = np.concatenate((self.y_hat_cbm, np.array([y_hat_cbm])))
        self.y_hat = np.concatenate((self.y_hat, np.array([y_hat_tabe])))
        self.truths = np.concatenate((self.truths, np.array([y])))

        # logger.info(f'Adj.predict : final_pred={final_pred:.5f}, y_hat={y_hat:.5f}, y_hat_cbm={y_hat_cbm:.5f}')
        return y_hat_tabe, y_hat_adj, y_hat_cbm, y_hat_bsm


    def forecast_onestep(self, invert=True):
        if self.test_set is None:
            self.test_set, _ = get_data_provider(self.configs, flag='test', step_by_step=True)
            self.y = self.test_set.data_y[self.configs.seq_len:, -1]
            self.need_to_invert_data = True if (self.test_set.scale and self.configs.inverse) else False
            self.tabe_pred = np.empty_like(self.y)
            self.cur_t = 0

        t = self.cur_t
        if t < len(self.y):
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.test_set[t]
            batch_x = torch.tensor(batch_x).unsqueeze(0)
            batch_y = torch.tensor(batch_y).unsqueeze(0)
            batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0)
            batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0)
            fcst, _, _, _ = \
                self.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, training=True)            

            truth = self.y[t]
            if self.need_to_invert_data:
                data_tf = np.zeros((1, self.test_set.data_y.shape[1]))
                data_tf[:, -1] = fcst
                fcst = self.test_set.inverse_transform(data_tf)[0, -1]
                data_tf[:, -1] = truth
                truth = self.test_set.inverse_transform(data_tf)[0, -1]

            self.tabe_pred[t] = fcst
            self.cur_t += 1
            return truth, fcst, self.test_set.df_raw.at[t,'date'], self.test_set.df_raw.at[t,'price']
        else:
            return None, None, None, None


    def test_step_by_step(self):
        fcsts = []
        while True:
            truth, fcst, date, price = self.forecast_onestep()
            if truth is None:
                logger.info("Test finished.")   
                break
            else:
                logger.info(f"{date} price={price:.1f}, Truth={truth:.5f}, Forecast={fcst:.5f}")
                fcsts.append(fcst)

        return self.y, np.array(fcsts)


    def test(self):
        test_set, test_loader = get_data_provider(self.configs, flag='test', step_by_step=True)
        y = test_set.data_y[self.configs.seq_len:, -1]
        need_to_invert_data = True if (test_set.scale and self.configs.inverse) else False

        tabe_pred = np.empty_like(y)
        y_hat_adj = np.empty_like(y)
        y_hat_cbm = np.empty_like(y)
        y_hat_bsm = np.empty((len(self.combiner_model.basemodels), len(y)))
        y_hat_q_low = np.empty_like(y)
        y_hat_q_high = np.empty_like(y)
        devi_stddev = np.empty_like(y)

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            tabe_pred[t], y_hat_adj[t], y_hat_cbm[t], y_hat_bsm[:,t] = \
                self.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark, training=True)            
            _mem_util.print_memory_usage()

        # if self.use_gpm:
        #     report.plot_gpmodel(self.gpm, filepath=self._get_result_path()+"/gpmodel_analysis.pdf")

        # TODO : check if dev_stddev is correctly used? 
        # z_val = norm.ppf(self.configs.quantile) 
        # y_hat_q_low = y_hat - devi_stddev * z_val 
        # y_hat_q_high = y_hat + devi_stddev * z_val

        # # TODO : check if dev_stddev is correctly used? 
        # z_val = norm.ppf(self.configs.buy_threshold_prob) 
        # buy_threshold_q = y_hat - devi_stddev * z_val

        if need_to_invert_data:
            n_features = test_set.data_y.shape[1]
            data_y = np.zeros((len(y), n_features))
            data_final_pred = np.zeros((len(y), n_features))
            # data_y_hat_q_low = np.zeros((len(y), n_features))
            # data_y_hat_q_high = np.zeros((len(y), n_features))
            data_buy_threshold_q = np.zeros((len(y), n_features))
            data_y_hat_cbm = np.zeros((len(y), n_features))
            data_y[:, -1] = y
            data_final_pred[:, -1] = tabe_pred
            # data_y_hat_q_low[:, -1] = y_hat_q_low
            # data_y_hat_q_high[:, -1] = y_hat_q_high
            # data_buy_threshold_q[:, -1] = buy_threshold_q
            data_y_hat_cbm[:, -1] = y_hat_cbm
            y = test_set.inverse_transform(data_y)[:, -1]
            tabe_pred = test_set.inverse_transform(data_final_pred)[:, -1]
            # y_hat_q_low = test_set.inverse_transform(data_y_hat_q_low)[:, -1]
            # y_hat_q_high = test_set.inverse_transform(data_y_hat_q_high)[:, -1]
            # buy_threshold_q = test_set.inverse_transform(data_buy_threshold_q)[:, -1]
            y_hat_cbm = test_set.inverse_transform(data_y_hat_cbm)[:, -1]
            for i in range(len(y_hat_bsm)):
                data_y_hat_bsm = np.zeros((len(y), n_features))
                data_y_hat_bsm[:, -1] = y_hat_bsm[i]
                y_hat_bsm[i] = test_set.inverse_transform(data_y_hat_bsm)[:, -1]

        # return y, tabe_pred, y_hat_cbm, y_hat_bsm, y_hat_q_low, y_hat_q_high, buy_threshold_q, devi_stddev
        return y, tabe_pred, y_hat_cbm, y_hat_bsm, y_hat_q_low, y_hat_q_high, None, devi_stddev
