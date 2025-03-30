
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
from tabe.data_provider.dataset_tabe import Dataset_TABE_Live
from tabe.models.abstractmodel import AbstractModel
from tabe.models.timemoe import TimeMoE
from tabe.utils.mem_util import MemUtil
from tabe.utils.misc_util import logger, OptimTracker
import tabe.utils.weighting as weighting
import tabe.utils.report as report


_mem_util = MemUtil(rss_mem=False, python_mem=False)


class TabeModel(AbstractModel):

    def __init__(self, configs, device, combiner_model, adjuster_model=None):
        super().__init__(configs, device, "Tabe")
        self.combiner = combiner_model # must've been trained already
        self.adjuster = adjuster_model # Model used to adjust. TimeMoE

        # scalers used to normalize the training data while performing test (proceed_onestep)
        self.use_scaler = configs.use_batch_norm 
        self.scaler_x = StandardScaler() 
        self.scaler_y = StandardScaler() 
        self.scaler_x_m = StandardScaler() 
        self.scaler_y_m = StandardScaler() 

    def prepare_result_storage(self, total_steps=None):
        super().prepare_result_storage(total_steps)
        self.combiner.prepare_result_storage(total_steps)

    def train(self, esb_train_dataset=None, esm_train_loader=None):
        self.combiner.train()

        if esb_train_dataset is None :
            esb_train_dataset, esm_train_loader = get_data_provider(self.configs, flag='ensemble_train', step_by_step=True)

        self.combiner.prepare_result_storage(len(esb_train_dataset)) 

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(esm_train_loader):
            self.combiner.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)

        truths = esb_train_dataset.data_y[-len(esb_train_dataset):, -1] 
        cbm_preds = self.combiner.predictions

        if self.adjuster is not None:
            self.adjuster.train(truths, cbm_preds)


    # def _normalize_batch(self, batch_list, inverse=False):
    #     scaler_list = [self.scaler_x, self.scaler_y, self.scaler_x_m, self.scaler_y_m]
    #     for i, batch in enumerate(batch_list):
    #         scaler = scaler_list[i]
    #         batch_np = batch.squeeze(0).numpy() # batch.shape = (1, seq_len, feature_dim)
    #         if not inverse:
    #             scaled_batch_np = scaler.fit_transform(batch_np) 
    #         else:
    #             scaled_batch_np = scaler.inverse_transform(batch_np) 
    #         batch_list[i] = torch.from_numpy(scaled_batch_np).unsqueeze(0)
    #     return batch_list


    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # truth (target value) for the previous timestep
        truth = batch_y[0, -1, -1] 

        # if self.use_scaler:
        #     batch_x, batch_y, batch_x_mark, batch_y_mark = \
        #         self._normalize_batch([batch_x, batch_y, batch_x_mark, batch_y_mark], inverse=False)
            
        cbm_pred, _ = self.combiner.proceed_onestep(
            batch_x, batch_y, batch_x_mark, batch_y_mark)

        # if self.use_scaler:
        #     batch_x, batch_y, batch_x_mark, batch_y_mark = \
        #         self._normalize_batch([batch_x, batch_y, batch_x_mark, batch_y_mark], inverse=True)

        if self.adjuster is None:
            pred = cbm_pred
        else:
            pred = self.adjuster.adjust_onestep(truth, cbm_pred)

        # logger.info(f'Adj.predict : final_pred={final_pred:.5f}, y_hat={y_hat:.5f}, y_hat_cbm={y_hat_cbm:.5f}')
        return pred


    def test(self):
        assert self.configs.data != 'TABE_LIVE', 'tabe.test() is not intended to be used for TABE_LIVE data.'

        test_set, test_loader = get_data_provider(self.configs, flag='test', step_by_step=True)

        self.prepare_result_storage(len(test_set)) 

        for t, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            self.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)

        # 'self.dataset' is used when result reporting 
        self.dataset = test_set

        # TODO : check if dev_stddev is correctly used? 
        # z_val = norm.ppf(self.configs.quantile) 
        # y_hat_q_low = y_hat - devi_stddev * z_val 
        # y_hat_q_high = y_hat + devi_stddev * z_val

        # # TODO : check if dev_stddev is correctly used? 
        # z_val = norm.ppf(self.configs.buy_threshold_prob) 
        # buy_threshold_q = y_hat - devi_stddev * z_val


    #===========================================================================
    # Interface for training/testing with 'live' data (fed on-the-fly)

    def forecast_onestep(self, df_new_data):
        """
        Intended to be used only for 'test' dataset of 'Live' data.
        """
        # assert self.live_dataset.isinstance(Dataset_TABE_Live) and self.live_dataset.flag == 'test'
        
        batch_x, batch_y, batch_x_mark, batch_y_mark, = self.dataset.feed_data(df_new_data)

        batch_x = torch.tensor(batch_x).unsqueeze(0)
        batch_y = torch.tensor(batch_y).unsqueeze(0)
        batch_x_mark = torch.tensor(batch_x_mark).unsqueeze(0)
        batch_y_mark = torch.tensor(batch_y_mark).unsqueeze(0)
        fcst, _ = \
            self.proceed_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)            

        if self.dataset.scale:
            data_tf = np.zeros((1, self.dataset.data_y.shape[1]))
            data_tf[:, -1] = fcst
            fcst = self.dataset.inverse_transform(data_tf)[0, -1]

        return fcst, None, None
    

    def prepare_live(self, df_previous_data):
        self.dataset, data_loader = get_data_provider(self.tabe_model.configs, 
                                                                flag='ensemble_train', step_by_step=True, df_raw=df_previous_data)
        
        self.prepare_result_storage(len(self.dataset)) 

        self.train(self.dataset, data_loader)

        self.dataset.set_flag('test')

        self.prepare_result_storage() 

