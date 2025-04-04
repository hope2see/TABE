
import os
import numpy as np
import random
import torch
import time
import warnings
import numpy as np

from utils.metrics import metric

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

from tabe.utils.logger import logger
from tabe.utils.misc_util import OptimTracker
from tabe.data_provider.dataset_loader import get_data_provider
from tabe.models.abstractmodel import AbstractModel
from tabe.utils.logger import logger


# warnings.filterwarnings('ignore')


# NOTE 
# Statistical models (ETS, and SARIMA) are available only for univariate forecasting,
# So, they are fitted only using the target variable.
class StatisticalModel(AbstractModel):
    def __init__(self, configs, name):
        super().__init__(configs, name) 

    def _fit(self, endog):
        raise NotImplementedError

    # Statistical Models do not need to be trained, 
    # since they can fit with the 'seq_len' number of data points when prediction is needed.
    def train(self):
        pass

    def load_saved_model(self):
        pass

    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        assert self.configs.label_len == 1  
        # batch_x.shape=(B,S,F) B(Batch Size)=1, S(Sequence Length), F(Feature Dimension)
        endog = batch_x[0, :, -1].numpy()
        pred = self._fit(endog).forecast(steps=1)
        return pred[0]
    

class EtsModel(StatisticalModel):
    def __init__(self, configs, name="ETS"):
        super().__init__(configs, name) 

    def _fit(self, endog):
        # NOTE: Use auto-finding of the hyperparameters for ETS
        return ExponentialSmoothing(endog, trend='add', damped_trend=True).fit()


class SarimaModel(StatisticalModel):
    def __init__(self, configs, name="SARIMA"):
        super().__init__(configs, name) 

    def _fit(self, endog):
        # NOTE: Use auto-finding of the hyperparameters for SARIMA
        return SARIMAX(endog, order=(1,1,0), trend='ct', enforce_stationarity=False).fit(disp=False)


class AutoSarimaModel(StatisticalModel):
    def __init__(self, configs, name="AutoSARIMA"):
        super().__init__(configs, name) 

    def _fit(self, endog):
        # pmdarima.auto_arima() often fails when endog series contains outliers.
        # Signs of Failure
        # •	Forecast returns [0.] or constant values
        # •	Model summary shows all coefficients are 0
        # •	d, p, or q values look unexpected (e.g., all 0s)
        # •	Model converges but performance is bad
        # 
        # To walk-around this issue, we do 'winsorizing' (clipping outliers)        
        #
        # OUTLIER_THRESHOLD = 3 * np.std(endog) # clip top 0.135%, and bottom -0.135% outliers 
        # endog = np.clip(endog, -OUTLIER_THRESHOLD, OUTLIER_THRESHOLD)
        # Alternative way : 
        # from scipy.stats.mstats import winsorize
        # endog = winsorize(endog, limits=[0.1, 0.1])  # clip top/bottom 10% 

        self.model = pm.auto_arima(
            endog,                 # training series
            start_p=1, #start_q=1,  # initial range for p and q
            # max_p=3, max_q=3,      # upper bounds for p and q
            # start_P=0, start_Q=0,  # initial range for P and Q
            # max_P=2, max_Q=2,      # upper bounds for seasonal P and Q
            seasonal=False,         # enable SARIMA
            # m=12,                  # seasonal period (e.g. 12 for monthly data with yearly seasonality)
            # d=None,                # let auto_arima find non-seasonal differencing
            # D=None,                # let auto_arima find seasonal differencing
            trace=False,            # print output of model selection
            error_action='warn',
            suppress_warnings=False,
            # stepwise=True          # more efficient, stepwise approach
        )
        return self
    
    def forecast(self, steps=1):
        pred = self.model.predict(n_periods=steps)
        if pred[0] == 0 and self.model.order == (0,0,0):
            logger.debug('AutoSarimaModel seems to have failed in fitting! (pred is 0.0)')
            # self.model.summary()
        return pred
