import os 
from datetime import datetime, timedelta
import re
import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from utils.timefeatures import time_features
from tabe.utils.misc_util import logger

import warnings
# warnings.filterwarnings('ignore') 


_default_feature_config = {
    # feature name : parameter(s)
    'LogRet' : ('Close', (1,)), # log return (col, spans)
    # 'ETS' : ('LogRet1', 20, 'add', True), # ETS (col, lookback_win, trend, damped_trend)
    # 'SARIMA' : ('LogRet1', 20, 'ct', False), # ETS (col, lookback_win, trend, enforce_stationarity)
    # 'LogRet' : ('Close', (1, 3, 5, 7)), # log return (col, spans)
    # 'SMA' : ('LogRet1', (3, 5, 7)), # SMA (col, spans)
    # 'EMA' : ('LogRet1', (3, 5, 7), False), # EMA (col, spans, adjust)
    # 'MACD' : ('LogRet1', 12, 26, False), # MACD (col, short_span, long_span, adjust)
    # 'MACD_Signal' : ('LogRet1', 9, False), # MACD_Signal (col, span, adjust)
    # 'RSI' : ('Close', 14, False), # RSI (col, spans, adjust)    
    # 'ETS' : ('LogRet1', 20, 'add', True), # ETS (col, lookback_win, trend, damped_trend)
    # 'SARIMA' : ('LogRet1', 20, 'ct', False), # ETS (col, lookback_win, trend, enforce_stationarity)
}


def gen_log_return(df_data, col, spans):
    for span in spans:
        pct_return = df_data[col].pct_change(periods=span)
        df_data['LogRet' + str(span)] = np.log(1 + pct_return)


def gen_sma(df_data, col, spans):
    for span in spans:
        df_data["SMA" + str(span)] = df_data[col].rolling(window=span).mean()


def gen_ema(df_data, col, spans, adjust):
    for span in spans:
        df_data["EMA" + str(span)] = df_data[col].ewm(span=span, adjust=adjust).mean()


def gen_macd(df_data, col, short_span, long_span, adjust):
    signal_span = 9
    df_data["EMA_short"] = df_data[col].ewm(span=short_span, adjust=adjust).mean()
    df_data["EMA_long"] = df_data[col].ewm(span=long_span, adjust=adjust).mean()
    df_data["MACD"] = df_data["EMA_short"] - df_data["EMA_long"]  
    del df_data["EMA_short"], df_data["EMA_long"]


def gen_macd_signal(df_data, macd_col, signal_span, adjust):
    df_data["MACD_Signal"] = df_data[macd_col].ewm(span=signal_span, adjust=adjust).mean()  


def gen_rsi(df_data, col, span, adjust):
    delta = df_data[col].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=span, adjust=adjust).mean()
    avg_loss = pd.Series(loss).ewm(span=span, adjust=adjust).mean()
    rs = avg_gain / avg_loss
    df_data["RSI"] = 100 - (100 / (1 + rs.values))


def gen_ets(df_data, col, lookback_win, trend, damped_trend, steps=1):
    endog = df_data[col].values
    pred = np.empty(len(df_data))
    # The data at the current time (data at index 't') should be included in lookback window 
    # So, lookback_win = [t+1-lookback_win : t+1]
    for t in range(len(df_data)):
        if t+1 < lookback_win:
            pred[t] = np.nan
        else:
            pred[t] = ExponentialSmoothing(endog[t+1-lookback_win : t+1], trend=trend, 
                                           damped_trend=damped_trend).fit().forecast(steps=steps)
    df_data['ETS'] = pred
    

def gen_sarima(df_data, col, lookback_win, trend, enforce_stationarity, steps=1):
    endog = df_data[col].values
    pred = np.empty(len(df_data))
    # The data at the current time (data at index 't') should be included in lookback window 
    # So, lookback_win = [t+1-lookback_win : t+1]
    for t in range(len(df_data)):
        if t+1 < lookback_win:
            pred[t] = np.nan
        else:
            # FIX ME! to AutoSarima
            pred[t] = SARIMAX(endog[t+1-lookback_win : t+1], order=(1,1,0), trend=trend, 
                              enforce_stationarity=enforce_stationarity).fit(disp=False).forecast(steps=steps)
    df_data['SARIMA'] = pred


# # 5. Momentum: close[t] - close[t-n]
# for i in [3, 5, 10]:
#     df["Momentum" + str(i)] = df["Close"].diff(periods=i)

# # 6. ROC (Rate of Change): ((close[t] - close[t-n]) / close[t-n]) * 100
# for i in [3, 5, 7]:
#     df["ROC" + str(i)] = df["Close"].diff(periods=i) / df["Close"].shift(i) * 100


_feature_fn = {
    'LogRet' : gen_log_return,
    'SMA' : gen_sma,
    'EMA' : gen_ema,
    'MACD' : gen_macd,
    'MACD_Signal' : gen_macd_signal,
    'RSI' : gen_rsi,
    'ETS' : gen_ets,
    'SARIMA' : gen_sarima
}


def _populate_features(config, feature_config, df, copy=False):
    if copy :
        df = df.copy()

    for key, val in feature_config.items():
        _feature_fn[key](df, *val)

    df = df.dropna()
    df.reset_index(inplace=True)
    return df


def _stock_data_gen(config, filepathname, feature_config):    
    # 'Close' price is used instead of 'Adj Close', since auto_adjust is True by default. 
    df = yf.download(config.data_asset, start=config.data_start_date, end=config.data_end_date, interval=config.data_interval)['Close']
    df.columns = ['Close']

    df = _populate_features(config, feature_config, df)

    # Save the dataset as csv file
    df.to_csv(path_or_buf=filepathname, index=False)


# def _parse_interval(interval: str) -> timedelta:
#     """
#     Parses a yfinance interval string (e.g., '1m', '5h', '3d', '1wk') into a tuple.
#     Returns:
#         (value: int, unit: str) or raises ValueError if invalid format
#     """
#     match = re.fullmatch(r"(\d+)(m|h|d)", interval)
#     if not match:
#         raise ValueError(f"Invalid interval format: '{interval}'")
    
#     value = int(match.group(1))
#     unit = match.group(2)

#     td_interval = None
#     if unit == 'm':
#         td_interval =  timedelta(minutes=value)
#     elif unit == 'h':
#         td_interval =  timedelta(hours=value)
#     elif unit == 'd':
#         td_interval =  timedelta(days=value)
#     else:
#         print(f"Unknown interval unit : {unit}")
            
#     return td_interval, value, unit


# To be able to keep the temporal sequence of the 'continous onestep test/train' for the test_peroid, 
# validation_set should not be located in the middle of train_set and test_set. 
# In that case, lookup_window (length of seq_len) of the first datapoint of test_set 
# would be overlapped with the validation_set, which should be kept 'unseen' area for validation! 
# So, we moved validation_period to the first part of dataset. 
# [validateion_set (20%) | train_set (50%) | ensemble_train_set (20%) | test_set (10%)]
#
class Dataset_TABE_Base(Dataset):
    def __init__(self, 
        config, 
        size,  # [seq_len, label_len, pred_len]
        flag, # 'val', 'base_train', 'ensemble_train', 'test'
        timeenc,
        df_data, 
        copy=True, 
        scale=True,
    ):
        self.config = config
        self.timeenc = timeenc
        self.features = config.features
        self.target = config.target
        self.freq = config.freq

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['val', 'base_train', 'ensemble_train', 'test']
        self.flag = flag

        self.scale = scale
        if scale:
            self.scaler = StandardScaler()

        self._read_data(df_data, copy)


    def _parse_test_split(self, test_split):
        try:
            len = int(test_split)
            return 'length', len
        except ValueError:
            pass

        try:
            ratio = float(test_split)
            return 'ratio', ratio
        except ValueError:
            pass

        try:
            dt = pd.to_datetime(test_split)
            return 'Date', dt
        except ValueError:
            pass

        return 'unknown', None


    def _get_border(self, df_data):
        raise NotImplementedError


    def _prepare_timestamp(self, df_data, border1, border2):
        df_stamp = df_data[['Date']][border1:border2]
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['Date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['Date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        self.data_stamp = data_stamp
        
    
    def _prepare_data_xy(self, df_data, data_begin, data_end, ref_data_begin=None, ref_data_end=None):
        """
            data_begin, data_end : borders of data
            ref_data_begin, ref_data_end : borders of reference_data used to normalize the given data
        """
        # move 'target' to the right-most column
        cols = list(df_data.columns)
        cols.remove(self.target)
        df_data = df_data[cols + [self.target]]

        # Remove unused columns 
        if self.features == 'M' or self.features == 'MS':
            del df_data['Date']
        elif self.features == 'S':
            # remove all except target column
            df_data = df_data[[self.target]]

        # normalize 
        if self.scale:
            ref_data = df_data[ref_data_begin:ref_data_end]
            self.scaler.fit(ref_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[data_begin:data_end]
        self.data_y = data[data_begin:data_end]
        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)


    def _read_data(self, df_data, copy=True):
        if copy:
            df_data = df_data.copy()

        begin, end, ref_data_begin, ref_data_end = self._get_border(df_data)

        self._prepare_timestamp(df_data, begin, end)

        self._prepare_data_xy(df_data, begin, end, ref_data_begin, ref_data_end)


    def __getitem__(self, index):
        """
        NOTE : 
        For 'ensemble_train' or 'test' dataset, 
        seq_y and seq_y_mark do not have the truths for next prediction! 
        In that case, loss calculation is performed not for the last prediction, not for the current prediction. 
        By changing the implementation this way, we can use the last (most recent) data point as 
        normal input data for predict the next target values. 
        """
        if index < 0: 
            index = len(self) + index

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        # NOTE: The truths for next prediction is NOT included 
        # in 'ensemble_train' and 'test' dataset! 
        r_end = r_begin + self.label_len 
        if self.flag in ['val', 'train']:
            r_end += self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # For 'ensemble_train' and 'test' dataset, we don't discard the last 'pred_len' data points.
        # This allows the model to forecast for the next target values after the last data points.
        # Since the next target values are unavailable at the last time steps,
        # forecasting losses cannot be computed for that prediction.  
        if self.flag in ['val', 'train']:
            return len(self.data_x) - self.seq_len - self.pred_len + 1
        else:
            return len(self.data_x) - self.seq_len + 1


    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        else :
            return data


class Dataset_TABE_Live(Dataset_TABE_Base):

    def __init__(self, config, size, flag, timeenc, df_previous, feature_config=_default_feature_config):
        # assume that Live Dataset is used for ensemble_train or test 
        assert flag in ['ensemble_train']

        self.df_raw = df_previous.copy()
        self.feature_config = feature_config

        df_data = _populate_features(config, feature_config, self.df_raw, copy=True)
        super().__init__(config, size, flag, timeenc, df_data, copy=False, scale=True)


    def set_flag(self, flag):
        # By setting flag, ensemble_train dataset can be changed to be 'test' dataset.
        self.flag = flag


    def _get_border(self, df_data):
        """
        return borders : begin, end, ref_data_begin, ref_data_end
        """
        assert len(df_data) > self.seq_len, f"len(df_data)({len(df_data)}) should be larger than seq_len({self.seq_len})"
        
        if self.flag == 'ensemble_train':
            begin = 0
            end = len(df_data)
        else: # test 
            # Assume that the latest data has been fed! 
            begin = len(df_data) - self.seq_len
            end = len(df_data)

        return begin, end, begin, end 

    
    def feed_data(self, df_onestep_data: pd.DataFrame, steps=1):
        # assert len(df_onestep_data) == self.config.seq_len
        assert len(df_onestep_data) == 1
        assert steps == 1 # currently only 1 step supported 

        # assume the df_raw is the right next data of self.df_raw     
        # TODO : minimize the size of keeping data 
        self.df_raw = pd.concat([self.df_raw, df_onestep_data])

        df_data = _populate_features(self.config, self.feature_config, self.df_raw, copy=True)

        self._read_data(df_data, copy=False)

        return self[-1] # return the latest data


class Dataset_TABE_File(Dataset_TABE_Base):
    def __init__(self, config, size, flag, timeenc, root_path=None, data_path=None):
        root_path = config.root_path if root_path is None else root_path
        data_path = config.data_path if data_path is None else data_path        
        df_data = pd.read_csv(os.path.join(root_path,data_path))
        df_data['Date'] = pd.to_datetime(df_data['Date'])
        super().__init__(config, size, flag, timeenc, df_data, copy=False, scale=True)

    def _get_border(self, df_data):
        """
        return borders : begin, end, ref_data_begin, ref_data_end
        """
        # test_split : ratio (0.2), or length (100), or start_date ('2020-01-01') of the test_period. 
        test_split_type, test_split = self._parse_test_split(self.config.data_test_split)

        # train_splits : [0.2, 0.6, 0.2], # [val, base_train, ensemble_train] ratios for the period before test period. 
        train_splits = self.config.data_train_splits

        if test_split_type == 'Date':
            assert df_data['Date'].values[0] < test_split < df_data['Date'].values[-1]
            num_test = len(df_data[df_data['Date'] >= test_split])
        elif test_split_type == 'ratio':
            num_test = int(len(df_data) * test_split)
        else: # length
            num_test = test_split

        num_rest = len(df_data) - num_test

        num_vali = int(num_rest * train_splits[0]) 
        num_base_train = int(num_rest * train_splits[1])
        num_ensemble_train = num_rest - (num_vali + num_base_train)
        logger.info(f"Dataset Period : [Val {num_vali} | Base Train {num_base_train} | Ensemble Train {num_ensemble_train} | Test {num_test}")
        assert num_vali > 20+self.seq_len, f"num_vali({num_vali}) should be larger than 20+seq_len({self.seq_len})"

        val_borders             = (0,                                       num_vali)
        base_train_borders      = (val_borders[1]-self.seq_len,             val_borders[1] + num_base_train)
        # For 'ensemble_train', and 'test', the target(truth) value of next step is not included. 
        ensemble_train_borders  = (base_train_borders[1]-(self.seq_len-1),    base_train_borders[1] + num_ensemble_train)
        test_borders            = (ensemble_train_borders[1]-(self.seq_len-1), ensemble_train_borders[1] + num_test)
        assert test_borders[1] == len(df_data)

        borders = {
            'val' : val_borders,
            'base_train' : base_train_borders,
            'ensemble_train' : ensemble_train_borders,
            'test' : test_borders
        }

        # base_train_dataset is used as 'reference data' when normalizing other dataset
        return borders[self.flag][0], borders[self.flag][1], borders['base_train'][0],  borders['base_train'][1]


class Dataset_TABE_Online(Dataset_TABE_File):
    def __init__(self, config, size, flag, timeenc, feature_config=_default_feature_config):        
        
        data_path = config.data_asset  + '_' + config.target + '_' \
            + config.data_start_date + '_' + config.data_end_date + '_' + config.data_interval + '.csv'
        full_path = config.root_path + '/' + data_path

        if not os.path.exists(full_path):
            pathdir = os.path.dirname(full_path)
            if not os.path.exists(pathdir):
                os.makedirs(pathdir)
            _stock_data_gen(config, full_path, feature_config)   

        super().__init__(config, size, flag, timeenc, config.root_path, data_path)

