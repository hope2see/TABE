

import os 
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
warnings.filterwarnings('ignore') 



def _stock_data_gen(
    filepathname, # filepath to store 
    asset = 'BTC-USD',  # ticker 
    target = 'LogRet', # One of ['Ret', 'LogRet', 'Price']
    start_date = '2021-01-01', 
    end_date = '2023-01-01',
    interval='1d'
):
    # auto_adjust=True is applied by default. 
    # So,'Close' price is used instead of 'Adj Close'
    df = yf.download(asset, start=start_date, end=end_date, interval=interval)['Close']
    df.columns = ['Close']

    # Target ---------

    percentage_return = df["Close"].pct_change()
    log_return = np.log(1 + percentage_return)

    if target == 'Ret':
        df["OT"] = percentage_return.shift(-1) 
        df[target] = percentage_return
    elif target == 'LogRet':
        df["OT"] = log_return.shift(-1)
        df[target] = log_return
    else:
        assert False

    # Features ---------

    # 0. Daily Return
    for i in [3, 5, 7]:
        df[target+ str(i)] = df[target].shift(-i)

    # 1. SMA: Simple Moving Average over a window
    for i in [3, 5, 7]:
        df["SMA" + str(i)] = df[target].rolling(window=i).mean()

    # 2. EMA: Exponential Moving Average
    for i in [3, 5, 7]:
        df["EMA" + str(i)] = df[target].ewm(span=i, adjust=False).mean()

    # 3. MACD: Usually MACD = EMA(12) - EMA(26), plus a 9-day signal line
    short_span = 12
    long_span = 26
    signal_span = 9
    df["EMA_short"] = df[target].ewm(span=short_span, adjust=False).mean()
    df["EMA_long"] = df[target].ewm(span=long_span, adjust=False).mean()
    df["MACD"] = df["EMA_short"] - df["EMA_long"]  # MACD line
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_span, adjust=False).mean()  # Signal line
    del df["EMA_short"], df["EMA_long"]

    # 4. RSI: Relative Strength Index (14-day typical)
    rsi_period = 14
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Exponential moving average of gains and losses
    avg_gain = pd.Series(gain).ewm(span=rsi_period, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(span=rsi_period, adjust=False).mean()
    # Calculate RS and then RSI
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs.values))

    # # 5. Momentum: close[t] - close[t-n]
    # for i in [3, 5, 10]:
    #     df["Momentum" + str(i)] = df["Close"].diff(periods=i)

    # # 6. ROC (Rate of Change): ((close[t] - close[t-n]) / close[t-n]) * 100
    # for i in [3, 5, 7]:
    #     df["ROC" + str(i)] = df["Close"].diff(periods=i) / df["Close"].shift(i) * 100

    # 7. ETS, 8. SARIMA
    df = df.dropna() # drop nan for calculation
    endog = df[target].values
    lookback_win = 20
    ets_pred = np.empty(len(df))
    sarima_pred = np.empty(len(df))
    for t in range(len(df)):
        if t < lookback_win:
            ets_pred[t] = np.nan
            sarima_pred[t] = np.nan
        else:
            ets_pred[t] = ExponentialSmoothing(endog[t-lookback_win:t], trend='add', damped_trend=True).fit().forecast(steps=1)
            sarima_pred[t] = SARIMAX(endog[t-lookback_win:t], order=(1,1,0), trend='ct', enforce_stationarity=False).fit(disp=False).forecast(steps=1)
    df['ETS'] = ets_pred
    df['SARIMA'] = sarima_pred

    # -------------------------

    # del df["Close"]
    df.rename(columns={'Close': 'price'}, inplace=True)

    df = df.dropna()

    df.reset_index(inplace=True)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.rename(columns={'Date': 'date'}, inplace=True)

    # Save the dataset as csv file
    df.to_csv(path_or_buf=filepathname, index=False)



# To be able to keep the temporal sequence of the 'continous onestep test/train' for the test_peroid, 
# validation_set should not be located in the middle of train_set and test_set. 
# In that case, lookup_window (length of seq_len) of the first datapoint of test_set 
# would be overlapped with the validation_set, which should be kept 'unseen' area for validation! 
# So, we moved validation_period to the first part of dataset. 
# [validateion_set (20%) | train_set (50%) | ensemble_train_set (20%) | test_set (10%)]
#
class Dataset_TABE_File(Dataset):
    def __init__(self, 
        args, 
        root_path, data_path, 
        size,  # [seq_len, label_len, pred_len]
        flag='base_train', # 'val', 'base_train', 'ensemble_train', 'test'
        features='S', 
        target='OT', 
        scale=True, 
        timeenc=0, 
        freq='h', 
        seasonal_patterns=None
    ):
        self.configs = args

        # test_split : ratio (0.2), or length (100), or start_date ('2020-01-01') of the test_period. 
        self.test_split_type, self.test_split = self._parse_test_split(self.configs.data_test_split)

        # train_splits : [0.2, 0.6, 0.2], # [val, base_train, ensemble_train] ratios for the period before test period. 
        self.train_splits = self.configs.data_train_splits

        self.root_path = root_path
        self.data_path = data_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        assert flag in ['val', 'base_train', 'ensemble_train', 'test']
        type_map = {'val': 0, 'base_train': 1, 'ensemble_train': 2, 'test': 3}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq

        self.scale = scale
        if scale:
            self.scaler = StandardScaler()

        self.__read_data__()

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
            return 'date', dt
        except ValueError:
            pass

        return 'unknown', None

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        if self.test_split_type == 'date':
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            assert df_raw['date'].values[0] < self.test_split < df_raw['date'].values[-1]
            num_test = len(df_raw[df_raw['date'] >= self.test_split])
        elif self.test_split_type == 'ratio':
            num_test = int(len(df_raw) * self.test_split)
        else: # length
            num_test = self.test_split

        num_rest = len(df_raw) - num_test

        num_vali = int(num_rest * self.train_splits[0]) 
        num_base_train = int(num_rest * self.train_splits[1])
        num_ensemble_train = num_rest - (num_vali + num_base_train)
        logger.info(f"Dataset Period : [Val {num_vali} | Base Train {num_base_train} | Ensemble Train {num_ensemble_train} | Test {num_test}")
        assert num_vali > 20+self.seq_len, f"num_vali({num_vali}) should be larger than 20+seq_len({self.seq_len})"

        border1s = [0, num_vali - self.seq_len, num_vali + num_base_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_vali, num_vali + num_base_train, num_vali + num_base_train + num_ensemble_train, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[1]:border2s[1]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # if self.set_type == 0 and self.args.augmentation_ratio > 0:
        #     self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
        self.data_stamp = data_stamp

        # df_raw is used for testing & trade simulation 
        if self.set_type == 3: # test 
            self.df_raw = df_raw[-num_test:].reset_index(drop=True) 


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        else :
            return data
        
    
class Dataset_TABE_Online(Dataset_TABE_File):
    def __init__(self, args, root_path='./dataset', flag='base_train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):        
        
        data_path = args.data_asset  + '_' + args.target_datatype + '_' \
            + args.data_start_date + '_' + args.data_end_date + '_' + args.data_interval + '.csv'
        full_path = root_path + '/' + data_path

        if not os.path.exists(full_path):
            pathdir = os.path.dirname(full_path)
            if not os.path.exists(pathdir):
                os.makedirs(pathdir)
            _stock_data_gen(full_path, args.data_asset, args.target_datatype, args.data_start_date, args.data_end_date, args.data_interval)   

        super().__init__(args, root_path, data_path, size, flag, features, target, scale, timeenc, freq, seasonal_patterns)

