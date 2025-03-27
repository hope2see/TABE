import os
import numpy as np
import pandas as pd
from utils.timefeatures import time_features
from torch.utils.data import Dataset, DataLoader
from tabe.utils.misc_util import logger
from tabe.data_provider.dataset_tabe import Dataset_TABE_File, Dataset_TABE_Online, Dataset_TABE_Live

import warnings
warnings.filterwarnings('ignore')


data_dict = {
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour,
    # 'ETTm1': Dataset_ETT_minute,
    # 'ETTm2': Dataset_ETT_minute,
    # 'custom': Dataset_Custom,
    # 'm4': Dataset_M4,
    # 'PSM': PSMSegLoader,
    # 'MSL': MSLSegLoader,
    # 'SMAP': SMAPSegLoader,
    # 'SMD': SMDSegLoader,
    # 'SWAT': SWATSegLoader,
    # 'UEA': UEAloader,
    'TABE_FILE': Dataset_TABE_File,
    'TABE_ONLINE': Dataset_TABE_Online,
    'TABE_LIVE': Dataset_TABE_Live,
}


def _data_provider(configs, flag, step_by_step=False, df_raw=None):
    Data = data_dict[configs.data]
    timeenc = 0 if configs.embed != 'timeF' else 1
    shuffle_flag = False if (flag == 'test' or flag == 'TEST' or step_by_step) else True
    drop_last = False
    batch_size = configs.batch_size if not step_by_step else 1

    if configs.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = configs,
            root_path=configs.root_path,
            win_size=configs.seq_len,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=configs.num_workers,
            drop_last=drop_last)    
    elif configs.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args = configs,
            root_path=configs.root_path,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=configs.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=configs.seq_len)
        )
    else:
        if configs.data == 'm4':
            drop_last = False
        if configs.data in ['TABE_FILE', 'TABE_ONLINE']:
            data_set = Data(configs, 
                [configs.seq_len, configs.label_len, configs.pred_len],
                flag, timeenc)            
        elif configs.data == 'TABE_LIVE':
            data_set = Data(configs, 
                [configs.seq_len, configs.label_len, configs.pred_len],
                flag, timeenc, df_raw)   
        else:
            data_set = Data(
                args = configs,
                root_path=configs.root_path,
                data_path=configs.data_path,
                flag=flag,
                size=[configs.seq_len, configs.label_len, configs.pred_len],
                features=configs.features,
                target=configs.target,
                timeenc=timeenc,
                freq=configs.freq,
                seasonal_patterns=configs.seasonal_patterns
            )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=configs.num_workers,
            drop_last=drop_last)
        
    return data_set, data_loader


# Return (Dataset, DataLoader) tuple for the given parameters. 
# We assume that the values of 'args' are always the same. 
# If a (Dataset, DataLoader) correspoinding to the 'flag' and 'step_by_step' parameters
# has been created before, then the cached objects are retunred. 
_cache_dataset = {}
_cache_dataloader = {}
def get_data_provider(configs, flag, step_by_step=False, df_raw=None):
    if configs.data == 'Dataset_TABE_Live':
        key = configs.data # Dataset_TABE_Live is shared for all flag (ensemble_train, train)
    else :
        key = configs.data + flag
    
    if key not in _cache_dataset:
        _cache_dataset[key], _cache_dataloader[key] = _data_provider(configs, flag, step_by_step, df_raw)

    if configs.data == 'Dataset_TABE_Live':
        _cache_dataset[key].set_flag(flag)

    return _cache_dataset[key], _cache_dataloader[key]
