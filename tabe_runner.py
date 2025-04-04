# ".env" file is used instead of the sys.path.append() method
# import sys
# Add Time-Series-Library directory to module lookup paths
# sys.path.append('Time-Series-Library')

import os 
from datetime import datetime
import argparse
import copy
import numpy as np
import pandas as pd
import random
import torch
import logging

# Time-Series-Library
from models import DLinear, PatchTST, iTransformer, TimeXer

# CMamba
from cmamba.models import CMamba

# TABE 
from tabe.models.abstractmodel import AbstractModel
from tabe.models.stat_model import EtsModel, SarimaModel, AutoSarimaModel
from tabe.models.synth_model import DrifterModel, NoiseModel
from tabe.models.tslibmodel import TSLibModel
from tabe.models.timemoe import TimeMoE
from tabe.models.timer import Timer
from tabe.models.timesfm import TimesFM
from tabe.models.combiner import CombinerModel
from tabe.models.adjuster import AdjusterModel, TimeMoeAdjuster #, GpmAdjuster
from tabe.models.tabe import TabeModel
from tabe.utils.mem_util import MemUtil
from tabe.utils.report import report_results
from tabe.utils.misc_util import print_configs
from tabe.utils.logger import logger, default_formatter


_mem_util = MemUtil(rss_mem=True, python_mem=True)


def _set_seed(fix_seed = 2025):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


def _basemodel_args(arg_value):
    argv_list = [v.strip() for v in arg_value.split()]
    model_name = argv_list[0]
    model_args = None
    if len(argv_list) > 1:
        model_args = argv_list[1:] if len(argv_list) > 1 else []
        model_args = _get_parser(model_name).parse_args(model_args)
    return (model_name, model_args)


def _model_args(arg_value, model_name):
    assert model_name in ['combiner', 'adjuster']
    argv_list = [v.strip() for v in arg_value.split()]
    model_args = None
    if len(argv_list) > 0:
        model_args = _get_parser(model_name).parse_args(argv_list)
    return model_args


def _get_parser(model_name=None):
    parser = argparse.ArgumentParser()

    # global arguments, not-overidable by the model arguments
    if model_name is None: 

        # basic config
        parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                            help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
        parser.add_argument('--model_id', type=str, required=False, default='TABE1.0', help='model id')
        parser.add_argument('--model', type=str, required=False, default='TABE',
                            help='model name, options: [DLinear, PatchTST, iTransformer, TimeXer, CMamba, TimeMoE, TABE]')
        parser.add_argument('--tabe_root_path', type=str, default='.', help='root path of tabe directory tree')

        # data loader
        parser.add_argument('--data', type=str, required=False, default='TABE_FILE', help='dataset type')
        parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
        parser.add_argument('--data_path', type=str, default='BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv', help='data file')
        parser.add_argument('--features', type=str, default='MS',
                            help='forecasting task, options:[M, S, MS]; M:multichannel predict multichannel, S:unichannel predict unichannel, MS:multichannel predict unichannel')
        parser.add_argument('--target', type=str, default='LogRet1', help='target feature in S or MS task [LogRet1]')
        parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
        parser.add_argument('--data_asset', type=str, default='BTC-USD', help='ticker of asset')
        parser.add_argument('--data_start_date', type=str, default='2021-01-01', help='start date for the data to download. Used for TABE_ONLINE data')
        parser.add_argument('--data_end_date', type=str, default='2023-01-01', help='end date for the data to download. Used for TABE_ONLINE data')
        # TODO : FIX the duplicaation of 'freq' and 'data_interval' 
        parser.add_argument('--freq', type=str, default='d', 
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        parser.add_argument('--data_interval', type=str, default='1d', help='interval parameter for downloading')
        parser.add_argument('--data_test_split', type=str, default='0.3', help='ratio (0.3), or length (100), or start_date (\'2020-01-01\') of the test_period')
        parser.add_argument('--data_train_splits', type=float, nargs='+', default=[0.3, 0.6, 0.1], 
                            help='[val, base_train, ensemble_train] ratios for the period before test period')
        
        # trading simulation 
        # parser.add_argument('--buy_fee', type=float, default=0.001, help='')
        # parser.add_argument('--sell_fee', type=float, default=0.001, help='')
        parser.add_argument('--fee_rate', type=float, default=0.001, help='')
        parser.add_argument('--buy_threshold_ret', type=float, default=0.002, 
                            help='The threshold of model\'s predicted return to buy [0.0 ~ 1.0]')
        parser.add_argument('--buy_threshold_prob', type=float, default=0.6, 
                            help='The threshold of model\'s estimated probability for the predicted_return to be over buy_threshold_ret [0.0 ~ 1.0]')

        # forecasting task
        parser.add_argument('--label_len', type=int, default=96, help='start token length')
        parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
        parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4') # used only for M4 dataset

        # imputation task
        parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

        # anomaly detection task
        parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

        # Augmentation
        parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
        parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
        parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
        parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
        parser.add_argument('--permutation', default=False, action="store_true",
                            help="Equal Length Permutation preset augmentation")
        parser.add_argument('--randompermutation', default=False, action="store_true",
                            help="Random Length Permutation preset augmentation")
        parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
        parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
        parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
        parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
        parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
        parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
        parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
        parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
        parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
        parser.add_argument('--discdtw', default=False, action="store_true",
                            help="Discrimitive DTW warp preset augmentation")
        parser.add_argument('--discsdtw', default=False, action="store_true",
                            help="Discrimitive shapeDTW warp preset augmentation")
        parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

        # GPU
        parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        parser.add_argument('--gpu', type=int, default=0, help='gpu')
        parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
        parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False) # not working well
        parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        # basemodel arguments for adding to (or overriding) the common arguments
        parser.add_argument('--basemodel', action='append', type=_basemodel_args, default=[], 
                            help="name and arguments for a base model to be inlcuded in the combiner model [name --option1 val1 ...]")

        # Combiner arguments for adding or overriding 
        parser.add_argument('--combiner', type=lambda s: _model_args(s,'combiner'), default=None, 
                            help="arguments for the combiner model [--option1 val1 ...]")

        # Adjuster arguments for adding or overriding 
        parser.add_argument('--adjuster', type=lambda s: _model_args(s,'adjuster'), default=None, 
                            help="arguments for the adjuster model [--option1 val1 ...]")

    # Addable / Overidable by the model arguments ---------------------

    # seq_len (aka. context_len or lookback_win) of input (X) 
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description') # NOTE : for what?  
    parser.add_argument('--loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # common model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # CMamba
    parser.add_argument('--dt_rank', type=int, default=32)
    parser.add_argument('--patch_num', type=int, default=32)
    parser.add_argument('--d_state', type=int, default=16) 
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--dt_min', type=float, default=0.001)
    parser.add_argument('--dt_init', type=str, default='random', help='random or constant')
    parser.add_argument('--dt_max', type=float, default=0.1)
    parser.add_argument('--dt_scale', type=float, default=1.0)
    parser.add_argument('--dt_init_floor', type=float, default=1e-4)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--conv_bias', type=bool, default=True)
    parser.add_argument('--pscan', action='store_true', help='use parallel scan mode or sequential mode when training', default=True)
    parser.add_argument('--avg', action='store_true', help='avg pooling', default=True)
    parser.add_argument('--max', action='store_true', help='max pooling', default=True)
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--gddmlp', action='store_true', help='global data-dependent mlp', default=True)
    parser.add_argument('--channel_mixup', action='store_true', help='channel mixup', default=True)
    parser.add_argument('--sigma', type=float, default=1.0)

    # HPO (for Combiner, Adjuster)
    parser.add_argument('--hpo_policy', type=int, default=0, help="HPO policy. 0: no HPO, 1: once when training, 2: adaptive")
    parser.add_argument('--max_hpo_eval', type=int, default=200, 
                        help="max number of evaluation for a HPO [default: 200]")
    parser.add_argument('--hpo_interval', type=int, default=10, help="interval (timesteps >= 1) for Adaptive HPO")
    
    # Weighting models (for Combiner, Adjuster)
    parser.add_argument('--lookback_win', type=int, default=10, help="")
    parser.add_argument('--discount_factor', type=float, default=1.5, help="")
    parser.add_argument('--avg_method', type=int, default=0, help="")
    parser.add_argument('--weighting_method', type=int, default=2, help="")
    parser.add_argument('--scaling_factor', type=int, default=30, help="")
    parser.add_argument('--smoothing_factor', type=float, default=0.0, help="")
    parser.add_argument('--max_models', type=float, default=0.0, help="")
    
    # Adjuster
    parser.add_argument('--adj_model', type=str, default='moe', help='model used to adjust combiner\'s prediction. [gpm, moe]')
    # parser.add_argument('--adj_lookback_win', type=int, default=10, 
    #                     help="window size for the past predictions of combiner [5 ~ 50]"
    #                         "When 'adpative_hpo' applied, adj_lookback_win is adpatively changed")
    parser.add_argument('--gpm_kernel', type=str, default='Matern32', help='kernel of Gaussian Process [RBF, Matern32, Matern52, Linear, Brownian]')
    parser.add_argument('--gpm_noise', type=float, default=0.1, help='noise for Gaussian Process Kernel [0.0~]')
    parser.add_argument('--max_gp_opt_steps', type=int, default=5000, 
                        help="max number of optimization steps for the Gaussian Process model in the Adjuster [default: 5000]")
    parser.add_argument('--max_gp_opt_patience', type=int, default=30, 
                        help="max patience in the optimization steps without improvement [default: 30]")
    parser.add_argument('--quantile', type=float, default=0.975, 
                        help="quantile level for the probabilistic prediction in the Adjuster [default: 0.975]")

    # Etc 
    parser.add_argument('--use_batch_norm', type=bool, default=False)
    parser.add_argument('--prob_stat_win', type=int, default=50, 
                        help="window size for computing probabilistic statistics (larger than seq_len)")


    # If model_name is given, then all the default arguments are suppressed, and only explicitly given arguments are included
    if model_name is not None:
        for action in parser._actions:
            if action.dest != 'help':
                action.default = argparse.SUPPRESS

    return parser


def _set_device_configs(configs):
    if configs.use_gpu and torch.cuda.is_available():
        configs.gpu_type = 'cuda'
        if configs.use_multi_gpu:
            configs.devices = configs.devices.replace(' ', '')
            device_ids = configs.devices.split(',')
            configs.device_ids = [int(id_) for id_ in device_ids]
            configs.gpu = configs.device_ids[0]
        configs.device = torch.device('cuda:{}'.format(configs.gpu))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(configs.gpu) if not configs.use_multi_gpu else configs.devices
        logger.info('Use GPU: cuda:{}'.format(configs.gpu))
    elif configs.use_gpu and configs.gpu_type == 'mps' \
        and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS is not fully supported. It causes error in CMamba (and probably in TimeMoE)
        configs.gpu_type = 'mps' # If not set, it causes an error in Exp_Basic._acquire_device()
        configs.device = torch.device("mps")
        logger.info('Use GPU: mps')
    else:
        configs.use_gpu = False
        configs.device = torch.device("cpu")
        logger.info('Use CPU')


def _create_base_model(configs, model_name) -> AbstractModel:
    tslib_models = { # models in Time-Series-Library
        # 'TimesNet': TimesNet,
        'DLinear': DLinear,
        'PatchTST': PatchTST,
        'iTransformer': iTransformer,
        'TimeXer': TimeXer,
        # 'TSMixer': TSMixer

        # CMamba is not in TSLib, but follows the TSLib framework
        # So, we can treat it as a TSLib model
        'CMamba': CMamba  
    }
    other_models = {
        'TimesFM': TimesFM,
        'Timer': Timer,
        'TimeMoE': TimeMoE,
        'ETS': EtsModel,
        'SARIMA': SarimaModel,
        'AutoSARIMA': AutoSarimaModel,
        'Drifter': DrifterModel, 
        'Noiser': NoiseModel
    }

    if model_name in tslib_models:
        model = TSLibModel(configs, model_name, tslib_models[model_name])
    elif model_name in other_models:
        model = other_models[model_name](configs, model_name)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model


def _cleanup_gpu_cache(configs):
    if configs.gpu_type == 'mps':
        # Only if mps.empty_cache() is available, then call it
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
    elif configs.gpu_type == 'cuda':
        torch.cuda.empty_cache()


def _get_experiment_sig(configs):
    _experiment_signature =  f"{configs.model_id}_sl{configs.seq_len}"
    _experiment_signature += f"_ep{configs.train_epochs}_" 
    _experiment_signature += datetime.now().strftime("%Y%m%d_%H%M%S")
    return _experiment_signature


def _prepare_config(args):
    configs = _get_parser().parse_args(args)

    assert configs.label_len == 1
    assert configs.pred_len == 1
    assert configs.features != 'M', \
        "Combiner only supports feature type ['MS', 'S'], and label_len ==1 "

    if len(configs.basemodel) == 0:
        raise ValueError("At least one base model should be specified in the configuration.")

    _set_device_configs(configs)

    experiment_sig = _get_experiment_sig(configs)

    result_dir = configs.tabe_root_path + "/result/" + experiment_sig
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    configs.result_dir = result_dir

    checkpoints_dir = os.path.join(configs.tabe_root_path, configs.checkpoints, experiment_sig)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    configs.checkpoints = checkpoints_dir    

    return configs


def _calculate_warm_up_length(combiner_config, adjuster_config=None):
    length_for_combining = combiner_config.lookback_win if combiner_config.hpo_policy == 0 else CombinerModel.HPO_PERIOD
    warm_up_length = max(combiner_config.prob_stat_win, length_for_combining)
    if adjuster_config is not None:
        warm_up_length = max(warm_up_length, adjuster_config.lookback_win, adjuster_config.prob_stat_win)
    return warm_up_length


def create_tabe(args):
    configs = _prepare_config(args)

    # add file logging
    h_file = logging.FileHandler(configs.result_dir+'/tabe.log', mode='w')  
    h_file.setFormatter(default_formatter)
    h_file.setLevel(logging.DEBUG)
    logger.addHandler(h_file)

    print_configs(configs)

    # start memory tracking 
    _mem_util.start_python_memory_tracking()
    _mem_util.print_memory_usage()

    basemodels = []
    for (model_name, model_args) in configs.basemodel:
        bm_configs = configs
        if model_args is not None:
            bm_configs = copy.deepcopy(configs)
            bm_configs.__dict__.update(model_args.__dict__) # add/update with model-specific arguments
        basemodels.append(_create_base_model(bm_configs, model_name))
    
    combiner_configs = configs
    if configs.combiner is not None:
        combiner_configs = copy.deepcopy(configs)
        combiner_configs.__dict__.update(configs.combiner.__dict__) # add/update with model-specific arguments

    adjuster_configs = None
    if configs.adjuster is not None:
        adjuster_configs = copy.deepcopy(configs)
        adjuster_configs.__dict__.update(configs.adjuster.__dict__) # add/update with model-specific arguments

    # Calculate length to warm-up ensemble models in 'test' period.
    # This should be done after each model's configuration is set. 
    warm_up_length = _calculate_warm_up_length(combiner_configs, adjuster_configs)

    combiner_configs.warm_up_length = warm_up_length
    combinerModel = CombinerModel(combiner_configs, basemodels)

    if adjuster_configs is None:
        adjusterModel = None
    else:
        adjuster_configs.warm_up_length = warm_up_length
        assert configs.adj_model == 'moe', f"Model {configs.adj_model} is not supported as an Adjuster model"
        # if configs.adj_model == 'moe': 
        adjusterModel = TimeMoeAdjuster(adjuster_configs, combinerModel)

    configs.warm_up_length = warm_up_length    
    tabeModel = TabeModel(configs, combinerModel, adjusterModel)

    return tabeModel, combinerModel, basemodels
    

def destroy_tabe(tabeModel):
    _mem_util.print_memory_usage()
    _mem_util.stop_python_memory_tracking()
    _cleanup_gpu_cache(tabeModel.configs)


def _run(args=None):
    tabeModel, combinerModel, basemodels = create_tabe(args)

    logger.info('Training Tabe model ======================\n')
    tabeModel.train()    
    
    logger.info('Testing ==================================\n')
    tabeModel.test()

    logger.info('Reporting ==================================\n')

    report_results(tabeModel, combinerModel, basemodels)

    destroy_tabe(tabeModel)
    logger.info('Bye ~~~~~~')


if __name__ == '__main__':
    _run()
