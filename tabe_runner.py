# ".env" file is used instead of the sys.path.append() method
# import sys
# Add Time-Series-Library directory to module lookup paths
# sys.path.append('Time-Series-Library')

import os 
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
import tabe.utils.report as report
from tabe.utils.misc_util import set_experiment_sig, experiment_sig, print_configs
from tabe.utils.logger import logger, default_formatter
from tabe.utils.trade_sim import simulate_trading


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
    
    # batch-normalization
    parser.add_argument('--use_batch_norm', type=bool, default=False)

    # Adjuster
    parser.add_argument('--adj_model', type=str, default='moe', help='model used to adjust combiner\'s prediction. [gpm, moe]')
    # parser.add_argument('--adj_lookback_win', type=int, default=10, 
    #                     help="window size for the past predictions of combiner [5 ~ 50]"
    #                         "When 'adpative_hpo' applied, adj_lookback_win is adpatively changed")
    parser.add_argument('--adj_stat_win', type=int, default=50, 
                        help="window size for computing probabilistic statistics (larger than seq_len)")
    parser.add_argument('--gpm_kernel', type=str, default='Matern32', help='kernel of Gaussian Process [RBF, Matern32, Matern52, Linear, Brownian]')
    parser.add_argument('--gpm_noise', type=float, default=0.1, help='noise for Gaussian Process Kernel [0.0~]')
    parser.add_argument('--max_gp_opt_steps', type=int, default=5000, 
                        help="max number of optimization steps for the Gaussian Process model in the Adjuster [default: 5000]")
    parser.add_argument('--max_gp_opt_patience', type=int, default=30, 
                        help="max patience in the optimization steps without improvement [default: 30]")
    parser.add_argument('--quantile', type=float, default=0.975, 
                        help="quantile level for the probabilistic prediction in the Adjuster [default: 0.975]")

    # If model_name is given, then all the default arguments are suppressed, and only explicitly given arguments are included
    if model_name is not None:
        for action in parser._actions:
            if action.dest != 'help':
                action.default = argparse.SUPPRESS

    return parser


def _set_device_configs(configs):
    if configs.use_gpu and torch.cuda.is_available():
        configs.device = torch.device('cuda:{}'.format(configs.gpu))
        configs.gpu_type = 'cuda' # by default
        logger.info('configured to use GPU')
    # MPS is not fully supported. It causes error in CMamba (and probably in TimeMoE)
    # elif configs.use_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #     configs.device = torch.device("mps")
    #     configs.gpu_type = 'mps' # If not set, it causes an error in Exp_Basic._acquire_device()
    #     logger.info('Using mps')
    else:
        configs.use_gpu = False
        configs.device = torch.device("cpu")
        logger.info('configured to use CPU')

    if configs.use_gpu and configs.use_multi_gpu:
        configs.devices = configs.devices.replace(' ', '')
        device_ids = configs.devices.split(',')
        configs.device_ids = [int(id_) for id_ in device_ids]
        configs.gpu = configs.device_ids[0]


def _acquire_device(configs):
    if configs.use_gpu and configs.gpu_type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = \
            str(configs.gpu) if not configs.use_multi_gpu else configs.devices
        device = torch.device('cuda:{}'.format(configs.gpu))
        logger.info('Use GPU: cuda:{}'.format(configs.gpu))
    elif configs.use_gpu and configs.gpu_type == 'mps':
        device = torch.device('mps')
        logger.info('Use GPU: mps')
    else:
        device = torch.device('cpu')
        logger.info('Use CPU')
    return device


def _create_base_model(configs, device, model_name) -> AbstractModel:
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
        model = TSLibModel(configs, device, model_name, tslib_models[model_name])
    elif model_name in other_models:
        model = other_models[model_name](configs, device, model_name)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model


def _report_results(configs, result_dir, tabe, combiner, basemodels):

    test_set = tabe.dataset  
    truths = test_set.data_y[-len(test_set):, -1]
    truths = truths[1:] # truths[0] is the thuth of the time step just before testing
    len_result = len(truths)
    
    tabe_preds = tabe.predictions[:-1] # exclude the last one, which does not have corresponding truth
    cbm_preds = combiner.predictions[:-1]
    bsm_preds = [bm.predictions[:-1] for bm in basemodels]    

    need_to_invert_data = True if test_set.scale else False
    if need_to_invert_data:            
        n_features = test_set.data_y.shape[1]
        data_truths = np.zeros((len_result, n_features))
        data_tabe_preds = np.zeros((len_result, n_features))
        data_cbm_preds = np.zeros((len_result, n_features))
        data_truths[:, -1] = truths
        data_tabe_preds[:, -1] = tabe_preds
        data_cbm_preds[:, -1] = cbm_preds
        truths = test_set.inverse_transform(data_truths)[:, -1]
        tabe_preds = test_set.inverse_transform(data_tabe_preds)[:, -1]
        cbm_preds = test_set.inverse_transform(data_cbm_preds)[:, -1]
        for i in range(len(bsm_preds)):
            data_bsm_preds = np.zeros((len_result, n_features))
            data_bsm_preds[:, -1] = bsm_preds[i]
            bsm_preds[i] = test_set.inverse_transform(data_bsm_preds)[:, -1]

    report.report_losses(truths, tabe_preds, cbm_preds, bsm_preds, basemodels)
    report.report_classifier_performance(truths, tabe_preds, cbm_preds, bsm_preds, basemodels)

    report.plot_forecasts_with_deviations(truths, tabe_preds,  cbm_preds, 
                                        bsm_preds, [bm.name for bm in basemodels], 
                                        tabe.adjuster.adj_deviations[-len_result:], tabe.adjuster.cbm_deviations[-len_result:],
                                        filepath = result_dir + "/models_forecast_comparison.pdf")

    # report.plot_forecast_result(truths, tabe_preds,  None, None, cbm_preds, bsm_preds, basemodels,
    #                         filepath = result_dir + "/models_forecast_comparison.pdf")
    # report.plot_deviations_over_time(tabe.adjuster.cbm_deviations[-len_result:], 
    #                                  filepath = result_dir + "/combiner_deviations_over_time.pdf")
    report.plot_combiner_weights(tabe.combiner.weights_history, 
                                 filepath = result_dir + "/combiner_bm_weights.pdf")

    # save forecast result
    df_fcst_result = pd.DataFrame() 
    df_fcst_result['Truths'] = truths
    df_fcst_result['Adjuster'] = tabe_preds
    # df_fcst_result[f'Adjuster_q_low_{configs.quantile}'] = y_hat_q_low
    # df_fcst_result[f'Adjuster_q_high_{configs.quantile}'] = y_hat_q_high
    # df_fcst_result['Adjuster_devi_sd'] = devi_stddev
    if cbm_preds is not None:
        df_fcst_result['Combiner'] = cbm_preds
    if bsm_preds is not None:
        for i, bm in enumerate(basemodels):
            df_fcst_result[bm.name] = bsm_preds[i]
    df_fcst_result.to_csv(path_or_buf=result_dir + "/forecast_results.csv", index=False)

    # Trading Simulations ---------------

    target = configs.target
    if target in ['LogRet1',]:
        # Convert data to 'Ret'
        truths = np.exp(truths) - 1
        tabe_preds = np.exp(tabe_preds) - 1
        # y_hat_q_low = np.exp(y_hat_q_low) - 1
        if cbm_preds is not None:
            cbm_preds = np.exp(cbm_preds) - 1
        if bsm_preds is not None:
            for i in range(len(bsm_preds)):
                bsm_preds[i] = np.exp(bsm_preds[i]) - 1

        # Simulation with 'Ret'
        # for apply_threshold_prob in [False, True]:
        for apply_threshold_prob in [False]:
            for strategy in ['buy_and_hold', 'daily_buy_sell', 'buy_hold_sell_v1', 'buy_hold_sell_v2']:
                df_sim_result = pd.DataFrame() 
                df_sim_result['Tabe'] = simulate_trading(truths, tabe_preds, strategy, None, apply_threshold_prob, 
                                                             buy_threshold=configs.buy_threshold_ret, buy_threshold_q=None, fee_rate=configs.fee_rate)
                if cbm_preds is not None:
                    df_sim_result['Combiner'] = simulate_trading(truths, cbm_preds, strategy, None, apply_threshold_prob,
                                                             buy_threshold=configs.buy_threshold_ret, buy_threshold_q=None, fee_rate=configs.fee_rate)
                if bsm_preds is not None:
                    for i, bm in enumerate(basemodels):
                        df_sim_result[bm.name] = simulate_trading(truths, bsm_preds[i], strategy=strategy,
                                                             buy_threshold=configs.buy_threshold_ret, buy_threshold_q=None, fee_rate=configs.fee_rate)
                df_sim_result.index = ['Acc. ROI', 'Mean ROI', '# Trades', '# Win_Trades', 'Winning Rate']
                report.report_trading_simulation(df_sim_result, 
                                                 strategy+'_prob' if apply_threshold_prob else strategy, 
                                                 len_result)
    else:
        logger.warning(f"Trading simulation for target {target} is not supported now.")


def _cleanup_gpu_cache(configs):
    if configs.gpu_type == 'mps':
        # Only if mps.empty_cache() is available, then call it
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
    elif configs.gpu_type == 'cuda':
        torch.cuda.empty_cache()


def create_tabe(args):
    _set_seed()

    configs = _get_parser().parse_args(args)

    assert configs.label_len == 1
    assert configs.pred_len == 1
    assert configs.features != 'M', \
        "Combiner only supports feature type ['MS', 'S'], and label_len ==1 "

    if len(configs.basemodel) == 0:
        raise ValueError("At least one base model should be specified in the configuration.")

    _set_device_configs(configs)
    set_experiment_sig(configs)

    result_dir = configs.tabe_root_path + "/result/" + experiment_sig()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # add file logging
    h_file = logging.FileHandler(result_dir+'/tabe.log', mode='w')  
    h_file.setFormatter(default_formatter)
    h_file.setLevel(logging.DEBUG)
    logger.addHandler(h_file)

    print_configs(configs)

    # start memory tracking 
    _mem_util.start_python_memory_tracking()
    _mem_util.print_memory_usage()

    device = _acquire_device(configs)

    basemodels = []
    for (model_name, model_args) in configs.basemodel:
        bm_configs = configs
        if model_args is not None:
            bm_configs = copy.deepcopy(configs)
            bm_configs.__dict__.update(model_args.__dict__) # add/update with model-specific arguments
        basemodels.append(_create_base_model(bm_configs, device, model_name))
    
    combiner_configs = configs
    if configs.combiner is not None:
        combiner_configs = copy.deepcopy(configs)
        combiner_configs.__dict__.update(configs.combiner.__dict__) # add/update with model-specific arguments
    combinerModel = CombinerModel(combiner_configs, device, basemodels)

    if configs.adjuster is None:
        adjusterModel = None
    else:
        adjuster_configs = copy.deepcopy(configs)
        adjuster_configs.__dict__.update(configs.adjuster.__dict__) # add/update with model-specific arguments
        if configs.adj_model == 'moe':            
            adjusterModel = TimeMoeAdjuster(adjuster_configs, device)
        elif configs.adj == 'gpm':            
            adjusterModel = GpmAdjuster(adjuster_configs)
        else:
            raise ValueError(f"Model {configs.adj_model} is not supported as an Adjuster model")

    tabeModel = TabeModel(configs, device, combinerModel, adjusterModel)
    return tabeModel, combinerModel, basemodels, result_dir
    

def destroy_tabe(tabeModel):
    _mem_util.print_memory_usage()
    _mem_util.stop_python_memory_tracking()
    _cleanup_gpu_cache(tabeModel.configs)


def _run(args=None):
    tabeModel, combinerModel, basemodels, result_dir = create_tabe(args)

    logger.info('Training Tabe model ======================\n')
    tabeModel.train()    
    
    logger.info('Testing ==================================\n')
    tabeModel.test()

    logger.info('Reporting ==================================\n')

    _report_results(tabeModel.configs, result_dir, tabeModel, combinerModel, basemodels)

    destroy_tabe(tabeModel)
    logger.info('Bye ~~~~~~')


if __name__ == '__main__':
    _run()
