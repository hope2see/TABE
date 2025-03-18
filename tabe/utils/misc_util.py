
import numpy as np
import torch
import time 
from datetime import datetime
from tabe.utils.logger import logger
from tabe.utils.losses import mape_loss, smape_loss, mase_loss
import torch.nn as nn


_experiment_signature = None
def set_experiment_sig(configs):
    global _experiment_signature
    _experiment_signature =  f"{configs.model_id}_sl{configs.seq_len}"
    _experiment_signature += f"_ep{configs.train_epochs}_" 
    _experiment_signature += datetime.now().strftime("%Y%m%d_%H%M%S")

def experiment_sig():
    assert _experiment_signature is not None
    return _experiment_signature

#  -----------------------------------------------------

def get_loss_func(loss:str):
    loss = loss.upper()
    if loss == 'MSE':
        return nn.MSELoss()
    elif loss == 'MAE':
        return nn.L1Loss()
    elif loss == 'MAPE':
        return mape_loss
    elif loss == 'SMAPE':
        return smape_loss
    elif loss == 'MASE':
        return mase_loss
    else:
        logger.error('Not suppprted loss : {loss}')
        raise NotImplementedError

#  -----------------------------------------------------

class OptimTracker:
    def __init__(self, use_early_stop=True, patience=7, verbose=False, delta=0, save_to_file=True):
        self.use_early_stop = use_early_stop
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.save_to_file = save_to_file
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf 

    def __call__(self, val_loss, model, path=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            if self.use_early_stop:
                self.counter += 1
                # logger.debug(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    logger.debug(f'EarlyStopping early_stop triggered!')
                    self.early_stop = True
            else:
                pass
        else: 
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(val_loss, model, path)
        return self.early_stop

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.debug(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        if self.save_to_file:
            torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        else:
            self.best_model = model

    # In order to do 'continous train()', we should keep tracking best_model. 
    # We use one EarlyStopping instance to track best_model while resetting it when needed. 
    # So, self.best_score should not be reset
    def reset(self):
        self.early_stop = False
        self.counter = 0



# -----------

def _args_to_str(args):
    if args is None:
        return ""
    return " ".join(f"{k}={v}" for k, v in vars(args).items())

def print_configs(configs):
    logger.info(f"\033[1m" + "TABE Config" + "\033[0m")
    logger.info(f'  {"Model:":<20}{configs.model:<20}{"Model ID:":<20}{configs.model_id:<20}')
    # logger.info(f'  {"Task Name:":<20}{configs.task_name:<20}{"Is Training:":<20}{configs.is_training:<20}')
    logger.info('')

    logger.info(f"\033[1m" + "Adjuster" + "\033[0m")
    logger.info(f'  {"gpm_lookback_win:":<20}{configs.gpm_lookback_win:<20}{"max_gp_opt_steps:":<20}{configs.max_gp_opt_steps}')
    logger.info(f'  {"gpm_kernel:":<20}{configs.gpm_kernel:<20}{"gpm_noise:":<20}{configs.gpm_noise}')
    logger.info(f'  {"Adjuster args:":<12}{_args_to_str(configs.adjuster):<20}')
    logger.info('')

    logger.info(f"\033[1m" + "Combiner" + "\033[0m")
    logger.info(f'  {"Combiner args:":<12}{_args_to_str(configs.combiner):<20}')
    logger.info('')

    logger.info(f"\033[1m" + "Base Models" + "\033[0m")
    for (model_name, model_args) in configs.basemodel:
        logger.info(f'   {model_name+" args:":<12}{_args_to_str(model_args):<20}')
    logger.info('')

    logger.info("\033[1m" + f"Data {configs.data:<20}" + "\033[0m")
    train_splits = ', '.join(map(str, configs.data_train_splits))   
    logger.info(f'  {"Split [val | train | ensemble_train]: ":<20}{train_splits:<20}{"Test: ":<20}{configs.data_test_split:<20}')
    logger.info(f'  {"Features:":<20}{configs.features:<20}{"Target Column:":<20}{configs.target:<20}')
    logger.info(f'  {"Target datatype:":<20}{configs.target_datatype:<20}{"Target Freq:":<20}{configs.freq:<20}')
    if configs.data == 'TABE_ONLINE':
        logger.info(f'  {"Asset:":<20}{configs.data_asset:<20}{"interval:":<20}{configs.data_interval:<20}')
        logger.info(f'  {"start_date:":<20}{configs.data_start_date:<20}{"end_date:":<20}{configs.data_end_date:<20}')
    else:
        logger.info(f'  {"Root Path:":<20}{configs.root_path:<20}{"Data Path:":<20}{configs.data_path:<20}')
    logger.info('')

    logger.info(f"\033[1m" + f"HPO Policy = {configs.hpo_policy}" + "\033[0m")
    if configs.hpo_policy == 2:
        logger.info(f'  {"interval:":<20}{configs.hpo_interval:<20}{"max_hpo_eval:":<20}{configs.max_hpo_eval:<20}')
    logger.info('')

    if configs.task_name in ['long_term_forecast', 'short_term_forecast']:
        logger.info("\033[1m" + "Forecasting Task" + "\033[0m")
        logger.info(f'  {"Seq Len:":<20}{configs.seq_len:<20}{"Label Len:":<20}{configs.label_len:<20}')
        logger.info(f'  {"Pred Len:":<20}{configs.pred_len:<20}{"Inverse:":<20}{configs.inverse:<20}')
        logger.info('')

    # if configs.task_name == 'imputation':
    #     logger.info("\033[1m" + "Imputation Task" + "\033[0m")
    #     logger.info(f'  {"Mask Rate:":<20}{configs.mask_rate:<20}')
    #     logger.info()

    # if configs.task_name == 'anomaly_detection':
    #     logger.info("\033[1m" + "Anomaly Detection Task" + "\033[0m")
    #     logger.info(f'  {"Anomaly Ratio:":<20}{configs.anomaly_ratio:<20}')
    #     logger.info()


    logger.info("\033[1m" + "Model Parameters" + "\033[0m")
    logger.info(f'  {"Top k:":<20}{configs.top_k:<20}{"Num Kernels:":<20}{configs.num_kernels:<20}')
    logger.info(f'  {"Enc In:":<20}{configs.enc_in:<20}{"Dec In:":<20}{configs.dec_in:<20}')
    logger.info(f'  {"C Out:":<20}{configs.c_out:<20}{"d model:":<20}{configs.d_model:<20}')
    logger.info(f'  {"n heads:":<20}{configs.n_heads:<20}{"e layers:":<20}{configs.e_layers:<20}')
    logger.info(f'  {"d layers:":<20}{configs.d_layers:<20}{"d FF:":<20}{configs.d_ff:<20}')
    logger.info(f'  {"Moving Avg:":<20}{configs.moving_avg:<20}{"Factor:":<20}{configs.factor:<20}')
    logger.info(f'  {"Distil:":<20}{configs.distil:<20}{"Dropout:":<20}{configs.dropout:<20}')
    logger.info(f'  {"Embed:":<20}{configs.embed:<20}{"Activation:":<20}{configs.activation:<20}')
    logger.info('')

    logger.info("\033[1m" + "Run Parameters" + "\033[0m")
    logger.info(f'  {"Num Workers:":<20}{configs.num_workers:<20}{"Itr:":<20}{configs.itr:<20}')
    logger.info(f'  {"Train Epochs:":<20}{configs.train_epochs:<20}{"Batch Size:":<20}{configs.batch_size:<20}')
    logger.info(f'  {"Patience:":<20}{configs.patience:<20}{"Learning Rate:":<20}{configs.learning_rate:<20}')
    logger.info(f'  {"Des:":<20}{configs.des:<20}{"Loss:":<20}{configs.loss:<20}')
    logger.info(f'  {"Lradj:":<20}{configs.lradj:<20}{"Use Amp:":<20}{configs.use_amp:<20}')
    logger.info('')

    logger.info("\033[1m" + "GPU" + "\033[0m")
    logger.info(f'  {"Use GPU:":<20}{configs.use_gpu:<20}{"GPU:":<20}{configs.gpu:<20}')
    logger.info(f'  {"Use Multi GPU:":<20}{configs.use_multi_gpu:<20}{"Devices:":<20}{configs.devices:<20}')
    logger.info('')

    # logger.info("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    # p_hidden_dims_str = ', '.join(map(str, configs.p_hidden_dims))
    # logger.info(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{configs.p_hidden_layers:<20}') 
    # logger.info()
