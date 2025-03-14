
import os
import numpy as np
import random
import torch
import torch.nn as nn
from torch import optim
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw

from utils.tools import adjust_learning_rate, visual
from utils.metrics import metric

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

from tabe.utils.logger import logger
from tabe.utils.misc_util import OptimTracker
from tabe.data_provider.dataset_loader import get_data_provider
from tabe.models.abstractmodel import AbstractModel


warnings.filterwarnings('ignore')


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

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1
        endog = batch_y[0, :self.configs.seq_len, -1] # shape=(B,S+1,F) B(Batch Size)=1, S(Sequence Length)+1, F(Feature Dimension)
        endog = endog.numpy()
        pred = self._fit(endog).forecast(steps=1)
        truth = batch_y[0, -1, -1] 
        loss = self.criterion(torch.tensor(pred), truth).item()
        return pred[0], loss
    

class EtsModel(StatisticalModel):
    def __init__(self, configs, device=None, name="ETS"):
        super().__init__(configs, name) 

    def _fit(self, endog):
        # NOTE: Use auto-finding of the hyperparameters for ETS
        return ExponentialSmoothing(endog, trend='add', damped_trend=True).fit()


class SarimaModel(StatisticalModel):
    def __init__(self, configs, device=None, name="SARIMA"):
        super().__init__(configs, name) 

    def _fit(self, endog):
        # NOTE: Use auto-finding of the hyperparameters for SARIMA
        return SARIMAX(endog, order=(1,1,0), trend='ct', enforce_stationarity=False).fit(disp=False)


class AutoSarimaModel(StatisticalModel):
    def __init__(self, configs, device=None, name="AutoSARIMA"):
        super().__init__(configs, name) 

    def _fit(self, endog):
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
            suppress_warnings=True,
            # stepwise=True          # more efficient, stepwise approach
        )
        return self
    
    def forecast(self, steps=1):
        return self.model.predict(n_periods=steps)


# Written by referencing Time-Series-Library/exp_longterm_forecasting.py,exp_basic.py
class TSLibModel(AbstractModel):
    def __init__(self, configs, device, name, model):
        super().__init__(configs, name)
        self.device = device
        self.model = self._build_model(model).to(self.device)
        self.early_stopping = None

    def _build_model(self, model):
        model = model.Model(self.configs).float()
        if self.configs.use_multi_gpu and self.configs.use_gpu: 
            model = nn.DataParallel(model, device_ids=self.configs.device_ids)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        return model_optim

    def load_saved_model(self):
        path = self._get_checkpoint_path() + '/checkpoint.pth'
        logger.debug(f'loading saved model from {path}')
        self.model.load_state_dict(torch.load(path))

    def _forward_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        n_batch, n_vars = batch_y.shape[0], batch_y.shape[2] 
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if self.configs.channel_mixup:
            perm = torch.randperm(n_vars, device=self.device, dtype=torch.int32)
            # NOTE: 'mps' device raises an error when using 'torch.normal' function.
            mix_up = torch.normal(mean=0, std=self.configs.sigma, size=(n_batch, n_vars), 
                                  device=self.device, dtype=torch.float32).unsqueeze(-2)
            batch_x = batch_x + batch_x[:, :, perm] * mix_up
            batch_y = batch_y + batch_y[:, :, perm] * mix_up

        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.configs.label_len, :], dec_inp], dim=1).float().to(self.device)

        # encoder - decoder
        if self.configs.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        f_dim = -1 if self.configs.features == 'MS' else 0
        outputs = outputs[:, -self.configs.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)
        loss = self.criterion(outputs, batch_y)
        return loss, outputs



    # TODO : NOTE:  
    # When training is needed, only one epoch (for the given batch) is performed! 
    # And, validation is not performed.. 
    # Is it okay?? 
    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1

        # forward onestep, and get the prediction and loss
        self.model.eval()
        loss, outputs = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)
        loss = loss.item()
        pred = outputs[0, -1, -1].item()

        if training: 
            self._train_batch_with_validation(batch_x, batch_y, batch_x_mark, batch_y_mark)

        return pred, loss


    # NOTE 
    # Is it okay to train repeatitively with the same one batch?? 
    # Or, do we have to use all 'train period' datapoint? 
    def _train_batch_with_validation(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        vali_data, vali_loader = get_data_provider(self.configs, flag='val')

        time_now = time.time()

        if self.early_stopping is None :
            self.early_stopping = OptimTracker(patience=self.configs.patience, verbose=True)
        else:
            self.early_stopping.reset()

        model_optim = self._select_optimizer()

        if self.configs.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.configs.train_epochs):
            train_loss = []
            self.model.train()
            model_optim.zero_grad()

            loss, _ = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)
            train_loss.append(loss.item())

            if self.configs.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self._validate(vali_data, vali_loader)

            logger.debug(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")

            self.early_stopping(vali_loss, self.model, self._get_checkpoint_path())
            if self.early_stopping.early_stop:
                logger.debug("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.configs)

        logger.debug(f"_train_batch_with_validation: cost time: {time.time() - time_now:.3f}sec")

        # load the best model found
        self.load_saved_model() 


    def train(self):
        train_data, train_loader =  get_data_provider(self.configs, flag='base_train')
        vali_data, vali_loader = get_data_provider(self.configs, flag='val')

        time_now = time.time()

        train_steps = len(train_loader)
        if self.early_stopping is None :
            self.early_stopping = OptimTracker(patience=self.configs.patience, verbose=True)
        else:
            self.early_stopping.reset()

        model_optim = self._select_optimizer()

        if self.configs.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.configs.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                loss, _ = self._forward_onestep(batch_x, batch_y, batch_x_mark, batch_y_mark)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.debug("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.configs.train_epochs - epoch) * train_steps - i)
                    logger.debug('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.configs.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self._validate(vali_data, vali_loader)
            logger.debug(f"Epoch {epoch+1}, Train Loss: {train_loss:.6f} Vali Loss: {vali_loss:.6f}, Spent Time: {time.time() - epoch_time:.6f}")

            self.early_stopping(vali_loss, self.model, self._get_checkpoint_path())
            if self.early_stopping.early_stop:
                logger.debug("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.configs)

        # load the best model found
        self.load_saved_model() 
    

    # NOTE: 
    # The logic of this function is subordinate to train() function, 
    # and not intended to be used independently.
    def _validate(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.configs.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.configs.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.configs.features == 'MS' else 0
                outputs = outputs[:, -self.configs.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.configs.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = self.criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train() 
        return total_loss


    # NOTE
    # 'vali_data' is just used for evaluating the loss of the trained parameters.
    # So, test() can be called without 'vali_data' being used for train the model. 
    # Given the fact that test_data is actually those data that comes after vali_data, 
    # is it okay to exclude vali_data for training the model?
    def test(self, load_saved_model=False):
        if load_saved_model:
            self.load_saved_model()

        test_data, test_loader = get_data_provider(self.configs, flag='test')

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.configs.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.configs.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.configs.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.configs.features == 'MS' else 0
                outputs = outputs[:, -self.configs.pred_len:, :]
                batch_y = batch_y[:, -self.configs.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.configs.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.configs.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(self._get_result_path(), str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # logger.debug('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # logger.debug('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.configs.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    logger.debug("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        logger.info('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(self.setting_str + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # result_path = self._get_result_path()
        # np.save(result_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(result_path + 'pred.npy', preds)
        # np.save(result_path + 'true.npy', trues)
        return preds



# drift_factor :  +/- 10% = [-0.1 ~ 0.1] 
class DrifterModel(AbstractModel):
    def __init__(self, configs, drift_factor=0.1, duration=10, probability=0.1):
        super().__init__(configs, "Drifter") 
        self.drift_factor = drift_factor
        self.duration = duration
        self.probability = probability
        self.drift_occurred = False
        self.counter = 0

    def train(self):
        pass

    def load_saved_model(self):
        pass

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1
        truth = batch_y[0, -1, -1] 
        pred = truth
        if self.drift_occurred:
            if self.counter < self.duration:
                pred = truth * (1 + self.drift_factor * (1 + 0.1 * ((random.random() - 0.5) * 2))) 
                self.counter += 1
            else:
                self.drift_occurred = False
                self.counter = 0
        elif self.probability > random.random():
            self.drift_occurred = True
        loss = self.criterion(torch.tensor([pred]), truth).item()
        return pred, loss


# noise_factor : (0.0, 0.1)  = 0 ~ 10% of noise
class NoiseModel(AbstractModel):
    def __init__(self, configs, noise_factor=0.05, probability=0.01):
        super().__init__(configs, "Noiser") 
        self.noise_factor = noise_factor
        self.probability = probability

    def train(self):
        pass

    def load_saved_model(self):
        pass

    def proceed_onestep(self, batch_x, batch_y, batch_x_mark, batch_y_mark, training: bool = False):
        assert batch_x.shape[0]==1 and batch_y.shape[0]==1
        truth = batch_y[0, -1, -1] 
        pred = truth 
        if self.probability > random.random():
            pred = truth + self.noise_factor * ((random.random() - 0.5) * 2)
        loss = self.criterion(torch.tensor([pred]), truth).item()
        return pred, loss
    