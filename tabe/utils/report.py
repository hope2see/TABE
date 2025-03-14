
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pyro.contrib.gp as gp
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE
from tabe.utils.logger import logger



def print_dataframe(df, title, print_index=True, filepath=None):
    buffer = io.StringIO()
    print('\n'+title, file=buffer)
    print(df.to_string(index=print_index), file=buffer)
    logger.info(buffer.getvalue())
    if filepath is not None:
        f = open(filepath, 'w')
        f.write(buffer.getvalue())
        f.close()

def print_dict(dt, title):
    df = pd.DataFrame(dt,index=[0])
    print_dataframe(df, title, print_index=False)


# Plot the optimization progress (loss and loss variance) 
def plot_hpo_result(trials, title='HyperParameter Optimization', filepath=None):
    losses = [t['result']['loss'] for t in trials]           
    # variances = [t['result']['loss_variance'] for t in trials]
    trial_numbers = np.arange(1, len(losses)+1)

    plt.figure(figsize=(8, 5))
    plt.plot(trial_numbers, losses, marker='o')
    # plt.plot(trial_numbers, variances, marker='x', label='Loss Variance')
    plt.xlabel("Trial Number")
    plt.ylabel("Mean Loss")
    plt.legend()    
    plt.grid(True)
    if title:
        plt.title(title)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')


def plot_forecast(y, y_hat, title=None, filepath=None):
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame({
        'Ground Truth': y,
        'Forecast': y_hat
    })
    df.plot()
    if title:
        plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()    
    plt.grid(True)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')


def plot_forecast_result(truth, adjuster_pred,  adj_pred_q_low, adj_pred_q_high, combiner_pred, base_preds, basemodels, filepath=None):
    plt.figure(figsize=(12, 6))
    plt.title('Forecast Comparison')     
    plt.ylabel('Target')
    plt.xlabel('Test Duration (Days)')
    plt.plot(truth, label='GroundTruth', linewidth=1.5, color='black')
    plt.plot(adjuster_pred, label="Adjuster Model", linewidth=1.5, color='red')
    if adj_pred_q_low is not None:
        plt.fill_between(
            np.linspace(0, len(truth)-1, len(truth)), 
            adj_pred_q_low, adj_pred_q_high,
            color='red', alpha=0.1,
            label="area in quantiles"
        )                
    plt.plot(combiner_pred, label="Combiner Model", linewidth=1.5, linestyle="--", color='blue')
    for i, basemodel in enumerate(basemodels):
        plt.plot(base_preds[i], label=f"Base Model [{basemodel.name}]", linewidth=1.5, linestyle=":")
    plt.legend()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')


def plot_weights(weights_hist, title=None, filepath=None):
    plt.figure(figsize=(8, 5))
    for i in range(weights_hist.shape[1]):
        plt.plot(weights_hist[:,i], label=f'Component {i}')
    if title:
        plt.title(title)
    plt.ylim(0, 1)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()    
    plt.grid(True)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')



def plot_gpmodel(gpm, plot_observed_data=True, plot_predictions=True, n_test=500, x_range=None, filepath=None):
    if x_range is None:
        min = gpm.X.numpy().min()
        max = gpm.X.numpy().max()
        x_range = (min - abs(max-min)*0.1, max + abs(max-min)*0.1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Prediction Deviation Distribution (Analyzed in Gaussian Process)")
    ax.set_ylabel("Prediction deviation at day t")
    ax.set_xlabel("Prediction deviation at day t-1")

    if plot_observed_data:
        ax.plot(gpm.X.numpy(), gpm.y.numpy(), "kx", label="observations")

    if plot_predictions:
        Xtest = torch.linspace(x_range[0], x_range[1], n_test) 
        # compute predictive mean and variance
        with torch.no_grad():
            if type(gpm) == gp.models.VariationalSparseGP:
                mean, cov = gpm(Xtest, full_cov=True)
            else:
                mean, cov = gpm(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2, label="mean")  # plot the mean
        ax.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
            label="area in (-2σ, +2σ)"
        )        
    ax.legend()

    # result_path = self._get_result_path()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight')


def _measure_loss(p,t):
    return MAE(p,t), MSE(p,t), RMSE(p,t), MAPE(p,t), MSPE(p,t)

def report_losses(y, y_hat_adj, y_hat_cbm, y_hat_bsm, basemodels):
    df = pd.DataFrame()
    df['Adjuster'] = _measure_loss(y_hat_adj, y)
    df['Combiner'] = _measure_loss(y_hat_cbm, y)
    for i, bm in enumerate(basemodels):
        df[bm.name] = _measure_loss(y_hat_bsm[i], y)
    df.index = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
    print_dataframe(df, 'Model Losses')


def _measure_classifier_performance(truths, predictions, classification_method='up_down', threshold=0.005):
    if classification_method == 'up_down': # (1,0)
        true_labels = (truths > 0.0).astype(int) 
        pred_labels = (predictions > 0.0).astype(int) 
    else: # 'up_down_sideway' (1, -1, 0)
        true_labels = np.zeros_like(truths, dtype=int)
        true_labels[truths > threshold] = 1
        true_labels[truths < -threshold] = -1
        pred_labels = np.zeros_like(predictions, dtype=int)
        pred_labels[predictions > threshold] = 1
        pred_labels[predictions < -threshold] = -1
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, predictions)
    return precision, recall, f1, auc


def report_classifier_performance(y, y_hat_adj, y_hat_cbm, y_hat_bsm, basemodels, filepath=None):
    # for cl_method in ['up_down', 'up_down_sideway']:
    for cl_method in ['up_down']:
        df = pd.DataFrame() 
        df['Adjuster'] = _measure_classifier_performance(y, y_hat_adj, cl_method)
        df['Combiner'] = _measure_classifier_performance(y, y_hat_cbm, cl_method)
        for i, bm in enumerate(basemodels):
            df[bm.name] = _measure_classifier_performance(y, y_hat_bsm[i], cl_method)
        df.index = ['Precision', 'Recall', 'F1', 'AUC']
    print_dataframe(df, 'Classifier Performance', filepath=filepath)


def report_trading_simulation(df, strategy, days, filepath=None):
    title = f"[ Trading Simulation Results: (Strategy:{strategy}, Days:{days} ]"
    print_dataframe(df, title, filepath=filepath)
