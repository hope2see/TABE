
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


def plot_forecasts_with_deviations(truth, adj_pred, cbm_pred, bm_preds, bm_names, 
                                   adj_devi, cbm_devi, title=None, filepath=None):
    num_points = len(truth)
    assert len(adj_devi) == num_points
    assert len(cbm_devi) == num_points

    fig, (ax_pred, ax_devi) = plt.subplots(2, 1, figsize=(22, 12), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.02, 'wspace': 0.0}
    )    
    ax_pred.set_title("Forecasts and deviations")

    # Forecasts subplot
    ax_pred.set_ylabel("Forecasts")
    ax_pred.plot(truth, label='GroundTruth', linewidth=1.5, color='black')
    ax_pred.plot(adj_pred, label='Adjuster', linewidth=1.5, color='red')
    ax_pred.plot(cbm_pred, label='Combiner', linewidth=1.5, color='blue')
    for i, pred in enumerate(bm_preds):
        ax_pred.plot(pred, label=bm_names[i], linewidth=1, linestyle="--")
    ax_pred.legend()
    ax_pred.set_xticks(np.arange(0, num_points, step=1)) 
    ax_pred.margins(x=0.0, y=0.0)
    ax_pred.tick_params(axis='both', labelsize=6)
    ax_pred.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax_pred.axhline(0, color='black', linewidth=1.0)  
    height = max(abs(np.min(truth)), abs(np.max(truth))) 
    ytick = 0.002
    num_ticks = int(np.ceil(height / ytick))
    ax_pred.set_yticks(np.linspace(-num_ticks * ytick, num_ticks * ytick, num=2 * num_ticks + 1))
    ax_pred.set_ylim(-height - ytick, height + ytick)
    ax_pred.set_autoscale_on(False)

    # deviations subplot
    ax_devi.set_ylabel('deviation')
    ax_devi.set_xlabel("Timesteps")
    adj_devi_label = f'Adjuster [mae={np.mean(np.abs(adj_devi)):.3f}, mean={np.mean(adj_devi):.3f}, std={np.std(adj_devi):.3f}]'
    ax_devi.plot(adj_devi, label=adj_devi_label, linewidth=1, color='red')
    cbm_devi_label = f'Combiner [mae={np.mean(np.abs(cbm_devi)):.3f}, mean={np.mean(cbm_devi):.3f}, std={np.std(cbm_devi):.3f}]'
    ax_devi.plot(cbm_devi, label=cbm_devi_label, linewidth=1, color='blue')
    ax_devi.legend()
    ax_devi.grid(True, which='both', linestyle='--', linewidth=0.7)
    ax_devi.axhline(0, color='black', linewidth=1.0)  
    ax_devi.margins(x=0.0, y=0.0)
    ax_devi.tick_params(axis='both', labelsize=6)
    height = max(abs(np.min(cbm_devi)), abs(np.max(cbm_devi))) 
    ytick = 0.2
    num_ticks = int(np.ceil(height / ytick))
    ax_devi.set_yticks(np.linspace(-num_ticks * ytick, num_ticks * ytick, num=2 * num_ticks + 1))
    ax_devi.set_ylim(-height - ytick, height + ytick)
    ax_devi.set_autoscale_on(False)

    plt.show() if filepath is None else plt.savefig(filepath, bbox_inches='tight')


def plot_forecast_result(truth, pred,  adj_pred_q_low=None, adj_pred_q_high=None, combiner_pred=None, base_preds=None, basemodels=None, filepath=None):
    plt.figure(figsize=(20, 10))
    plt.title('Forecast Comparison')     
    plt.ylabel('Target')
    plt.xlabel('Test Duration (Days)')

    plt.plot(truth, label='GroundTruth', linewidth=1.5, color='black')
    plt.plot(pred, label="Tabe Model", linewidth=1.5, color='red')
    if adj_pred_q_low is not None:
        plt.fill_between(
            np.linspace(0, len(truth)-1, len(truth)), 
            adj_pred_q_low, adj_pred_q_high,
            color='red', alpha=0.1,
            label="area in quantiles"
        )                
    if combiner_pred is not None:
        plt.plot(combiner_pred, label="Combiner Model", linewidth=1.5, linestyle="--", color='blue')
    if base_preds is not None:
        for i, basemodel in enumerate(basemodels):
            plt.plot(base_preds[i], label=f"Base Model [{basemodel.name}]", linewidth=1.5, linestyle=":")
    plt.legend()

    plt.xticks(np.arange(0, len(truth), step=1)) 
    height = max(abs(np.min(truth)), abs(np.max(truth)))
    plt.yticks(np.arange(-height-0.002, height+0.002, 0.002))
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.gca().axhline(0, color='black', linewidth=1.0)  
    plt.tick_params(axis='both', labelsize=5)
    plt.show() if filepath is None else plt.savefig(filepath, bbox_inches='tight')



def plot_deviations_over_time(deviations, filepath=None):
    plt.figure(figsize=(20, 5))
    plt.title(f'Combiner Deviations [mean= {np.mean(deviations):.3f}, std={np.std(deviations):.3f}]') 
    plt.ylabel('deviation')
    plt.xlabel('timestep')
    plt.plot(deviations, label='deviations', linewidth=1.5, color='blue')
    
    num_points = len(deviations)
    plt.xticks(np.arange(0, num_points, step=1))  
    height = max(abs(np.min(deviations)), abs(np.max(deviations)))
    plt.yticks(np.arange(-height-0.1, height+0.1, 0.1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.gca().axhline(0, color='black', linewidth=1.0)  
    plt.tick_params(axis='both', labelsize=5)
    plt.show() if filepath is None else plt.savefig(filepath, bbox_inches='tight')    


def plot_combiner_weights(weights_history, filepath=None):
    plt.figure(figsize=(20, 10))
    for i in range(weights_history.shape[0]):
        plt.plot(weights_history[i,:], label=f'Component {i}')
    plt.title('Combiner Component Weights')
    plt.ylabel("Weights")
    plt.xlabel("Time")
    plt.legend()    

    plt.xticks(np.arange(0, weights_history.shape[1], step=1)) 
    plt.yticks(np.arange(0, 1, 0.05))
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    plt.gca().axhline(0.5, color='black', linewidth=1.0)  
    plt.tick_params(axis='both', labelsize=5)

    plt.show() if filepath is None else plt.savefig(filepath, bbox_inches='tight')


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

def report_losses(y, y_hat, y_hat_cbm=None, y_hat_bsm=None, basemodels=None):
    df = pd.DataFrame()
    df['Tabe'] = _measure_loss(y_hat, y)
    if y_hat_cbm is not None:
        df['Combiner'] = _measure_loss(y_hat_cbm, y)
    if y_hat_bsm is not None:
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


def report_classifier_performance(y, y_hat, y_hat_cbm=None, y_hat_bsm=None, basemodels=None, filepath=None):
    # for cl_method in ['up_down', 'up_down_sideway']:
    for cl_method in ['up_down']:
        df = pd.DataFrame() 
        df['Tabe'] = _measure_classifier_performance(y, y_hat, cl_method)
        if y_hat_cbm is not None:
            df['Combiner'] = _measure_classifier_performance(y, y_hat_cbm, cl_method)
        if y_hat_bsm is not None:
            for i, bm in enumerate(basemodels):
                df[bm.name] = _measure_classifier_performance(y, y_hat_bsm[i], cl_method)
        df.index = ['Precision', 'Recall', 'F1', 'AUC']
    print_dataframe(df, 'Classifier Performance', filepath=filepath)


def report_trading_simulation(df, strategy, days, filepath=None):
    title = f"[ Trading Simulation Results: (Strategy:{strategy}, Days:{days} ]"
    print_dataframe(df, title, filepath=filepath)
