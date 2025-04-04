
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pyro.contrib.gp as gp
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from utils.metrics import MAE, MSE, RMSE, MAPE, MSPE
from tabe.utils.trade_sim import simulate_trading
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


def plot_forecasts_with_deviations(truths, models, title=None, filepath=None):

    def _set_grid(ax, max_val, min_val):
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.margins(x=0.0, y=0.0)
        ax.tick_params(axis='both', labelsize=6)
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.axhline(0, color='black', linewidth=1.0)  
        height = max(abs(max_val), abs(min_val))
        ytick = 0.002
        num_ticks = int(np.ceil(height / ytick))
        ax.set_yticks(np.linspace(-num_ticks * ytick, num_ticks * ytick, num = 2*num_ticks+1))
        ax.set_ylim(-height - ytick, height + ytick)
        ax.set_autoscale_on(False)

    fig, (ax_pred, ax_devi) = plt.subplots(2, 1, figsize=(22, 12), sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.02, 'wspace': 0.0}
    )    
    ax_pred.set_title("Forecasts and deviations")
    ax_pred.set_xticks(np.arange(0, len(truths), step=1)) 

    # predictions subplot
    ax_pred.set_ylabel("Forecasts")
    ax_pred.plot(truths, label='GroundTruth', linewidth=1.5, color='black')
    max_pred, min_pred = np.max(truths), np.min(truths)
    for m in models:
        color = 'red' if m.name == 'Tabe' else ('blue' if m.name == 'Combiner' else None)
        linewidth = 1.5 if (m.name == 'Tabe' or m.name == 'Combiner') else 1
        linestype = '-' if (m.name == 'Tabe' or m.name == 'Combiner') else '--'        
        ax_pred.plot(m.result_predictions(), label=m.name, color=color, linewidth=linewidth, linestyle=linestype)
        if m.name == 'Combiner':
            ax_pred.fill_between(np.linspace(0, len(truths)-1, len(truths)), 
                m.result_dv_quantiles()[:,0], m.result_dv_quantiles()[:,1],
                color='red', alpha=0.1, label="area in quantiles")
        max_pred = max(max_pred, np.max(m.result_predictions()))
        min_pred = min(min_pred, np.min(m.result_predictions()))
    ax_pred.legend()    

    _set_grid(ax_pred, max_pred, min_pred)

    # deviations subplot
    ax_devi.set_ylabel('deviation')
    ax_devi.set_xlabel("Timesteps")
    max_devi, min_devi = -np.inf, np.inf
    for m in models:
        deviations = m.result_deviations()
        label = f'{m.name} [mae={np.mean(np.abs(deviations)):.3f}, mean={np.mean(deviations):.3f}, std={np.std(deviations):.3f}]'
        color = 'red' if m.name == 'Tabe' else ('blue' if m.name == 'Combiner' else None)
        linewidth = 1.5 if (m.name == 'Tabe' or m.name == 'Combiner') else 1
        linestype = '-' if (m.name == 'Tabe' or m.name == 'Combiner') else '--'        
        ax_devi.plot(deviations, label=label, color=color, linewidth=linewidth, linestyle=linestype)
        max_devi = max(max_devi, np.max(deviations))
        min_devi = min(min_devi, np.min(deviations))        
    ax_devi.legend()
    _set_grid(ax_devi, max_devi, min_devi)

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


# def plot_gpmodel(gpm, plot_observed_data=True, plot_predictions=True, n_test=500, x_range=None, filepath=None):
#     if x_range is None:
#         min = gpm.X.numpy().min()
#         max = gpm.X.numpy().max()
#         x_range = (min - abs(max-min)*0.1, max + abs(max-min)*0.1)

#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.set_title("Prediction Deviation Distribution (Analyzed in Gaussian Process)")
#     ax.set_ylabel("Prediction deviation at day t")
#     ax.set_xlabel("Prediction deviation at day t-1")

#     if plot_observed_data:
#         ax.plot(gpm.X.numpy(), gpm.y.numpy(), "kx", label="observations")

#     if plot_predictions:
#         Xtest = torch.linspace(x_range[0], x_range[1], n_test) 
#         # compute predictive mean and variance
#         with torch.no_grad():
#             if type(gpm) == gp.models.VariationalSparseGP:
#                 mean, cov = gpm(Xtest, full_cov=True)
#             else:
#                 mean, cov = gpm(Xtest, full_cov=True, noiseless=False)
#         sd = cov.diag().sqrt()  # standard deviation at each input point x
#         ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2, label="mean")  # plot the mean
#         ax.fill_between(
#             Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
#             (mean - 2.0 * sd).numpy(),
#             (mean + 2.0 * sd).numpy(),
#             color="C0",
#             alpha=0.3,
#             label="area in (-2σ, +2σ)"
#         )        
#     ax.legend()

#     # result_path = self._get_result_path()
#     if filepath is None:
#         plt.show()
#     else:
#         plt.savefig(filepath, bbox_inches='tight')


def _measure_loss(p,t):
    return MAE(p,t), MSE(p,t), RMSE(p,t), MAPE(p,t), MSPE(p,t)


def report_losses(truths, models):
    df = pd.DataFrame()
    for m in models:
        df[m.name] = _measure_loss(m.result_predictions(), truths)
    df.index = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
    print_dataframe(df, 'Model Losses')


def _measure_classifier_performance(truths, predictions, pred_probs,
        classification_method='up_down', value_threshold=(0.0), prob_threshold=(0.5)):
    """
    classification_method : 'up_down', 'up_down_sideway'
    value_threshold : (0.0), (0,002, -0.002)
    prob_threshold : (0.5) 
    """
    if classification_method == 'up_down': # (up=1, down=0)
        true_labels = (truths > value_threshold).astype(int) 
        # if prob_threshold is None:
        #     pred_labels = (predictions < value_threshold).astype(int) 
        # else: 
        #     pred_labels = ((predictions > value_threshold) & (pred_probs > prob_threshold)).astype(int) 
        if prob_threshold is None:
            pred_labels = (predictions > value_threshold).astype(int) 
        else: 
            pred_labels = ((predictions > value_threshold) & (pred_probs > prob_threshold)).astype(int)             
    else: # 'up_down_sideway' (up=1, sideway=0, down=-1)
        true_labels = np.zeros_like(truths, dtype=int)
        true_labels[truths > value_threshold[0]] = 1
        true_labels[truths < -value_threshold[1]] = -1
        pred_labels = np.zeros_like(predictions, dtype=int)
        pred_labels[predictions > value_threshold[0]] = 1
        pred_labels[predictions < -value_threshold[1]] = -1

    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    auc = roc_auc_score(true_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    return precision, recall, f1, auc, tn, fp, fn, tp


def report_classifier_performance(truths, models, buy_threshold_prob, filepath=None):
    # for cl_method in ['up_down', 'up_down_sideway']:
    df = pd.DataFrame() 
    for cl_method in ['up_down']:
        for val_threshold in [(0.0),(0.001),(0.002)]:
            for prob_threshold in [None, (0.5), (0.6), (0.7)]:
                for m in models:
                    df[m.name] = _measure_classifier_performance(truths, m.result_predictions(), m.result_prob_ascendings(), 
                                                            value_threshold=val_threshold, prob_threshold=prob_threshold, 
                                                            classification_method=cl_method)
                df.index = ['Precision', 'Recall', 'F1', 'AUC', 'tn', 'fp', 'fn', 'tp']
                print_dataframe(df, f'Classifier Performance [t_val:{val_threshold}, t_prob:{prob_threshold}]', filepath=filepath)


def _save_results(truths, models, filepath):
    df_fcst_result = pd.DataFrame() 
    df_fcst_result['Truths'] = truths
    for m in models:
        df_fcst_result[m.name] = m.result_predictions()
    df_fcst_result.to_csv(path_or_buf=filepath, index=False)


def report_results(tabe, combiner, basemodels):
    truths = tabe.dataset.get_labels()
    truths = truths[tabe.configs.warm_up_length:]   

    models = []
    if tabe.adjuster is not None:
        models += [tabe]
    models += [combiner] + basemodels

    for m in models:
        m.invert_result(tabe.dataset)

    report_losses(truths, models)

    report_classifier_performance(truths, models, tabe.configs.buy_threshold_prob)

    plot_forecasts_with_deviations(truths, models, filepath = tabe.configs.result_dir + "/models_forecast_comparison.pdf")

    plot_combiner_weights(combiner.weights_history, filepath = tabe.configs.result_dir + "/combiner_bm_weights.pdf")

    simulate_trading(tabe.configs, truths, models)

    _save_results(truths, models, tabe.configs.result_dir + "/forecast_results.csv")


