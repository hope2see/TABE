import numpy as np


class AvgMethod:
    MEAN = 0
    MEAN_SQUARED = 1

class WeightingMethod:
    INVERTED = 0
    SQUARED_INVERTED = 1
    SOFTMAX = 2


# NOTE : cool_start factor is ignored.  
# TODO: Need to give rational

def compute_model_weights(
    model_losses, 
    prev_model_weights, 
    lookback_window = 2, 
    discount_factor = 1.0, # for exponentially discounting losses by time distance. [1,1.5] The higher value, the more discounting.
    avg_method = AvgMethod.MEAN, 
    weighting_method = WeightingMethod.SOFTMAX, 
    softmax_scaling_factor = 1.0, # 1 or higher. 1 means no scaling. The higher value generates more diverged weights for small loss differences.
    smoothing_factor = 0.0, # To make weights change smoothly in time axis. [0,1)  Higher value makes smoother change.  
    max_models = None # None or integer. maximum number of models for weighting. 
):
    num_all_models = model_losses.shape[0]    
    model_losses = model_losses[:, -lookback_window:]

    epsilon = 1e-10 # tiny val to prevent dividision by zero, etc. 

    # Normalize to 'percentage losses' that is relative losses compared to other models' losses at the same timestep. 
    model_losses = np.abs(model_losses)
    model_losses = np.divide(model_losses, np.sum(model_losses, axis=0)+epsilon)
    model_losses += epsilon # to avoid the case that any loss is 0 ! (ex: synthetic model)


    # apply discounting to error_matrix, and compute the component_error
    try:
        discount = np.power(discount_factor, np.arange(1,lookback_window+1))
        model_losses = np.multiply(model_losses, discount)
    except:
        assert False, "Error in discounting!"
    
    if avg_method == AvgMethod.MEAN_SQUARED:
        model_losses = np.mean(np.power(model_losses,2), axis=1)
    else: # MEAN
        model_losses = np.mean(model_losses, axis=1)
    
    # Normalization again before applying softmax with scaling 
    model_losses = np.divide(model_losses, np.sum(model_losses, axis=0)+epsilon)

    # compute the weights of models by applying the weighting method
    if weighting_method == WeightingMethod.SOFTMAX:
        weights = np.exp(-softmax_scaling_factor * model_losses) / np.sum(np.exp(-softmax_scaling_factor * model_losses))
    elif weighting_method == WeightingMethod.INVERTED:
        weights = np.power(model_losses,-1) / np.sum(np.power(np.abs(model_losses),-1))
    else: # weighting_method == WeightingMethod.SquaredInverted':
        weights = np.power(model_losses,-2) / np.sum(np.power(model_losses,-2))

    assert not np.isnan(weights).any()
    assert abs(np.sum(weights) - 1.0) < 1e-3   

    # smoothing the weights
    if prev_model_weights is not None:
        weights = smoothing_factor * prev_model_weights + (1-smoothing_factor) * weights

    # apply max_models constraint : choose max_models components with the highest weights
    if (max_models is not None) and (max_models < num_all_models):
        chosen_indices = np.argpartition(weights, -max_models)[-max_models:] 
        weights = np.array([weights[i] if i in chosen_indices else 0 for i in range(num_all_models)])
        weights = weights / np.sum(weights)

    return weights

