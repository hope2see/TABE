import numpy as np 
import pandas as pd
from scipy.stats import norm as norm_dist
from scipy.stats import t as t_dist
from tabe.utils.logger import logger


def _fit_dist(data: np.array, dist: str = 'best'):
    """
    dist : 'best', 'norm' or 't'
        if 'best', fit both normal and t-distribution, and return the best fit.    

    returns (dist, dist_params) or dist_params
    """

    if dist in ['best', 'norm']:
        norm_params = norm_dist.fit(data)
        norm_ll = np.sum(norm_dist.logpdf(data, *norm_params))
        norm_aic = -2 * norm_ll + 2 * len(norm_params)
        logger.debug(f"Normal : Likelihood {norm_ll}, AIC {norm_aic}")

    if dist in ['best', 't']:
        t_params = t_dist.fit(data)
        t_ll = np.sum(t_dist.logpdf(data, *t_params))
        t_aic = -2 * t_ll + 2 * len(t_params)
        logger.debug(f"Stud-t : Likelihood {t_ll}, AIC {t_aic}")

    if dist == 'best':
        return ('norm', norm_params) if norm_aic < t_aic else ('t', t_params)
    elif dist == 'norm':
        return norm_params
    else: # if dist == 't':
        return t_params


def get_quantile_value(data: np.array, quantile, dist='best'):
    """
    return (left_value, right_value), 
        where quantile == P(X ≤ left_value) == P(X >= right_value) 
    """
    dist, params = _fit_dist(data)

    if dist == 'norm':
        left_value = norm_dist.ppf(quantile, *params)
        right_value = norm_dist.ppf(1-quantile, *params)
    else : # dist == 't':
        left_value = t_dist.ppf(quantile, *params)
        right_value = t_dist.ppf(1-quantile, *params)
    
    return (left_value, right_value)


def get_cumul_prob(data: np.array, value, left_tail=False, dist='best'):
    """
    x : value 
    left_tail : if True, return P(X ≤ value)
    """

    dist, params = _fit_dist(data)

    if dist == 'norm':
        p = norm_dist.cdf(value, *params)
    else : # dist == 't':
        p = t_dist.cdf(value, *params)
    
    return p if left_tail else 1-p

