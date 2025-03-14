import numpy as np


def _simulate_trading_buy_and_hold(true_rets, fee_rate):
    balance = 1.0
    trade_stats = []
    balance -= balance * fee_rate # buy 
    for t in range(len(true_rets)):
        balance += balance * true_rets[t]
    balance -= balance * fee_rate # sell at the last day 
    profit_rate = balance - 1.0
    trade_stats.append([t+1, balance, profit_rate]) 
    return balance - 1.0, trade_stats


def _pred_is_believable(deviation_stddev):
    return True if deviation_stddev < 1.2 else False


def _simulate_trading_daily_buy_sell(true_rets, pred_rets, buy_threshold, buy_threshold_q, fee_rate, devi_stddev=None, apply_threshold_prob=False):
    balance = 1.0
    trade_stats = []    
    for t in range(len(pred_rets)):
        buy_condition_met = (pred_rets[t] > buy_threshold)
        if apply_threshold_prob:
            # buy_condition_met = buy_condition_met and _pred_is_believable(devi_stddev[t])
            buy_condition_met = buy_condition_met and (buy_threshold_q[t] >= buy_threshold)
        if buy_condition_met: 
            orig_balance = balance
            balance -= balance * fee_rate # buy
            balance += balance * true_rets[t]
            balance -= balance * fee_rate # sell
            profit_rate = (balance - orig_balance) / orig_balance
            trade_stats.append([t+1, balance, profit_rate]) 
    return balance - 1.0, trade_stats


def _simulate_trading_buy_hold_sell(true_rets, pred_rets, buy_threshold, buy_threshold_q, sell_threshold, fee_rate, strategy='buy_hold_sell_v1',
                                     devi_stddev=None, apply_threshold_prob=False):
    balance = 1.0
    trade_stats = []
    holding = False
    profit_rate = 0.0
    for t in range(len(pred_rets)):
        if not holding:
            buy_condition_met = (pred_rets[t] > buy_threshold)
            if apply_threshold_prob:
                # buy_condition_met = buy_condition_met and _pred_is_believable(devi_stddev[t])
                buy_condition_met = buy_condition_met and (buy_threshold_q[t] >= buy_threshold)
            if buy_condition_met: # buy 
                orig_balance = balance
                balance -= balance * fee_rate
                holding = True
        else: 
            # Strategy 'buy_hold_sell_v1' sell condition : 
            #    If tomorrow's predicted return is below the sell_threshold, sell
            sell_condition_met = (pred_rets[t] < sell_threshold) 
            # Strategy 'buy_hold_sell_v2' sell condition : 
            #    If 'buy_hold_sell_v1' condition is met, OR if today's return is below the sell_threshold, then sell
            if strategy == 'buy_hold_sell_v2':
                sell_condition_met = sell_condition_met or (true_rets[t-1] < sell_threshold)
            if apply_threshold_prob:
                sell_condition_met = sell_condition_met or not _pred_is_believable(devi_stddev[t])

            if sell_condition_met: 
                balance += balance * true_rets[t-1] # sell at today's close price.
                balance -= balance * fee_rate
                profit_rate = (balance - orig_balance) / orig_balance
                trade_stats.append([t, balance, profit_rate])
                holding = False
            # Otherwise, accumulate today's return
            else:
                balance += balance * true_rets[t-1] 

    if holding: # sell at the last day
        balance += balance * true_rets[t]
        balance -= balance * fee_rate
        profit_rate = (balance - orig_balance) / orig_balance
        trade_stats.append([t, balance, profit_rate])

    return balance - 1.0, trade_stats


def simulate_trading(true_rets, pred_rets, strategy='daily_buy_sell', devi_stddev=None, apply_threshold_prob=False,
                     buy_threshold=0.002, buy_threshold_q=None, sell_threshold=0.0, fee_rate=0.001):
    assert strategy in ['buy_and_hold', 'daily_buy_sell', 'buy_hold_sell_v1', 'buy_hold_sell_v2']
    assert len(true_rets) == len(pred_rets)

    if strategy == 'buy_and_hold':
        accumulated_ret, trade_stats = _simulate_trading_buy_and_hold(true_rets, fee_rate)
    elif strategy == 'daily_buy_sell':
        accumulated_ret, trade_stats = _simulate_trading_daily_buy_sell(
            true_rets, pred_rets, buy_threshold, buy_threshold_q, fee_rate, devi_stddev, apply_threshold_prob)
    else: # 'buy_hold_sell_v1', 'buy_hold_sell_v2'
        accumulated_ret, trade_stats = _simulate_trading_buy_hold_sell(
            true_rets, pred_rets, buy_threshold, buy_threshold_q, sell_threshold, fee_rate, strategy, devi_stddev, apply_threshold_prob)
        
    num_of_trades = len(trade_stats)
    trade_stats = np.array(trade_stats)
    num_of_successful_trades = np.count_nonzero(trade_stats[:, 2] > 0) if num_of_trades > 0 else 0
    mean_profit_rate = np.mean(trade_stats[:, 2]) if num_of_trades > 0 else 0.0
    successful_trade_rate = float(num_of_successful_trades) / num_of_trades  if num_of_trades > 0 else 0.0
    return accumulated_ret, mean_profit_rate, num_of_trades, num_of_successful_trades, successful_trade_rate
