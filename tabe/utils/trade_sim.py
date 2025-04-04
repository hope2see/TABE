import numpy as np
import pandas as pd
import tabe.utils.report as report


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


def _simulate_trading_daily_buy_sell(true_rets, pred_rets, threshold_buy, 
                                     apply_threshold_prob=False, prob_ascendings=None, threshold_prob=0.5, fee_rate=0.001):
    balance = 1.0
    trade_stats = []    
    for t in range(len(pred_rets)):
        buy_condition_met = (pred_rets[t] > threshold_buy)
        if apply_threshold_prob:
            buy_condition_met = buy_condition_met and (prob_ascendings[t] > threshold_prob)
        if buy_condition_met: 
            orig_balance = balance
            balance -= balance * fee_rate # buy
            balance += balance * true_rets[t]
            balance -= balance * fee_rate # sell
            profit_rate = (balance - orig_balance) / orig_balance
            trade_stats.append([t+1, balance, profit_rate]) 
    return balance - 1.0, trade_stats


def _simulate_trading_buy_hold_sell(strategy, true_rets, pred_rets, threshold_buy, threshold_sell, 
                                    apply_threshold_prob=False, prob_ascendings=None, threshold_prob=0.5, fee_rate=0.001):
    balance = 1.0
    trade_stats = []
    holding = False
    profit_rate = 0.0
    for t in range(len(pred_rets)):
        if not holding:
            buy_condition_met = (pred_rets[t] > threshold_buy)
            if apply_threshold_prob:
                buy_condition_met = buy_condition_met and (prob_ascendings[t] > threshold_prob)
            if buy_condition_met: # buy 
                orig_balance = balance
                balance -= balance * fee_rate
                holding = True
        else: 
            # Strategy 'buy_hold_sell_v1' sell condition : 
            #    If tomorrow's predicted return is below the sell_threshold, sell
            sell_condition_met = (pred_rets[t] < threshold_sell) 
            if apply_threshold_prob:
                # if tomorrow's prob_ascendings is not above the threshold_prob, then sell
                sell_condition_met = sell_condition_met or not (prob_ascendings[t] > threshold_prob)
            if strategy == 'buy_hold_sell_v2':
                # If 'buy_hold_sell_v1' condition is met, OR if today's return is below the sell_threshold, then sell
                sell_condition_met = sell_condition_met or (true_rets[t-1] < threshold_sell)

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


def _simulate_trading(strategy, true_rets, pred_rets, threshold_buy=0.002, threshold_sell=0.0, 
                    apply_threshold_prob=False, prob_ascendings=None, threshold_prob=0.5,
                    fee_rate=0.001):
    assert strategy in ['buy_and_hold', 'daily_buy_sell', 'buy_hold_sell_v1', 'buy_hold_sell_v2']
    assert len(true_rets) == len(pred_rets)

    if strategy == 'buy_and_hold':
        accumulated_ret, trade_stats = _simulate_trading_buy_and_hold(true_rets, fee_rate)
    elif strategy == 'daily_buy_sell':
        accumulated_ret, trade_stats = _simulate_trading_daily_buy_sell(
            true_rets, pred_rets, threshold_buy, apply_threshold_prob, prob_ascendings, threshold_prob, fee_rate)
    else: # 'buy_hold_sell_v1', 'buy_hold_sell_v2'
        accumulated_ret, trade_stats = _simulate_trading_buy_hold_sell(strategy,
            true_rets, pred_rets, threshold_buy, threshold_sell, apply_threshold_prob, prob_ascendings, threshold_prob, fee_rate)
        
    num_of_trades = len(trade_stats)
    trade_stats = np.array(trade_stats)
    num_of_successful_trades = np.count_nonzero(trade_stats[:, 2] > 0) if num_of_trades > 0 else 0
    mean_profit_rate = np.mean(trade_stats[:, 2]) if num_of_trades > 0 else 0.0
    successful_trade_rate = float(num_of_successful_trades) / num_of_trades  if num_of_trades > 0 else 0.0
    return accumulated_ret, mean_profit_rate, num_of_trades, num_of_successful_trades, successful_trade_rate


def simulate_trading(configs, truths, models):
    assert configs.target in ['LogRet1', 'Ret1']

    preds_list = [m.result_predictions() for m in models]
    if configs.target == 'LogRet1': # Convert data to 'Ret'        
        truths = np.exp(truths) - 1
        for i in range(len(preds_list)):
            preds_list[i] = np.exp(preds_list[i]) - 1

    # Simulation with 'Ret'
    for strategy in ['buy_and_hold', 'daily_buy_sell', 'buy_hold_sell_v1', 'buy_hold_sell_v2']:
        apply_prob_conditions = [False] if strategy == 'buy_and_hold' else [True, False]       
        for apply_threshold_prob in apply_prob_conditions:
            df_sim_result = pd.DataFrame() 
            for i, m in enumerate(models):
                df_sim_result[m.name] = _simulate_trading(strategy, truths, preds_list[i], threshold_buy=configs.buy_threshold_ret, 
                                                        apply_threshold_prob=apply_threshold_prob, prob_ascendings=m.result_prob_ascendings(), 
                                                        threshold_prob=configs.buy_threshold_prob,
                                                        fee_rate=configs.fee_rate)      
            df_sim_result.index = ['Acc. ROI', 'Mean ROI', '# Trades', '# Win_Trades', 'Winning Rate']
            title = f"[ Trading Simulation Results: (Strategy:{strategy}, Days:{len(truths)}, Apply_Prob:{apply_threshold_prob} ]"
            report.print_dataframe(df_sim_result, title, filepath=None)
