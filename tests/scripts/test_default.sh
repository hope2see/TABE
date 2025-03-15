export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


test_name='Test_Default'

desc='all_models_BTC'

python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')' \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2022-01-01_2025-01-01_1d.csv' \
    --data_test_split 0.3 \
    --data_train_splits 0.4 0.5 0.1 \
    --seq_len 128 --label_len 128 \
    --train_epochs 10  \
    --basemodel 'ETS' \
    --basemodel 'AutoSARIMA' \
    --basemodel 'DLinear --batch_size 8' \
    --basemodel 'iTransformer' \
    --basemodel 'PatchTST --batch_size 16' \
    --basemodel 'TimeXer' \
    --basemodel 'CMamba --batch_size 64 --lradj type3 --learning_rate 0.0005 --d_model 128 --d_ff 128' \
    --basemodel 'TimeMoE' \
    --basemodel 'Timer' \
    --combiner '--lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' 
    # --adjuster '--gpm_lookback_win 3 --lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' 


desc='all_models_SPY'

python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')' \
    --data TABE_FILE --data_path 'SPY_LogRet_2023-01-01_2025-01-01_1d.csv' \
    --data_test_split 0.2 \
    --data_train_splits 0.4 0.5 0.1 \
    --seq_len 128 --label_len 128 \
    --train_epochs 10  \
    --basemodel 'ETS' \
    --basemodel 'AutoSARIMA' \
    --basemodel 'DLinear --batch_size 8' \
    --basemodel 'iTransformer' \
    --basemodel 'PatchTST --batch_size 16' \
    --basemodel 'TimeXer' \
    --basemodel 'CMamba --batch_size 64 --lradj type3 --learning_rate 0.0005 --d_model 128 --d_ff 128' \
    --basemodel 'TimeMoE' \
    --basemodel 'Timer' \
    --combiner '--lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' 
    # --adjuster '--gpm_lookback_win 3 --lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' 
