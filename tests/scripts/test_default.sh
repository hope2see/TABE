export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


test_name='Test_Default'

desc='all_models_BTC'

python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')' \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --basemodel 'EtsModel' \
    --basemodel 'SarimaModel' \
    --basemodel 'DLinear --batch_size 8' \
    --basemodel 'iTransformer' \
    --basemodel 'PatchTST --batch_size 16' \
    --basemodel 'TimeXer' \
    --basemodel 'CMamba --batch_size 64 --lradj type3 --learning_rate 0.0005 --d_model 128 --d_ff 128' \
    --basemodel 'TimeMoE' \
    --combiner '--lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' \
    --adjuster '--gpm_lookback_win 3 --lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' 


desc='all_models_SPY'

python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')' \
    --data TABE_FILE --data_path 'SPY_LogRet_2023-01-01_2025-01-01_1d.csv' \
    --train_epochs 3  \
    --basemodel 'EtsModel' \
    --basemodel 'SarimaModel' \
    --basemodel 'DLinear --batch_size 8' \
    --basemodel 'iTransformer' \
    --basemodel 'PatchTST --batch_size 16' \
    --basemodel 'TimeXer' \
    --basemodel 'CMamba --batch_size 64 --lradj type3 --learning_rate 0.0005 --d_model 128 --d_ff 128' \
    --basemodel 'TimeMoE' \
    --combiner '--lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' \
    --adjuster '--gpm_lookback_win 3 --lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' 
