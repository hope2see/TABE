export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


test_name='Cbm_Adapt_HPO'
desc='chosen_models_BTC'


for interval in 3 6 10 20 50 1000; do
python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')_'$interval \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --combiner '--hpo_policy 2 --hpo_interval '$interval' --max_hpo_eval 100' \
    --adjuster '--lookback_win 3 --discount_factor 1.5 --avg_method 0 --weighting_method 2 --scaling_factor 30 --smoothing_factor 0.0' \
    --basemodel 'EtsModel' \
    --basemodel 'SarimaModel' \
    --basemodel 'TimeMoE' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser'
done
