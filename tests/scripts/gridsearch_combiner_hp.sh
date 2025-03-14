export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


test_name='Cbm_HP_Grid'
desc='chosen_models_BTC'


for lookback_win in 1 3 5 7 10 15; do 
for discount_factor in 1.0 1.3 1.5 2.0; do 
for avg_method in 0 1; do 
for weighting_method in 0 1; do
python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')_'$lookback_win'_'$discount_factor'_'$avg_method'_'$weighting_method'_'$scaling_factor \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --adjuster '--lookback_win 3 --discount_factor 1.5 --avg_method 0 --weighting_method 2 --scaling_factor 30 --smoothing_factor 0.0' \
    --combiner '--lookback_win '$lookback_win' --discount_factor '$discount_factor' --avg_method '$avg_method' --weighting_method '$weighting_method \
    --basemodel 'EtsModel' \
    --basemodel 'SarimaModel' \
    --basemodel 'TimeMoE' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser' \
    --basemodel 'Drifter' 
done 
done
done 
done


for lookback_win in 1 3 5 7 10 15; do 
for discount_factor in 1.0 1.3 1.5 2.0; do 
for avg_method in 0 1; do 
for weighting_method in 2; do
for scaling_factor in 1 10 30 100; do 
python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')_'$lookback_win'_'$discount_factor'_'$avg_method'_'$weighting_method'_'$scaling_factor \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --adjuster '--lookback_win 3 --discount_factor 1.5 --avg_method 0 --weighting_method 2 --scaling_factor 30 --smoothing_factor 0.0' \
    --combiner '--lookback_win '$lookback_win' --discount_factor '$discount_factor' --avg_method '$avg_method' --weighting_method '$weighting_method' --scaling_factor '$scaling_factor \
    --basemodel 'EtsModel' \
    --basemodel 'SarimaModel' \
    --basemodel 'TimeMoE' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser' \
    --basemodel 'Drifter' 
done 
done
done 
done
done 
