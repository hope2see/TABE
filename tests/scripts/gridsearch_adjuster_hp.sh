export PYTHONPATH=$PYTHONPATH:./Time-Series-Library


test_name='Adj_HP_Grid'
desc='chosen_models_BTC'


for gpm_lookback_win in 10 30 50 -1; do 
for lookback_win in 3 5 10; do 
for discount_factor in 1.0 1.3 1.5 2.0; do 
for avg_method in 0 1; do 
for weighting_method in 0 1; do
python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')_'$gpm_lookback_win'_'$lookback_win'_'$discount_factor'_'$avg_method'_'$weighting_method'_'$scaling_factor \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --adjuster '--gpm_lookback_win '$gpm_lookback_win' --lookback_win  '$lookback_win' --discount_factor '$discount_factor' --avg_method '$avg_method' --weighting_method '$weighting_method \
    --combiner '--lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' \
    --basemodel 'EtsModel' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser'
done 
done
done 
done
done 


for gpm_lookback_win in 10 30 50 -1; do 
for lookback_win in 3 5 10; do 
for discount_factor in 1.0 1.3 1.5 2.0; do 
for avg_method in 0 1; do 
for weighting_method in 2; do
for scaling_factor in 1 10 30 100; do 
python -u run.py \
    --model TABE \
    --model_id $test_name'_('$desc')_'$gpm_lookback_win'_'$lookback_win'_'$discount_factor'_'$avg_method'_'$weighting_method'_'$scaling_factor \
    --data TABE_FILE --data_path 'BTC-USD_LogRet_2021-01-01_2023-01-01_1d.csv' \
    --train_epochs 3  \
    --adjuster '--gpm_lookback_win '$gpm_lookback_win' --lookback_win  '$lookback_win' --discount_factor '$discount_factor' --avg_method '$avg_method' --weighting_method '$weighting_method' --scaling_factor '$scaling_factor \
    --combiner '--lookback_win 1 --discount_factor 3.0 --avg_method 0 --weighting_method 1 --max_models 2' \
    --basemodel 'EtsModel' \
    --basemodel 'Drifter' \
    --basemodel 'Noiser'
done 
done
done 
done
done 
done 

