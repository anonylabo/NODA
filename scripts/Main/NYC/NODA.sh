
python -u run.py \
  --path ./ \
  --model NODA \
  --sample_time '60min' \
  --itrs 10 \
  --train_epochs 50 \
  --patience 5 \
  --lr 1e-03 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_model 64 \
  --n_head 8 \
  --temporal_num_layers 2 \
  --spatial_num_layers 1 \
  --use_kvr

python -u run.py \
  --path ./ \
  --model NODA \
  --sample_time '45min' \
  --itrs 10 \
  --train_epochs 50 \
  --patience 5 \
  --lr 1e-03 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_model 64 \
  --n_head 8 \
  --temporal_num_layers 2 \
  --spatial_num_layers 1 \
  --use_kvr

python -u run.py \
  --path ./ \
  --model NODA \
  --sample_time '30min' \
  --itrs 10 \
  --train_epochs 50 \
  --patience 5 \
  --lr 1e-03 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_model 64 \
  --n_head 8 \
  --temporal_num_layers 2 \
  --spatial_num_layers 1 \
  --use_kvr

python -u run.py \
  --path ./ \
  --model NODA \
  --sample_time '15min' \
  --itrs 10 \
  --train_epochs 50 \
  --patience 5 \
  --lr 1e-03 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_model 64 \
  --n_head 8 \
  --temporal_num_layers 2 \
  --spatial_num_layers 1 \
  --use_kvr
