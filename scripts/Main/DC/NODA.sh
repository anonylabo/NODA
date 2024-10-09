
python -u run.py \
  --path ./ \
  --model NODA \
  --sample_time '60min' \
  --itrs 3 \
  --train_epochs 50 \
  --patience 5 \
  --lr 1e-03 \
  --city 'DC' \
  --num_tiles 154 \
  --d_model 16 \
  --n_head 8 \
  --temporal_num_layers 2 \
  --spatial_num_layers 1 \
  --use_kvr

python -u run.py \
  --path ./ \
  --model NODA \
  --sample_time '45min' \
  --itrs 3 \
  --train_epochs 50 \
  --patience 5 \
  --lr 1e-03 \
  --city 'DC' \
  --num_tiles 154 \
  --d_model 16 \
  --n_head 8 \
  --temporal_num_layers 2 \
  --spatial_num_layers 1 \
  --use_kvr
