
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
  --use_relativepos True \
  --use_kvr True \
  --use_only 'None'


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
  --use_relativepos True \
  --use_kvr True \
  --use_only 'Spatial'


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
  --use_relativepos True \
  --use_kvr True \
  --use_only 'Temporal'
