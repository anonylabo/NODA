
python -u run.py \
  --path ./ \
  --model CrowdNet \
  --sample_time '60min' \
  --itrs 3 \
  --train_epochs 150 \
  --patience 5 \
  --lr 1e-04 \
  --city 'DC' \
  --num_tiles 154 \
  --d_temporal 16 \
  --d_spatual 8

python -u run.py \
  --path ./ \
  --model CrowdNet \
  --sample_time '45min' \
  --itrs 3 \
  --train_epochs 150 \
  --patience 5 \
  --lr 1e-04 \
  --city 'DC' \
  --num_tiles 154 \
  --d_temporal 16 \
  --d_spatual 8

