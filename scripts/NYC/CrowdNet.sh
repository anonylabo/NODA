
python -u run.py \
  --path ./ \
  --model CrowdNet \
  --sample_time '60min' \
  --itrs 10 \
  --train_epochs 150 \
  --patience 5 \
  --lr 1e-04 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_temporal 64 \
  --d_spatual 8

python -u run.py \
  --path ./ \
  --model CrowdNet \
  --sample_time '45min' \
  --itrs 10 \
  --train_epochs 150 \
  --patience 5 \
  --lr 1e-04 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_temporal 64 \
  --d_spatual 8

python -u run.py \
  --path ./ \
  --model CrowdNet \
  --sample_time '30min' \
  --itrs 10 \
  --train_epochs 150 \
  --patience 5 \
  --lr 1e-04 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_temporal 64 \
  --d_spatual 8

python -u run.py \
  --path ./ \
  --model CrowdNet \
  --sample_time '15min' \
  --itrs 10 \
  --train_epochs 150 \
  --patience 5 \
  --lr 1e-04 \
  --city 'NYC' \
  --num_tiles 47 \
  --d_temporal 64 \
  --d_spatual 8

