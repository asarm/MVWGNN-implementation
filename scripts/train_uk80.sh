#!/bin/bash

# Train MVWGNN on UK dataset with 80/10/10 split
# Reproduces paper results for UK-80 configuration

python train.py \
    --dataset ceda_data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --window_size 24 \
    --hidden_dim 128 \
    --max_wind_lag 6 \
    --k_geo_neighbors 5 \
    --k_feature_neighbors 5 \
    --residual_weight 0.2 \
    --layer_type sage \
    --num_gnn_layers 1 \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-3 \
    --patience 10 \
    --cuda 0 \
    --log_dir logs/uk80 \
    --checkpoint_dir checkpoints/uk80
