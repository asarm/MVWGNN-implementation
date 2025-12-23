#!/bin/bash

echo ""
echo "Example: Multi-step forecasting (forecast_horizon=3)"
echo ""
python train.py \
    --dataset hourly-data \
    --forecast_horizon 3 \
    --step 3 \
    --window_size 24 \
    --hidden_dim 128 \
    --epochs 50 \
    --batch_size 32 \
    --residual_weight 0.4 \
    --log_dir "horizon3"