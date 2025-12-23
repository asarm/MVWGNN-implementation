#!/bin/bash

# Ablation study: Test different view combinations
# This script trains models with different view configurations

echo "Training with all views (baseline)..."
python train.py \
    --dataset hourly-data \
    --use_feature_view True \
    --use_geo_view True \
    --use_prop_view True \
    --log_dir logs/ablation/all_views

echo "Training with Geographic + Feature views only..."
python train.py \
    --dataset hourly-data \
    --use_feature_view True \
    --use_geo_view True \
    --use_prop_view False \
    --log_dir logs/ablation/geo_feature

echo "Training with Geographic view only..."
python train.py \
    --dataset hourly-data \
    --use_feature_view False \
    --use_geo_view True \
    --use_prop_view False \
    --log_dir logs/ablation/geo_only

echo "Training with Feature views only..."
python train.py \
    --dataset hourly-data \
    --use_feature_view True \
    --use_geo_view False \
    --use_prop_view False \
    --log_dir logs/ablation/feature_only

echo "Training with Wind Propagation view only..."
python train.py \
    --dataset hourly-data \
    --use_feature_view False \
    --use_geo_view False \
    --use_prop_view True \
    --log_dir logs/ablation/wind_only

echo "Ablation study complete!"
