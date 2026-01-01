"""
Training script for MVWGNN (Multi-View Wind Graph Neural Network)

Reference:
    Multi-View Graph Neural Networks for Wind Speed Forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from datetime import datetime
from tqdm import tqdm

from models.mvwgnn import MVWGNN
from utils.data_loader import load_data, prepare_sequences, create_data_splits
from utils.metrics import evaluate_model, evaluate_per_city, evaluate_per_horizon
from utils.visualization import plot_training_history


def combined_loss(pred, target, alpha=0.3):
    """
    Combined loss function: α·MAE + (1-α)·MSE

    Args:
        pred: Predictions
        target: Ground truth
        alpha: Weight for MAE (default: 0.3 as per paper)

    Returns:
        Combined loss value
    """
    mae = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    return alpha * mae + (1 - alpha) * mse


def train_epoch(model, train_loader, optimizer, criterion, device, lat_tensor, lon_tensor, epoch, n_epochs):
    """
    Train model for one epoch.

    Args:
        model: MVWGNN model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device (CPU/GPU)
        lat_tensor: Station latitudes
        lon_tensor: Station longitudes
        epoch: Current epoch number
        n_epochs: Total number of epochs

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
    for batch in pbar:
        X = batch['X'].to(device)
        y = batch['y'].to(device)
        feature_sequences = {k: v.to(device) for k, v in batch['feature_sequences'].items()}
        timestamps = {k: v.to(device) for k, v in batch['timestamps'].items()}

        optimizer.zero_grad()

        # Forward pass
        predictions, attention_info = model(X, lat_tensor, lon_tensor, feature_sequences, timestamps)

        # Compute loss
        loss = criterion(predictions, y)

        # Add sparsity regularization on metapath attention (encourages peaked distributions)
        if 'sparsity_loss' in attention_info:
            sparsity_penalty = 0.01 * attention_info['sparsity_loss']
            loss = loss + sparsity_penalty

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / n_batches


def main(args):
    """Main training function."""

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Device configuration
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    log_file = os.path.join(args.log_dir, f"training_log_{timestamp}.txt")
    best_model_path = os.path.join(args.checkpoint_dir, f"best_model_{timestamp}.pth")

    # Print configuration
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Hidden Dim: {args.hidden_dim}")
    print(f"Layer Type: {args.layer_type}")
    print(f"Max Wind Lag: {args.max_wind_lag}")
    print(f"K Geo Neighbors: {args.k_geo_neighbors}")
    print(f"K Feature Neighbors: {args.k_feature_neighbors}")
    print(f"Residual Weight: {args.residual_weight}")
    print(f"Learning Rate: {args.lr}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Window Size: {args.window_size}")
    print(f"Forecast Horizon: {args.forecast_horizon}")
    print(f"Train/Val/Test: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print("="*60 + "\n")

    # Load data
    print("Loading data...")
    df, cities = load_data(dataName=args.dataset)

    # Prepare sequences with proper temporal splitting
    input_features = ["pressure", "humidity", "temperature", "wind_speed"]
    target_feature = "wind_speed"

    sequences, scalers, city_names, city_latitudes, city_longitudes, datetimes, split_indices = prepare_sequences(
        df, cities, input_features, target_feature,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_splits(
        sequences, city_names, input_features, target_feature,
        window_size=args.window_size,
        forecast_horizon=args.forecast_horizon,
        step=args.step,
        batch_size=args.batch_size,
        datetimes=datetimes,
        split_indices=split_indices
    )

    # Initialize model
    print("Initializing MVWGNN model...")
    n_stations = len(city_names)
    n_features = len(input_features)

    model = MVWGNN(
        n_stations=n_stations,
        n_features=n_features,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_wind_lag=args.max_wind_lag,
        k_geo_neighbors=args.k_geo_neighbors,
        k_feature_neighbors=args.k_feature_neighbors,
        residual_weight=args.residual_weight,
        layer_type=args.layer_type,
        num_gnn_layers=args.num_gnn_layers,
        use_feature_view=args.use_feature_view,
        use_geo_view=args.use_geo_view,
        use_prop_view=args.use_prop_view,
        forecast_horizon=args.forecast_horizon
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Loss function
    criterion = combined_loss

    # Station coordinates
    lat_tensor = torch.tensor(city_latitudes, dtype=torch.float32).to(device)
    lon_tensor = torch.tensor(city_longitudes, dtype=torch.float32).to(device)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'test_mae': [],
        'test_rmse': []
    }

    # Best model tracking
    best_val_mae = float('inf')
    patience_counter = 0

    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            lat_tensor, lon_tensor, epoch, args.epochs
        )

        # Evaluate on validation set with per-horizon metrics
        val_loss, val_mae, val_rmse, val_horizon_metrics = evaluate_per_horizon(
            model, val_loader, criterion, device,
            lat_tensor, lon_tensor, scalers[target_feature], "Validation"
        )

        # Evaluate on test set (for monitoring) with per-horizon metrics
        test_loss, test_mae, test_rmse, test_horizon_metrics = evaluate_per_horizon(
            model, test_loader, criterion, device,
            lat_tensor, lon_tensor, scalers[target_feature], "Test"
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['test_mae'].append(test_mae)
        history['test_rmse'].append(test_rmse)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")

        # Print per-horizon validation metrics if multi-horizon
        if args.forecast_horizon > 1:
            print("Val Per-Horizon Metrics:")
            for horizon, metrics in val_horizon_metrics.items():
                print(f"  {horizon}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

        print(f"Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")

        # Print per-horizon test metrics if multi-horizon
        if args.forecast_horizon > 1:
            print("Test Per-Horizon Metrics:")
            for horizon, metrics in test_horizon_metrics.items():
                print(f"  {horizon}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'history': history,
                'config': vars(args)
            }, best_model_path)
            print(f"✓ Saved best model (Val MAE: {val_mae:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set with per-horizon metrics
    test_loss, test_mae, test_rmse, test_horizon_metrics = evaluate_per_horizon(
        model, test_loader, criterion, device,
        lat_tensor, lon_tensor, scalers[target_feature], "Test (Final)"
    )

    print(f"\nOverall Test MAE: {test_mae:.4f}")
    print(f"Overall Test RMSE: {test_rmse:.4f}")

    # Print per-horizon results
    if args.forecast_horizon > 1:
        print("\n" + "-" * 60)
        print("Per-Horizon Test Results:")
        print("-" * 60)
        for horizon, metrics in test_horizon_metrics.items():
            print(f"{horizon:6s} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

    # Evaluate on test set with per-city metrics
    test_loss_city, test_mae_city, test_rmse_city, per_city_metrics = evaluate_per_city(
        model, test_loader, criterion, device,
        lat_tensor, lon_tensor, city_names, scalers[target_feature]
    )

    print("\n" + "-" * 60)
    print("Per-City Results:")
    print("-" * 60)
    for city, metrics in per_city_metrics.items():
        print(f"{city:20s} - MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

    # Save training history plot
    if args.save_plots:
        plot_training_history(history, save_path=f"results/training_history_{timestamp}.png")

    print(f"\n✓ Training complete. Best model saved to: {best_model_path}")

    return model, history, per_city_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MVWGNN for Wind Speed Forecasting')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='ceda_data',
                        choices=['ceda_data', 'hourly-data'],
                        help='Dataset name')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--window_size', type=int, default=24,
                        help='Lookback window size (hours)')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                        help='Forecast horizon (number of steps ahead to predict)')
    parser.add_argument('--step', type=int, default=1,
                        help='Step size between windows')

    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--max_wind_lag', type=int, default=6,
                        help='Maximum wind lag (hours)')
    parser.add_argument('--k_geo_neighbors', type=int, default=5,
                        help='Number of geographic neighbors')
    parser.add_argument('--k_feature_neighbors', type=int, default=5,
                        help='Number of feature-similar neighbors')
    parser.add_argument('--residual_weight', type=float, default=0.2,
                        help='Residual connection weight')
    parser.add_argument('--layer_type', type=str, default='sage',
                        choices=['sage', 'gcn', 'gat'],
                        help='Graph layer type')
    parser.add_argument('--num_gnn_layers', type=int, default=1,
                        help='Number of GNN layers per metapath')

    # View configuration
    parser.add_argument('--use_feature_view', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Enable feature similarity views')
    parser.add_argument('--use_geo_view', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Enable geographic view')
    parser.add_argument('--use_prop_view', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Enable wind propagation view')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')

    # System arguments
    parser.add_argument('--cuda', type=int, default=0,
                        help='CUDA device number')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--save_plots', type=lambda x: str(x).lower() == 'true', default=True,
                        help='Save training plots')

    args = parser.parse_args()

    # Run training
    main(args)
