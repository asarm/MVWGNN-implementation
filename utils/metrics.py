"""
Evaluation metrics and functions for MVWGNN
"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


def evaluate_per_horizon(model, data_loader, criterion, device, lat_tensor, lon_tensor, scaler=None, split_name="Val"):
    """
    Evaluate model on a dataset with per-horizon metrics.

    Args:
        model: MVWGNN model
        data_loader: Data loader
        criterion: Loss function
        device: Device (CPU/GPU)
        lat_tensor: Station latitudes
        lon_tensor: Station longitudes
        scaler: Scaler for inverse transformation (optional)
        split_name: Name of the split (for logging)

    Returns:
        avg_loss: Average loss
        overall_mae: Overall MAE (across all horizons)
        overall_rmse: Overall RMSE (across all horizons)
        per_horizon_metrics: Dict with MAE and RMSE for each horizon
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"[{split_name}]")
        for batch in pbar:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            feature_sequences = {k: v.to(device) for k, v in batch['feature_sequences'].items()}
            timestamps = {k: v.to(device) for k, v in batch['timestamps'].items()}

            predictions, _ = model(X, lat_tensor, lon_tensor, feature_sequences, timestamps)
            loss = criterion(predictions, y)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            n_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Shape: [total_samples, N_cities] or [total_samples, N_cities, forecast_horizon]

    # Check if multi-horizon
    is_multi_horizon = len(all_predictions.shape) == 3

    if is_multi_horizon:
        forecast_horizon = all_predictions.shape[2]

        # Inverse transform to original scale if scaler provided
        if scaler is not None:
            original_shape = all_predictions.shape
            all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(original_shape)
            all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(original_shape)

        # Calculate per-horizon metrics
        per_horizon_metrics = {}
        for h in range(forecast_horizon):
            # Extract predictions and targets for this horizon
            h_preds = all_predictions[:, :, h].flatten()
            h_targets = all_targets[:, :, h].flatten()

            h_mae = mean_absolute_error(h_targets, h_preds)
            h_rmse = np.sqrt(mean_squared_error(h_targets, h_preds))

            per_horizon_metrics[f't+{h+1}'] = {
                'mae': h_mae,
                'rmse': h_rmse
            }

        # Overall metrics (across all horizons)
        all_predictions_flat = all_predictions.flatten()
        all_targets_flat = all_targets.flatten()
        overall_mae = mean_absolute_error(all_targets_flat, all_predictions_flat)
        overall_rmse = np.sqrt(mean_squared_error(all_targets_flat, all_predictions_flat))

    else:
        # Single horizon case
        if scaler is not None:
            original_shape = all_predictions.shape
            all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(original_shape)
            all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(original_shape)

        all_predictions_flat = all_predictions.flatten()
        all_targets_flat = all_targets.flatten()

        overall_mae = mean_absolute_error(all_targets_flat, all_predictions_flat)
        overall_rmse = np.sqrt(mean_squared_error(all_targets_flat, all_predictions_flat))

        per_horizon_metrics = {
            't+1': {
                'mae': overall_mae,
                'rmse': overall_rmse
            }
        }

    return avg_loss, overall_mae, overall_rmse, per_horizon_metrics


def evaluate_model(model, data_loader, criterion, device, lat_tensor, lon_tensor, scaler=None, split_name="Val"):
    """
    Evaluate model on a dataset.

    Args:
        model: MVWGNN model
        data_loader: Data loader
        criterion: Loss function
        device: Device (CPU/GPU)
        lat_tensor: Station latitudes
        lon_tensor: Station longitudes
        scaler: Scaler for inverse transformation (optional)
        split_name: Name of the split (for logging)

    Returns:
        avg_loss: Average loss
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"[{split_name}]")
        for batch in pbar:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            feature_sequences = {k: v.to(device) for k, v in batch['feature_sequences'].items()}
            timestamps = {k: v.to(device) for k, v in batch['timestamps'].items()}

            predictions, _ = model(X, lat_tensor, lon_tensor, feature_sequences, timestamps)
            loss = criterion(predictions, y)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            n_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Shape: [total_samples, N_cities] or [total_samples, N_cities, forecast_horizon]

    # Inverse transform to original scale if scaler provided
    if scaler is not None:
        original_shape = all_predictions.shape
        all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(original_shape)
        all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(original_shape)

    # Flatten for metrics calculation
    all_predictions = all_predictions.flatten()
    all_targets = all_targets.flatten()

    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

    return avg_loss, mae, rmse


def evaluate_per_city(model, data_loader, criterion, device, lat_tensor, lon_tensor, city_names, scaler=None):
    """
    Evaluate model with per-city metrics.

    Args:
        model: MVWGNN model
        data_loader: Data loader
        criterion: Loss function
        device: Device (CPU/GPU)
        lat_tensor: Station latitudes
        lon_tensor: Station longitudes
        city_names: List of city names
        scaler: Scaler for inverse transformation (optional)

    Returns:
        avg_loss: Average loss
        overall_mae: Overall MAE
        overall_rmse: Overall RMSE
        per_city_metrics: Dict with per-city MAE and RMSE
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="[Test - Per City]")
        for batch in pbar:
            X = batch['X'].to(device)
            y = batch['y'].to(device)
            feature_sequences = {k: v.to(device) for k, v in batch['feature_sequences'].items()}
            timestamps = {k: v.to(device) for k, v in batch['timestamps'].items()}

            predictions, _ = model(X, lat_tensor, lon_tensor, feature_sequences, timestamps)
            loss = criterion(predictions, y)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            n_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    # Shape: [total_samples, N_cities] or [total_samples, N_cities, forecast_horizon]

    # Determine number of cities (shape[1] is always N_cities)
    n_cities = all_predictions.shape[1]
    per_city_metrics = {}

    # Handle both single-horizon and multi-horizon cases
    for city_idx in range(n_cities):
        # Extract city-specific predictions and targets
        if len(all_predictions.shape) == 2:
            # Single horizon: [total_samples, N_cities]
            city_preds = all_predictions[:, city_idx]
            city_targets = all_targets[:, city_idx]
        else:
            # Multi horizon: [total_samples, N_cities, forecast_horizon]
            city_preds = all_predictions[:, city_idx, :]
            city_targets = all_targets[:, city_idx, :]

        # Inverse transform to original scale if scaler provided
        if scaler is not None:
            original_shape = city_preds.shape
            city_preds = scaler.inverse_transform(city_preds.reshape(-1, 1)).reshape(original_shape)
            city_targets = scaler.inverse_transform(city_targets.reshape(-1, 1)).reshape(original_shape)

        # Flatten for metrics
        city_preds = city_preds.flatten()
        city_targets = city_targets.flatten()

        city_mae = mean_absolute_error(city_targets, city_preds)
        city_rmse = np.sqrt(mean_squared_error(city_targets, city_preds))

        per_city_metrics[city_names[city_idx]] = {
            'mae': city_mae,
            'rmse': city_rmse
        }

    # Overall metrics
    if scaler is not None:
        original_shape = all_predictions.shape
        all_predictions_flat = scaler.inverse_transform(all_predictions.reshape(-1, 1)).reshape(original_shape).flatten()
        all_targets_flat = scaler.inverse_transform(all_targets.reshape(-1, 1)).reshape(original_shape).flatten()
    else:
        all_predictions_flat = all_predictions.flatten()
        all_targets_flat = all_targets.flatten()

    overall_mae = mean_absolute_error(all_targets_flat, all_predictions_flat)
    overall_rmse = np.sqrt(mean_squared_error(all_targets_flat, all_predictions_flat))

    return avg_loss, overall_mae, overall_rmse, per_city_metrics
