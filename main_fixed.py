import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from dataLoad import load_data
from metapath_model import MetapathGNN
from metapath_model_shared import MetapathGNN_Shared

def prepare_sequences(df, cities, input_features, target_feature, train_ratio=0.8, val_ratio=0.15):
    """
    Prepare sequences for all cities with PROPER temporal splitting and normalization.
    
    CRITICAL FIX: Normalization is done ONLY on training data to prevent data leakage.
    
    Returns:
        sequences: dict {city: {feature: array}}
        scalers: dict {feature: StandardScaler} - fitted ONLY on train data
        city_names, city_latitudes, city_longitudes
        split_indices: dict with train/val/test boundaries
    """
    # Get city list and coordinates
    city_names = cities['City'].tolist()
    city_latitudes = cities['Latitude'].tolist()
    city_longitudes = cities['Longitude'].tolist()

    print(f"Number of cities: {len(city_names)}")

    # First, determine temporal split boundaries BEFORE normalization
    total_length = len(df)
    train_end_idx = int(total_length * train_ratio)
    val_end_idx = int(total_length * (train_ratio + val_ratio))
    
    split_indices = {
        'train_end': train_end_idx,
        'val_end': val_end_idx,
        'total': total_length
    }
    
    print(f"\n{'='*60}")
    print(f"TEMPORAL SPLIT BOUNDARIES")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_length}")
    print(f"Train: 0 to {train_end_idx} ({train_end_idx} timesteps, {train_ratio*100:.1f}%)")
    print(f"Val: {train_end_idx} to {val_end_idx} ({val_end_idx - train_end_idx} timesteps, {val_ratio*100:.1f}%)")
    print(f"Test: {val_end_idx} to {total_length} ({total_length - val_end_idx} timesteps, {(1-train_ratio-val_ratio)*100:.1f}%)")
    print(f"{'='*60}\n")

    # Prepare raw sequences for all cities
    sequences_raw = {}
    for city in city_names:
        imputer = SimpleImputer(strategy="mean")
        imputed_data = imputer.fit_transform(df[city].values)
        sequences_raw[city] = {
            "humidity": imputed_data[:, 0],
            "temperature": imputed_data[:, 1],
            "pressure": imputed_data[:, 2],
            "wind_speed": imputed_data[:, 4],
            "wind_direction": imputed_data[:, 3],
        }

    # Normalize features using ONLY TRAINING DATA (CRITICAL FIX for data leakage)
    scalers = {}
    features_to_scale = [f for f in dict.fromkeys(input_features + [target_feature]) if f != 'wind_direction']

    print(f"{'='*60}")
    print(f"NORMALIZATION (using ONLY training data)")
    print(f"{'='*60}")
    
    for feature in features_to_scale:
        # Collect ONLY TRAINING data for this feature across all cities
        train_feature_data = []
        for city in city_names:
            train_feature_data.extend(sequences_raw[city][feature][:train_end_idx])

        # Fit scaler ONLY on training data
        scaler = StandardScaler()
        scaler.fit(np.array(train_feature_data).reshape(-1, 1))
        scalers[feature] = scaler
        
        print(f"{feature:15s} - mean: {scaler.mean_[0]:8.4f}, std: {scaler.scale_[0]:8.4f} (from {len(train_feature_data)} train samples)")

    print(f"{'='*60}\n")

    # Apply scaling to ALL data (but scaler was fitted only on train)
    sequences = {}
    for city in city_names:
        sequences[city] = {}
        for feature in sequences_raw[city].keys():
            if feature == 'wind_direction':
                # Don't scale wind direction (circular variable)
                sequences[city][feature] = sequences_raw[city][feature]
            else:
                # Apply scaler fitted on train data to all data
                sequences[city][feature] = scalers[feature].transform(
                    sequences_raw[city][feature].reshape(-1, 1)
                ).flatten()

    datetimes = df.index

    return sequences, scalers, city_names, city_latitudes, city_longitudes, datetimes, split_indices


def create_data_splits(sequences, city_names, input_features, target_feature,
                      window_size=24, step=1, datetimes=None, split_indices=None):
    """
    Create train/validation/test splits using TEMPORAL splitting (no random shuffle).
    
    CRITICAL FIX: Uses temporal boundaries to prevent data leakage between splits.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Use the pre-determined temporal boundaries
    train_end = split_indices['train_end']
    val_end = split_indices['val_end']
    total_length = split_indices['total']
    
    # Calculate valid starting positions for each split
    # Important: subtract window_size + 1 to ensure we have enough data for target
    train_positions = list(range(0, train_end - window_size - 1, step))
    val_positions = list(range(train_end, val_end - window_size - 1, step))
    test_positions = list(range(val_end, total_length - window_size - 1, step))

    print(f"{'='*60}")
    print(f"DATA SPLIT STATISTICS")
    print(f"{'='*60}")
    print(f"Window size: {window_size}, Step size: {step}")
    print(f"Overlap per window: {window_size - step} timesteps ({(window_size - step)/window_size*100:.1f}%)")
    print(f"\nTrain: {len(train_positions)} samples (timesteps: 0-{train_end})")
    print(f"Val:   {len(val_positions)} samples (timesteps: {train_end}-{val_end})")
    print(f"Test:  {len(test_positions)} samples (timesteps: {val_end}-{total_length})")
    print(f"{'='*60}\n")

    # Create datasets
    train_dataset = WeatherDataset(sequences, city_names, input_features, target_feature,
                                  train_positions, window_size, datetimes)
    val_dataset = WeatherDataset(sequences, city_names, input_features, target_feature,
                                val_positions, window_size, datetimes)
    test_dataset = WeatherDataset(sequences, city_names, input_features, target_feature,
                                 test_positions, window_size, datetimes)

    # Create data loaders - NO SHUFFLE for validation and test (time series!)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Only train can shuffle
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class WeatherDataset(Dataset):
    """
    Dataset for weather forecasting with multiple cities.
    """
    def __init__(self, sequences, city_names, input_features, target_feature,
                 indices, window_size, datetimes=None):
        self.sequences = sequences
        self.city_names = city_names
        self.input_features = input_features
        self.target_feature = target_feature
        self.indices = indices
        self.window_size = window_size
        self.n_cities = len(city_names)
        self.n_features = len(input_features)
        # datetimes should be an index-like object (e.g., pd.DatetimeIndex)
        # used to extract hour of day and day of year for each window
        self.datetimes = datetimes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Collect window for all cities: [n_cities, window_size, n_features]
        city_windows = []
        for city in self.city_names:
            window_features = []
            for feature in self.input_features:
                window_features.append(self.sequences[city][feature][i:i+self.window_size])
            city_windows.append(np.stack(window_features, axis=1))

        X = np.stack(city_windows, axis=0)  # [n_cities, window_size, n_features]

        # Target: next value for ALL cities
        next_idx = i + self.window_size
        y = np.array([self.sequences[city][self.target_feature][next_idx]
                     for city in self.city_names])

        # Feature sequences for metapaths
        feature_sequences = {}
        for feature in ['temperature', 'humidity', 'wind_speed', 'wind_direction']:
            feature_sequences[feature] = np.array([
                self.sequences[city][feature][i:i+self.window_size]
                for city in self.city_names
            ])

        # Build timestamps for the window: hour_of_day and day_of_year
        if self.datetimes is not None:
            window_dt = self.datetimes[i:i+self.window_size]
            # window_dt may be a DatetimeIndex or list of datetime-like objects
            hours = np.array([int(t.hour) for t in window_dt])
            # prefer dayofyear attribute if available
            try:
                days = np.array([int(t.dayofyear) for t in window_dt])
            except Exception:
                days = np.array([int(t.timetuple().tm_yday) for t in window_dt])
        else:
            # fallback: zeros
            hours = np.zeros(self.window_size, dtype=np.int64)
            days = np.zeros(self.window_size, dtype=np.int64)

        timestamps = {
            'hour': torch.tensor(hours, dtype=torch.float32),
            'day': torch.tensor(days, dtype=torch.float32)
        }

        return {
            'X': torch.tensor(X, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'feature_sequences': {k: torch.tensor(v, dtype=torch.float32)
                                for k, v in feature_sequences.items()},
            'timestamps': timestamps
        }


def train_epoch(model, train_loader, optimizer, criterion, device,
                lat_tensor, lon_tensor, epoch, n_epochs):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
    for batch in pbar:
        X = batch['X'].to(device)  # [batch, N, window, n_features]
        y = batch['y'].to(device)  # [batch, N]
        feature_sequences = {k: v.to(device) for k, v in batch['feature_sequences'].items()}
        # timestamps is a dict with tensors [batch, lookback]
        timestamps = {k: v.to(device) for k, v in batch['timestamps'].items()}

        optimizer.zero_grad()

        # Forward pass (pass temporal information)
        predictions, _ = model(X, lat_tensor, lon_tensor, feature_sequences, timestamps)

        # Compute loss
        loss = criterion(predictions, y)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches
    return avg_loss


def evaluate(model, data_loader, criterion, device, lat_tensor, lon_tensor, split_name="Val", scaler=None):
    """
    Evaluate model on validation or test set.
    Returns MAE and loss (in original scale if scaler provided).
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
    all_predictions = np.concatenate(all_predictions, axis=0)  # [total_samples, N_cities]
    all_targets = np.concatenate(all_targets, axis=0)  # [total_samples, N_cities]

    # Inverse transform to original scale if scaler is provided
    if scaler is not None:
        all_predictions = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        all_targets = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
    else:
        all_predictions = all_predictions.flatten()
        all_targets = all_targets.flatten()

    # Calculate MAE across all cities and samples
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))

    return avg_loss, mae, rmse, all_predictions, all_targets

def main():
    # Hyperparameters
    window_size = 24
    hidden_dim = 128
    n_epochs = 35
    learning_rate = 1e-3
    weight_decay = 1e-4
    dropout = 0.2
    max_wind_lag = 6
    k_geo_neighbors = 5
    k_feature_neighbors = 5
    
    # Data split ratios (matching benchmark paper)
    train_ratio = 0.8
    val_ratio = 0.10
    # test_ratio = 0.05 (implicit)
    
    # Step (stride) between successive windows (matching benchmark paper)
    step = 6
    
    # SWA (Stochastic Weight Averaging) hyperparameters
    swa_start_epoch = 20
    swa_mae_threshold = 0.901  # Only average models with MAE < 0.901

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    df, cities = load_data()
    df = df[50:]  # Skip first 50 samples

    # Input features and target
    input_features = ["pressure", "humidity", "temperature", "wind_speed"]
    target_feature = "wind_speed"

    # Prepare sequences with PROPER normalization (only on train data)
    print("Preparing sequences with temporal split and proper normalization...")
    sequences, scalers, city_names, city_latitudes, city_longitudes, datetimes, split_indices = prepare_sequences(
        df, cities, input_features, target_feature, train_ratio, val_ratio
    )

    # Create data splits using temporal boundaries
    print("Creating temporally-split data loaders...")
    train_loader, val_loader, test_loader = create_data_splits(
        sequences, city_names, input_features, target_feature, window_size, step=step,
        datetimes=datetimes, split_indices=split_indices
    )

    # Initialize model
    print("Initializing model...")
    n_stations = len(city_names)
    n_features = len(input_features)

    model = MetapathGNN(
        n_stations=n_stations,
        n_features=n_features,
        hidden_dim=hidden_dim,
        dropout=dropout,
        max_wind_lag=max_wind_lag,
        k_geo_neighbors=k_geo_neighbors,
        k_feature_neighbors=k_feature_neighbors
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Use MAE Loss (L1) - more robust for wind speed prediction
    criterion = nn.L1Loss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # City coordinates
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
    best_model_path = 'checkpoints/metapath_model_best_fixed.pth'
    
    # SWA: Track good models for averaging
    swa_model_states = []  # List to store model state dicts
    swa_model_epochs = []  # Track which epochs were included
    swa_model_maes = []    # Track MAE values of included models

    os.makedirs('checkpoints', exist_ok=True)

    print("\n" + "="*60)
    print("STARTING TRAINING (FIXED VERSION - NO DATA LEAKAGE)")
    print("="*60)

    for epoch in range(n_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                               lat_tensor, lon_tensor, epoch, n_epochs)

        # Evaluate on validation set
        val_loss, val_mae, val_rmse, _, _ = evaluate(
            model, val_loader, criterion, device, lat_tensor, lon_tensor, "Val", scalers[target_feature]
        )

        # Evaluate on test set (to monitor test performance during training)
        test_loss, test_mae, test_rmse, _, _ = evaluate(
            model, test_loader, criterion, device, lat_tensor, lon_tensor, "Test", scalers[target_feature]
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
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f} | Test RMSE: {test_rmse:.4f}")
        print(f"Val-Test MAE Gap: {(test_mae - val_mae):.4f} ({((test_mae - val_mae) / val_mae * 100):.2f}%)")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'history': history
            }, best_model_path)
            print(f"✓ Saved best model (Val MAE: {val_mae:.4f}, Test MAE: {test_mae:.4f})")
        
        # SWA: Collect good models for averaging (only if MAE < threshold)
        if epoch >= swa_start_epoch and val_mae < swa_mae_threshold:
            # Deep copy model state to CPU to save memory
            model_state_copy = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            swa_model_states.append(model_state_copy)
            swa_model_epochs.append(epoch + 1)
            swa_model_maes.append(val_mae)
            print(f"✓ Added to SWA pool (epoch {epoch+1}, MAE: {val_mae:.4f}) - Total models in pool: {len(swa_model_states)}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Create SWA model if we have collected enough good models
    swa_model = None
    if len(swa_model_states) >= 2:
        print(f"\n{'='*60}")
        print(f"CREATING SWA MODEL")
        print(f"{'='*60}")
        print(f"Number of models to average: {len(swa_model_states)}")
        print(f"Epochs included: {swa_model_epochs}")
        print(f"MAE values: {[f'{mae:.4f}' for mae in swa_model_maes]}")
        print(f"Average MAE of included models: {np.mean(swa_model_maes):.4f}")
        
        # Create averaged state dict
        swa_state_dict = {}
        
        # Average all parameters
        for key in swa_model_states[0].keys():
            # Stack all model parameters and take mean
            stacked_params = torch.stack([state[key] for state in swa_model_states])
            swa_state_dict[key] = stacked_params.mean(dim=0)
        
        # Create new model and load averaged weights
        swa_model = MetapathGNN(
            n_stations=n_stations,
            n_features=n_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_wind_lag=max_wind_lag,
            k_geo_neighbors=k_geo_neighbors,
            k_feature_neighbors=k_feature_neighbors
        ).to(device)
        
        swa_model.load_state_dict(swa_state_dict)
        
        # Save SWA model
        swa_model_path = 'checkpoints/metapath_model_swa_fixed.pth'
        torch.save({
            'model_state_dict': swa_state_dict,
            'swa_epochs': swa_model_epochs,
            'swa_maes': swa_model_maes,
            'n_models_averaged': len(swa_model_states)
        }, swa_model_path)
        print(f"✓ SWA model saved to {swa_model_path}")
    else:
        print(f"\n⚠ Not enough models for SWA (found {len(swa_model_states)}, need at least 2)")
        print(f"  Consider lowering swa_mae_threshold or swa_start_epoch")

    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print("EVALUATING BEST SINGLE MODEL")
    print(f"{'='*60}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_mae, test_rmse, test_predictions, test_targets = evaluate(
        model, test_loader, criterion, device, lat_tensor, lon_tensor, "Test", scalers[target_feature]
    )

    print("\nBest Single Model - Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Evaluate SWA model if available
    swa_test_mae = None
    swa_test_rmse = None
    if swa_model is not None:
        print(f"\n{'='*60}")
        print("EVALUATING SWA MODEL")
        print(f"{'='*60}")
        swa_test_loss, swa_test_mae, swa_test_rmse, swa_test_predictions, swa_test_targets = evaluate(
            swa_model, test_loader, criterion, device, lat_tensor, lon_tensor, "Test (SWA)", scalers[target_feature]
        )
        
        print("\nSWA Model - Test Results:")
        print(f"Test Loss: {swa_test_loss:.4f}")
        print(f"Test MAE: {swa_test_mae:.4f}")
        print(f"Test RMSE: {swa_test_rmse:.4f}")
        
        improvement = test_mae - swa_test_mae
        print(f"\n{'='*60}")
        print(f"SWA Improvement: {improvement:.4f} MAE")
        if improvement > 0:
            print(f"✓ SWA is BETTER by {improvement:.4f} MAE ({improvement/test_mae*100:.2f}%)")
        else:
            print(f"✗ SWA is worse by {abs(improvement):.4f} MAE")
        print(f"{'='*60}")

    # Print comparison with validation set
    print(f"\n{'='*60}")
    print("VALIDATION vs TEST COMPARISON")
    print(f"{'='*60}")
    print(f"Best Val MAE:  {best_val_mae:.4f}")
    print(f"Test MAE:      {test_mae:.4f}")
    print(f"Difference:    {abs(test_mae - best_val_mae):.4f}")
    print(f"Gap:           {((test_mae - best_val_mae) / best_val_mae * 100):.2f}%")
    print(f"{'='*60}")

    # Plot training history
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.title('Training and Validation Loss (FIXED)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.plot(history['val_mae'], label='Val MAE', color='blue')
    plt.plot(history['test_mae'], label='Test MAE', color='red', linestyle='--')
    plt.axhline(y=swa_mae_threshold, color='green', linestyle='--', alpha=0.5, label=f'SWA Threshold ({swa_mae_threshold})')
    if len(swa_model_epochs) > 0:
        # Highlight epochs included in SWA
        for epoch_num in swa_model_epochs:
            plt.axvline(x=epoch_num-1, color='green', alpha=0.2, linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title(f'Val vs Test MAE (SWA: {len(swa_model_epochs)} models)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 3)
    # Plot Val-Test gap over epochs
    val_test_gap = [history['test_mae'][i] - history['val_mae'][i] for i in range(len(history['val_mae']))]
    plt.plot(val_test_gap, label='Test - Val MAE', color='purple')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Epoch')
    plt.ylabel('MAE Gap')
    plt.legend()
    plt.title('Validation-Test Gap Over Time')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.plot(history['val_rmse'], label='Val RMSE', color='orange')
    plt.plot(history['test_rmse'], label='Test RMSE', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('Val vs Test RMSE')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nTraining history plot saved as 'training_history_fixed.png'")

    return model, history, test_mae, test_rmse, swa_model, swa_test_mae, swa_test_rmse


if __name__ == "__main__":
    results = main()
    model, history, test_mae, test_rmse, swa_model, swa_test_mae, swa_test_rmse = results
