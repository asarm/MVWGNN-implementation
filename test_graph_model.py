import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataLoad import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from graph_temporal_model import GraphTemporalModel

# Load data
df, cities = load_data()
df = df[50:]

# Get city list and coordinates
city_names = cities['City'].tolist()
city_latitudes = cities['Latitude'].tolist()
city_longitudes = cities['Longitude'].tolist()

print(f"Number of cities: {len(city_names)}")
print(f"Cities: {city_names}")

# Find city for detailed visualization (Vancouver)
target_city = "Vancouver"
target_city_idx = city_names.index(target_city)
print(f"\nCity for detailed visualization: {target_city} (index: {target_city_idx})")
print(f"Note: Model predicts wind speed for ALL {len(city_names)} stations simultaneously")

# Input features
input_features = ["pressure", "humidity", "temperature", "wind_speed"]
target_feature = "wind_speed"

# Prepare sequences for all cities
sequences = {}  # city: {humidity: [], temperature: [], ...}
for city in city_names:
    imputer = SimpleImputer(strategy="mean")
    imputed_data = imputer.fit_transform(df[city].values)
    sequences[city] = {
        "humidity": imputed_data[:, 0],
        "temperature": imputed_data[:, 1],
        "pressure": imputed_data[:, 2],
        "wind_speed": imputed_data[:, 4],
        "wind_direction": imputed_data[:, 3],
    }

# Normalize all features across all cities
scalers = {}
features_to_scale = list(dict.fromkeys(input_features + [target_feature]))

for feature in features_to_scale:
    # Collect all data for this feature across all cities
    all_feature_data = []
    for city in city_names:
        all_feature_data.extend(sequences[city][feature])
    
    scaler = StandardScaler()
    scaler.fit(np.array(all_feature_data).reshape(-1, 1))
    scalers[feature] = scaler
    
    # Apply scaling to all cities
    for city in city_names:
        sequences[city][feature] = scaler.transform(
            sequences[city][feature].reshape(-1, 1)
        ).flatten()


def prepare_graph_data(city_names, sequences, input_features, target_feature, 
                       window_size=24, step=1, target_city_idx=0, df_index=None):
    """
    Prepare windowed data for all cities simultaneously.
    
    Returns:
        X: [n_samples, n_cities, window_size, n_features]
        y: [n_samples, n_cities] - target values for all cities
        last_values: [n_samples, n_cities] - last observed values for all cities
        wind_dirs: [n_samples, n_cities] - wind directions for all cities (in degrees)
        dayofyear: [n_samples] - day of year for each sample
        hourofday: [n_samples] - hour of day for each sample
    """
    n_cities = len(city_names)
    n_features = len(input_features)
    
    # Get the minimum sequence length across all cities
    min_length = min(len(sequences[city][input_features[0]]) for city in city_names)
    
    X, y, last_values, wind_dirs, dayofyear_list, hourofday_list = [], [], [], [], [], []
    
    for i in range(0, min_length - window_size - 1, step):
        # Collect window for all cities
        city_windows = []
        
        for city in city_names:
            window_features = []
            for feature in input_features:
                window_features.append(sequences[city][feature][i:i+window_size])
            city_windows.append(np.stack(window_features, axis=1))  # [window_size, n_features]
        
        # Stack all cities: [n_cities, window_size, n_features]
        multi_city_window = np.stack(city_windows, axis=0)
        
        # Target: next value for ALL cities (vector)
        next_idx = i + window_size
        target_values = [sequences[city][target_feature][next_idx] for city in city_names]
        last_values_vec = [sequences[city][target_feature][i + window_size - 1] for city in city_names]
        
        # Wind directions at the prediction time (next_idx) for all cities
        wind_dir_values = [sequences[city]["wind_direction"][next_idx] for city in city_names]
        
        # Temporal information: use the timestamp at the end of the window
        if df_index is not None:
            timestamp_idx = i + window_size - 1
            timestamp = df_index[timestamp_idx]
            day_of_year = timestamp.dayofyear
            hour_of_day = timestamp.hour
        else:
            day_of_year = 0
            hour_of_day = 0
        
        # Check for NaN/Inf in targets or windows
        target_values_arr = np.array(target_values)
        if np.isnan(target_values_arr).any() or np.isinf(target_values_arr).any():
            continue
        if np.isnan(multi_city_window).any() or np.isinf(multi_city_window).any():
            continue
        
        X.append(multi_city_window)
        y.append(target_values_arr)
        last_values.append(np.array(last_values_vec))
        wind_dirs.append(np.array(wind_dir_values))
        dayofyear_list.append(day_of_year)
        hourofday_list.append(hour_of_day)
    
    return (np.array(X), np.array(y), np.array(last_values), 
            np.array(wind_dirs), np.array(dayofyear_list), np.array(hourofday_list))


class GraphWeatherDataset(Dataset):
    def __init__(self, city_names, sequences, input_features, target_feature, 
                 window_size=24, step=1, target_city_idx=0, df_index=None):
        (self.X, self.y, self.last_values, self.wind_dirs,
         self.dayofyear, self.hourofday) = prepare_graph_data(
            city_names, sequences, input_features, target_feature, 
            window_size, step, target_city_idx, df_index
        )
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.last_values[idx], dtype=torch.float32),
            torch.tensor(self.wind_dirs[idx], dtype=torch.float32),
            torch.tensor(self.dayofyear[idx], dtype=torch.float32),
            torch.tensor(self.hourofday[idx], dtype=torch.float32)
        )


# Hyperparameters
window_size = 24
step = 1
batch_size = 32
learning_rate = 0.001 # was 0.001
num_epochs = 5
num_gcn_layers = 2 # was 2

# ====== TEMPORAL ENCODER SEÇİMİ ======
# Seçenekler: 'conv1d', 'dilated', 'stacked', 'wavenet'
temporal_encoder_type = 'conv1d' 

print(f"\n{'='*70}")
print(f"TEMPORAL ENCODER: {temporal_encoder_type.upper()}")
print(f"{'='*70}")

# Create dataset
dataset = GraphWeatherDataset(
    city_names, sequences, input_features, target_feature,
    window_size=window_size, step=step, target_city_idx=target_city_idx,
    df_index=df.index
)

# Data validation
print(f"\nTotal samples: {len(dataset)}")
sample_X, sample_y, sample_last, sample_wind_dirs, sample_day, sample_hour = dataset[0]
print(f"Sample X shape: {sample_X.shape}")  # [n_cities, window_size, n_features]
print(f"Sample y shape: {sample_y.shape}")  # [n_cities]
print(f"Sample wind_dirs shape: {sample_wind_dirs.shape}")  # [n_cities]
print(f"Sample dayofyear: {sample_day.item()}, hourofday: {sample_hour.item()}")
print(f"X has NaN: {torch.isnan(sample_X).any()}, y has NaN: {torch.isnan(sample_y).any()}")
print(f"Wind directions (degrees): min={sample_wind_dirs.min():.2f}, max={sample_wind_dirs.max():.2f}")

# Train/test split (temporal split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_indices = list(range(0, train_size))
test_indices = list(range(train_size, len(dataset)))

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = GraphTemporalModel(
    n_stations=len(city_names),
    n_features=len(input_features),
    temporal_embed_dim=256,
    positional_embed_dim=32,
    gcn_hidden_dim=256,
    num_gcn_layers=num_gcn_layers,
    dropout=0.3,
    target_station_idx=target_city_idx,
    temporal_encoder_type=temporal_encoder_type
).to(device)

print("\nModel architecture:")
print(model)
print(f"\nTemporal Encoder Type: {temporal_encoder_type}")
print(f"Number of GCN layers: {num_gcn_layers}")
print(f"Self-loops: Removed (using residual connections instead)")
print(f"Prediction mode: All {len(city_names)} stations simultaneously")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")


# Convert city coordinates to tensors (for positional encoder)
city_latitudes_tensor = torch.tensor(city_latitudes, dtype=torch.float32, device=device)
city_longitudes_tensor = torch.tensor(city_longitudes, dtype=torch.float32, device=device)

# Loss and optimizer
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss(beta=0.6)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5) # 1e-5

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=0.0001,
    max_lr=learning_rate,
    step_size_up=10,  # 10 epochs up
    step_size_down=10, # 10 epochs down
    mode='triangular2',
    cycle_momentum=False
)

# Training loop
print(f"\nTraining on {device}")
print(f"Train samples: {train_size}, Test samples: {test_size}\n")

train_losses = []
test_losses = []
adj_densities = []  # Track adjacency matrix densities

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    epoch_densities = []
    
    for batch_idx, (batch_X, batch_y, batch_last, batch_wind_dirs, batch_day, batch_hour) in enumerate(train_loader):
        batch_X = batch_X.to(device)  # [batch, n_cities, window_size, n_features]
        batch_y = batch_y.to(device)  # [batch, n_cities]
        batch_wind_dirs = batch_wind_dirs.to(device)  # [batch, n_cities]
        batch_day, batch_hour = batch_day.to(device), batch_hour.to(device)
        
        # Forward pass - returns predictions for all nodes
        predicted = model(
            batch_X,
            latitude=city_latitudes_tensor,
            longitude=city_longitudes_tensor,
            dayofyear=batch_day,
            hourofday=batch_hour,
            wind_directions=batch_wind_dirs # TODO: None
        )  # [batch, n_cities]
        
        # Calculate adjacency density for first batch of first epoch
        if epoch == 0 and batch_idx == 0:
            with torch.no_grad():
                # Get positional embeddings
                batch_size = batch_X.shape[0]
                station_ids = torch.arange(len(city_names), device=device).unsqueeze(0).expand(batch_size, -1)
                positional_embeddings = model.positional_encoder(
                    station_ids, 
                    latitude=city_latitudes_tensor, 
                    longitude=city_longitudes_tensor
                )
                
                # Compute adjacency
                adj = model.compute_adjacency(
                    positional_embeddings,
                    remove_self_loops=True
                )
                
                # Calculate density (percentage of non-zero entries)
                # For softmax normalized adjacency, we consider values > threshold as edges
                threshold = 1.0 / len(city_names)  # Uniform distribution baseline
                adj_binary = (adj > threshold).float()
                density = adj_binary.mean().item()
                
                # Also calculate mean and std of adjacency values
                adj_mean = adj.mean().item()
                adj_std = adj.std().item()
                adj_max = adj.max().item()
                adj_min = adj.min().item()
                
                print(f"\nAdjacency Matrix Statistics (Epoch 1, Batch 1):")
                print(f"  Shape: {adj.shape}")
                print(f"  Self-loops: REMOVED (using residual connections)")
                print(f"  Density (values > {threshold:.4f}): {density*100:.2f}%")
                print(f"  Mean: {adj_mean:.4f}, Std: {adj_std:.4f}")
                print(f"  Min: {adj_min:.4f}, Max: {adj_max:.4f}")
                print(f"  Expected uniform value (no self-loop): {1.0/(len(city_names)-1):.4f}")
                
                # Print first adjacency matrix
                print(f"\nFirst Adjacency Matrix (batch sample 0):")
                adj_sample = adj[0].cpu().numpy()
                print("  Connections from each city (top 5 weights):")
                for i, city in enumerate(city_names):
                    top_indices = np.argsort(adj_sample[i])[::-1][:5]
                    top_weights = adj_sample[i][top_indices]
                    top_cities = [city_names[idx] for idx in top_indices]
                    print(f"    {city:15s} -> {', '.join([f'{c}({w:.3f})' for c, w in zip(top_cities, top_weights)])}")
        
        # Compute loss on all nodes
        loss = criterion(predicted, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # Testing
    model.eval()
    test_loss = 0.0
    test_predictions_all = []
    test_targets_all = []
    
    with torch.no_grad():
        for batch_X, batch_y, batch_last, batch_wind_dirs, batch_day, batch_hour in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)  # [batch, n_cities]
            batch_wind_dirs = batch_wind_dirs.to(device)  # [batch, n_cities]
            batch_day, batch_hour = batch_day.to(device), batch_hour.to(device)
            
            predicted = model(
                batch_X,
                latitude=city_latitudes_tensor,
                longitude=city_longitudes_tensor,
                dayofyear=batch_day,
                hourofday=batch_hour,
                wind_directions=batch_wind_dirs
            )  # [batch, n_cities]
            
            # Compute loss on all nodes
            loss = criterion(predicted, batch_y)
            test_loss += loss.item()
            
            # Store for MAE calculation in original scale
            test_predictions_all.append(predicted.cpu().numpy())
            test_targets_all.append(batch_y.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    # Calculate MAE in original scale
    test_predictions_concat = np.concatenate(test_predictions_all)  # [n_test_samples, n_cities]
    test_targets_concat = np.concatenate(test_targets_all)
    
    # Inverse transform to original scale
    scaler_wind = scalers[target_feature]
    test_predictions_flat = test_predictions_concat.reshape(-1, 1)
    test_targets_flat = test_targets_concat.reshape(-1, 1)
    test_predictions_inv = scaler_wind.inverse_transform(test_predictions_flat).reshape(test_predictions_concat.shape)
    test_targets_inv = scaler_wind.inverse_transform(test_targets_flat).reshape(test_targets_concat.shape)
    
    # Calculate average MAE across all stations in original scale
    test_mae_original = np.mean([mean_absolute_error(test_targets_inv[:, i], test_predictions_inv[:, i]) 
                                  for i in range(len(city_names))])
    
    # Calculate adjacency density every 10 epochs
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            # Sample a batch to check adjacency
            sample_batch = next(iter(train_loader))
            batch_X = sample_batch[0].to(device)
            batch_size = batch_X.shape[0]
            station_ids = torch.arange(len(city_names), device=device).unsqueeze(0).expand(batch_size, -1)
            positional_embeddings = model.positional_encoder(
                station_ids, 
                latitude=city_latitudes_tensor, 
                longitude=city_longitudes_tensor
            )
            adj = model.compute_adjacency(
                positional_embeddings,
                remove_self_loops=True
            )
            
            threshold = 1.0 / len(city_names)
            adj_binary = (adj > threshold).float()
            density = adj_binary.mean().item()
            adj_densities.append(density)
    
    if (epoch + 1) % 2 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        current_alpha = model.alpha.item()
        current_sigma = model.sigma.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE (m/s): {test_mae_original:.4f}, LR: {current_lr:.6f}, Alpha: {current_alpha:.4f}, Sigma: {current_sigma:.2f}")
    
    # Step the scheduler based on test loss
    scheduler.step()

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'Training and Test Loss - Graph Model (Target: {target_city})')
plt.legend()
plt.grid(True)
plt.savefig('results/graph_training_loss.png')
print("\nLoss curve saved as 'results/graph_training_loss.png'")

# Evaluate on test set
model.eval()
all_predictions = []
all_targets = []

# Analyze final adjacency matrix
print("\n" + "="*70)
print("FINAL ADJACENCY MATRIX ANALYSIS")
print("="*70)
with torch.no_grad():
    # Get a sample batch
    sample_batch = next(iter(test_loader))
    batch_X = sample_batch[0].to(device)
    batch_size = batch_X.shape[0]
    station_ids = torch.arange(len(city_names), device=device).unsqueeze(0).expand(batch_size, -1)
    
    positional_embeddings = model.positional_encoder(
        station_ids, 
        latitude=city_latitudes_tensor, 
        longitude=city_longitudes_tensor
    )
    
    adj = model.compute_adjacency(
        positional_embeddings,
        remove_self_loops=True
    )
    
    # Use first sample's adjacency matrix for analysis
    adj_final = adj[0].cpu().numpy()
    
    # Calculate statistics
    threshold = 1.0 / len(city_names)
    density = (adj_final > threshold).mean()
    
    print(f"\nAdjacency Matrix Shape: {adj_final.shape}")
    print(f"Self-loops: REMOVED (diagonal should be 0 or very small)")
    print(f"Diagonal values: min={np.diag(adj_final).min():.6f}, max={np.diag(adj_final).max():.6f}, mean={np.diag(adj_final).mean():.6f}")
    print(f"Density (edges > {threshold:.4f}): {density*100:.2f}%")
    print(f"Mean: {adj_final.mean():.4f}, Std: {adj_final.std():.4f}")
    print(f"Min: {adj_final.min():.4f}, Max: {adj_final.max():.4f}")
    
    # Show strongest connections for target city (Vancouver)
    print(f"\n{target_city}'s Strongest Connections (Top 10) - for visualization:")
    vancouver_connections = adj_final[target_city_idx]
    top_indices = np.argsort(vancouver_connections)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        weight = vancouver_connections[idx]
        city = city_names[idx]
        marker = " (self)" if idx == target_city_idx else ""
        print(f"  {rank:2d}. {city:20s}: {weight:.4f}{marker}")
    
    # Show all cities' top 3 connections
    print(f"\nAll Cities' Top 3 Connections:")
    for i, city in enumerate(city_names):
        top_indices = np.argsort(adj_final[i])[::-1][:3]
        connections = [(city_names[idx], adj_final[i][idx]) for idx in top_indices]
        conn_str = ", ".join([f"{c}({w:.3f})" for c, w in connections])
        marker = " <- TARGET" if i == target_city_idx else ""
        print(f"  {city:20s} -> {conn_str}{marker}")

with torch.no_grad():
    for batch_X, batch_y, batch_last, batch_wind_dirs, batch_day, batch_hour in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)  # [batch, n_cities]
        batch_wind_dirs = batch_wind_dirs.to(device)  # [batch, n_cities]
        batch_day, batch_hour = batch_day.to(device), batch_hour.to(device)
        
        predicted = model(
            batch_X,
            latitude=city_latitudes_tensor,
            longitude=city_longitudes_tensor,
            dayofyear=batch_day,
            hourofday=batch_hour,
            wind_directions=batch_wind_dirs
        )  # [batch, n_cities]
        
        all_predictions.append(predicted.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

torch.save(model.state_dict(), 'checkpoints/graph_model_best.pth')
# Convert to numpy arrays
all_predictions = np.concatenate(all_predictions)  # [n_samples, n_cities]
all_targets = np.concatenate(all_targets)        # [n_samples, n_cities]

# Inverse transform to original scale per value
scaler_wind = scalers[target_feature]
all_predictions_flat = all_predictions.reshape(-1, 1)
all_targets_flat = all_targets.reshape(-1, 1)
all_predictions_inv = scaler_wind.inverse_transform(all_predictions_flat).reshape(all_predictions.shape)
all_targets_inv = scaler_wind.inverse_transform(all_targets_flat).reshape(all_targets.shape)

print(f"\nAll Stations - Prediction Statistics (Original Scale):")
print(f"Predictions shape: {all_predictions_inv.shape}")
print(f"Target range: [{all_targets_inv.min():.2f}, {all_targets_inv.max():.2f}]")
print(f"Prediction range: [{all_predictions_inv.min():.2f}, {all_predictions_inv.max():.2f}]")

# Calculate metrics for ALL stations (averaged)
mae_per_station = []
rmse_per_station = []
r2_per_station = []

for i, city in enumerate(city_names):
    city_preds = all_predictions_inv[:, i]
    city_targets = all_targets_inv[:, i]
    
    mae = mean_absolute_error(city_targets, city_preds)
    rmse = np.sqrt(mean_squared_error(city_targets, city_preds))
    r2 = r2_score(city_targets, city_preds)
    
    mae_per_station.append(mae)
    rmse_per_station.append(rmse)
    r2_per_station.append(r2)

# Average metrics across all stations
avg_mae = np.mean(mae_per_station)
avg_rmse = np.mean(rmse_per_station)
avg_r2 = np.mean(r2_per_station)

print(f"\n{'='*70}")
print(f"AVERAGE METRICS ACROSS ALL {len(city_names)} STATIONS (1-hour ahead):")
print(f"{'='*70}")
print(f"Average MAE:  {avg_mae:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average R²:   {avg_r2:.4f}")

# Show per-station metrics
print(f"\nPer-Station Metrics:")
print(f"{'City':<20s} {'MAE':>8s} {'RMSE':>8s} {'R²':>8s}")
print("-" * 50)
for i, city in enumerate(city_names):
    marker = " <- TARGET" if i == target_city_idx else ""
    print(f"{city:<20s} {mae_per_station[i]:8.4f} {rmse_per_station[i]:8.4f} {r2_per_station[i]:8.4f}{marker}")

# Focus on target city (Vancouver) for detailed analysis
vancouver_preds = all_predictions_inv[:, target_city_idx]
vancouver_targets = all_targets_inv[:, target_city_idx]

print(f"\n{'='*70}")
print(f"DETAILED METRICS FOR {target_city} (for visualization):")
print(f"{'='*70}")
print(f"MAE:  {mae_per_station[target_city_idx]:.4f}")
print(f"RMSE: {rmse_per_station[target_city_idx]:.4f}")
print(f"R²:   {r2_per_station[target_city_idx]:.4f}")

# Plot predictions vs actual for Vancouver
plt.figure(figsize=(15, 5))
n_samples_to_plot = min(300, len(vancouver_preds))

plt.subplot(1, 2, 1)
plt.plot(vancouver_targets[:n_samples_to_plot], label='Actual', alpha=0.7)
plt.plot(vancouver_preds[:n_samples_to_plot], label='Predicted', alpha=0.7)
plt.xlabel('Sample')
plt.ylabel('Wind Speed (m/s)')
plt.title(f'{target_city} - 1-Hour Ahead Prediction (Example Station)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(vancouver_targets, vancouver_preds, alpha=0.3)
plt.plot([vancouver_targets.min(), vancouver_targets.max()],
         [vancouver_targets.min(), vancouver_targets.max()],
         'r--', label='Perfect prediction')
plt.xlabel('Actual Wind Speed (m/s)')
plt.ylabel('Predicted Wind Speed (m/s)')
plt.title(f'{target_city} - Scatter Plot (Example Station)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('results/graph_predictions.png')
print("\nPredictions plot saved as 'results/graph_predictions.png'")
print(f"(Showing {target_city} as example - model predicts all {len(city_names)} stations)")

# Save model
torch.save(model.state_dict(), 'checkpoints/graph_model_best.pth')
print(f"\nModel saved as 'checkpoints/graph_model_best.pth'")
