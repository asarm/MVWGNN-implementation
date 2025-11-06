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

from graph_layers import GraphSAGELayer, haversine_distance, calculate_bearing
from graph_layers import GeographicMetapath, FeatureSimilarityMetapath, MultiTemporalWindMetapath, MetapathFusion

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
target_city = "New York"
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
# IMPORTANT: wind_direction is circular (0-360°) and should not be
# normalized with StandardScaler. Exclude it explicitly here.
scalers = {}
features_to_scale = [f for f in dict.fromkeys(input_features + [target_feature]) if f != 'wind_direction']

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


# Test code (main.py'nin sonuna ekleyin)
print("\n" + "="*60)
print("PHASE 1: Testing GraphSAGE Layer")
print("="*60)

# Create dummy data
batch_size = 2
n_cities = len(city_names)
hidden_dim = 32
test_features = torch.randn(batch_size, n_cities, hidden_dim)

# Create dummy adjacency (identity for now)
test_adj = torch.eye(n_cities).unsqueeze(0).expand(batch_size, -1, -1)

# Initialize layer
sage_layer = GraphSAGELayer(hidden_dim, hidden_dim)

# Forward pass
output = sage_layer(test_features, test_adj)

print(f"✓ Input shape: {test_features.shape}")
print(f"✓ Adjacency shape: {test_adj.shape}")
print(f"✓ Output shape: {output.shape}")
print(f"✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

# Check for NaN
if torch.isnan(output).any():
    print("✗ ERROR: Output contains NaN!")
else:
    print("✓ No NaN in output")

print("\n" + "="*60)
print("PHASE 1: Testing Distance/Bearing Functions")
print("="*60)

# Convert to tensors
lat_tensor = torch.tensor(city_latitudes, dtype=torch.float32)
lon_tensor = torch.tensor(city_longitudes, dtype=torch.float32)

# Test haversine
distances = haversine_distance(lat_tensor, lon_tensor, lat_tensor, lon_tensor)
print(f"✓ Distance matrix shape: {distances.shape}")
print(f"✓ Diagonal (self-distance): {distances.diagonal()[:5].tolist()}")
print(f"✓ Distance range: [{distances.min().item():.1f}, {distances.max().item():.1f}] km")

# Test bearing
bearings = calculate_bearing(lat_tensor, lon_tensor, lat_tensor, lon_tensor)
print(f"✓ Bearing matrix shape: {bearings.shape}")
print(f"✓ Bearing range: [{bearings.min().item():.1f}, {bearings.max().item():.1f}] degrees")

# Test specific pair (Vancouver to another city)
if n_cities > 1:
    dist_01 = distances[0, 1].item()
    bear_01 = bearings[0, 1].item()
    print(f"\n✓ Example: {city_names[0]} → {city_names[1]}")
    print(f"  Distance: {dist_01:.1f} km")
    print(f"  Bearing: {bear_01:.1f}°")


print("\n" + "="*60)
print("PHASE 2: Testing Geographic Metapath")
print("="*60)

# Initialize geographic metapath
geo_metapath = GeographicMetapath(
    hidden_dim=hidden_dim,
    k_neighbors=5,
    distance_scale=200.0
)

# Test forward pass
output_geo = geo_metapath(test_features, lat_tensor, lon_tensor)

print(f"✓ Input shape: {test_features.shape}")
print(f"✓ Output shape: {output_geo.shape}")
print(f"✓ Output range: [{output_geo.min().item():.3f}, {output_geo.max().item():.3f}]")

# Check for NaN
if torch.isnan(output_geo).any():
    print("✗ ERROR: Output contains NaN!")
else:
    print("✓ No NaN in output")

# Inspect adjacency matrix
A_geo = geo_metapath._cached_adjacency
print(f"\n✓ Adjacency matrix shape: {A_geo.shape}")
print(f"✓ Adjacency sparsity: {(A_geo == 0).sum().item() / A_geo.numel() * 100:.1f}% zeros")
print(f"✓ Row sums (should be ~1.0): {A_geo.sum(dim=-1)[:5].tolist()}")

# Visualize Vancouver's neighbors
vancouver_idx = target_city_idx
vancouver_neighbors = torch.nonzero(A_geo[vancouver_idx] > 0).squeeze()
vancouver_weights = A_geo[vancouver_idx, vancouver_neighbors]

print(f"\n✓ {target_city} (index {vancouver_idx}) has {len(vancouver_neighbors)} neighbors:")
for i, (neighbor_idx, weight) in enumerate(zip(vancouver_neighbors, vancouver_weights)):
    neighbor_name = city_names[neighbor_idx.item()]
    dist = distances[vancouver_idx, neighbor_idx].item()
    print(f"  {i+1}. {neighbor_name:15s} (weight={weight.item():.3f}, dist={dist:.0f}km)")


print("\n" + "="*60)
print("PHASE 3: Testing Feature Similarity Metapath")
print("="*60)

# Prepare test feature data (temperature example)
# Extract temperature sequences for all cities
window_size = 48
temperature_sequences = []
for city in city_names:
    temp_seq = sequences[city]["temperature"][:window_size]
    temperature_sequences.append(temp_seq)

# Stack: [N_cities, window_size]
temp_data = np.stack(temperature_sequences, axis=0)
temp_tensor = torch.tensor(temp_data, dtype=torch.float32).unsqueeze(0)  # [1, N, window]

# Expand for batch
temp_tensor_batch = temp_tensor.expand(batch_size, -1, -1)  # [batch, N, window]

print(f"✓ Temperature data shape: {temp_tensor_batch.shape}")
print(f"✓ Temperature range: [{temp_tensor_batch.min().item():.2f}, {temp_tensor_batch.max().item():.2f}]")

# Initialize feature metapath
temp_metapath = FeatureSimilarityMetapath(
    hidden_dim=hidden_dim,
    feature_name='temperature',
    top_k=8,
    lookback_hours=24
)

# Test forward pass
output_temp = temp_metapath(test_features, temp_tensor_batch)

print(f"\n✓ Input features shape: {test_features.shape}")
print(f"✓ Output shape: {output_temp.shape}")
print(f"✓ Output range: [{output_temp.min().item():.3f}, {output_temp.max().item():.3f}]")

# Check for NaN
if torch.isnan(output_temp).any():
    print("✗ ERROR: Output contains NaN!")
else:
    print("✓ No NaN in output")

# Inspect adjacency for first batch
A_temp = temp_metapath.construct_similarity_adjacency(temp_tensor_batch)
A_temp_0 = A_temp[0]  # First batch

print(f"\n✓ Temperature adjacency shape: {A_temp_0.shape}")
print(f"✓ Adjacency sparsity: {(A_temp_0 == 0).sum().item() / A_temp_0.numel() * 100:.1f}% zeros")
print(f"✓ Row sums: {A_temp_0.sum(dim=-1)[:5].tolist()}")

# Visualize Vancouver's temperature-similar stations
vancouver_temp_neighbors = torch.nonzero(A_temp_0[vancouver_idx] > 0).squeeze()
if vancouver_temp_neighbors.dim() == 0:
    vancouver_temp_neighbors = vancouver_temp_neighbors.unsqueeze(0)

vancouver_temp_weights = A_temp_0[vancouver_idx, vancouver_temp_neighbors]

print(f"\n✓ {target_city} has {len(vancouver_temp_neighbors)} temperature-similar stations:")

# Sort by weight (descending)
sorted_indices = torch.argsort(vancouver_temp_weights, descending=True)
for i in sorted_indices[:8]:  # Show top 8
    neighbor_idx = vancouver_temp_neighbors[i].item()
    weight = vancouver_temp_weights[i].item()
    neighbor_name = city_names[neighbor_idx]
    
    # Show actual temperature values for comparison
    vancouver_temp = temp_tensor_batch[0, vancouver_idx, -24:].mean().item()
    neighbor_temp = temp_tensor_batch[0, neighbor_idx, -24:].mean().item()
    
    print(f"  {neighbor_name:15s} (weight={weight:.3f}, temp_diff={abs(vancouver_temp - neighbor_temp):.2f})")


print("\n" + "="*60)
print("PHASE 4: Testing Multi-Temporal Wind Metapath")
print("="*60)

# Prepare wind direction and speed data
wind_dir_sequences = []
wind_speed_sequences = []
for city in city_names:
    wind_dir_seq = sequences[city]["wind_direction"][:window_size]
    wind_speed_seq = sequences[city]["wind_speed"][:window_size]
    wind_dir_sequences.append(wind_dir_seq)
    wind_speed_sequences.append(wind_speed_seq)

# Stack: [N, window]
wind_dir_data = np.stack(wind_dir_sequences, axis=0)
wind_speed_data = np.stack(wind_speed_sequences, axis=0)

# Convert to tensors: [1, N, window]
wind_dir_tensor = torch.tensor(wind_dir_data, dtype=torch.float32).unsqueeze(0)
wind_speed_tensor = torch.tensor(wind_speed_data, dtype=torch.float32).unsqueeze(0)

# Expand for batch: [batch, N, window]
wind_dir_batch = wind_dir_tensor.expand(batch_size, -1, -1)
wind_speed_batch = wind_speed_tensor.expand(batch_size, -1, -1)

print(f"✓ Wind direction shape: {wind_dir_batch.shape}")
print(f"✓ Wind direction range: [{wind_dir_batch.min().item():.1f}°, {wind_dir_batch.max().item():.1f}°]")
print(f"✓ Wind speed shape: {wind_speed_batch.shape}")
print(f"✓ Wind speed range: [{wind_speed_batch.min().item():.2f}, {wind_speed_batch.max().item():.2f}]")

# Prepare node features with temporal dimension
# Shape: [batch, N, lookback, hidden_dim]
test_features_temporal = test_features.unsqueeze(2).expand(
    batch_size, n_cities, window_size, hidden_dim
)

print(f"\n✓ Node features (temporal) shape: {test_features_temporal.shape}")

# Initialize multi-temporal wind metapath
wind_metapath = MultiTemporalWindMetapath(
    hidden_dim=hidden_dim,
    max_lag=12,
    distance_scale=500.0
)

# Test forward pass
output_wind, lag_attention = wind_metapath(
    test_features_temporal, 
    wind_dir_batch, 
    wind_speed_batch,
    lat_tensor, 
    lon_tensor
)

print(f"\n✓ Output shape: {output_wind.shape}")
print(f"✓ Output range: [{output_wind.min().item():.3f}, {output_wind.max().item():.3f}]")

# Check for NaN
if torch.isnan(output_wind).any():
    print("✗ ERROR: Output contains NaN!")
else:
    print("✓ No NaN in output")

# Visualize learned lag attention
print(f"\n✓ Learned Lag Attention Weights:")
print("  " + "─" * 50)
for tau in range(wind_metapath.max_lag):
    weight = lag_attention[tau].item()
    bar_length = int(weight * 100)
    bar = "█" * bar_length
    print(f"  τ={tau:2d}h: {bar} {weight:.4f}")

# Find most important lags
top_lags = torch.argsort(lag_attention, descending=True)[:3]
print(f"\n✓ Top 3 most important lags: {top_lags.tolist()} hours")
print(f"  Weights: {[lag_attention[i].item() for i in top_lags]}")

# Inspect adjacency for a specific lag
tau_inspect = 3  # 3-hour lag
A_wind_3h = wind_metapath.construct_lag_adjacency(
    wind_dir_batch, wind_speed_batch, lat_tensor, lon_tensor, tau_inspect
)
A_wind_3h_0 = A_wind_3h[0]  # First batch

print(f"\n✓ Wind adjacency at τ={tau_inspect}h:")
print(f"  Shape: {A_wind_3h_0.shape}")
print(f"  Sparsity: {(A_wind_3h_0 == 0).sum().item() / A_wind_3h_0.numel() * 100:.1f}% zeros")
print(f"  Non-zero edges: {(A_wind_3h_0 > 0).sum().item()}")

# Visualize Vancouver's wind-influenced stations at τ=3h
vancouver_wind_neighbors = torch.nonzero(A_wind_3h_0[vancouver_idx] > 0).squeeze()
if vancouver_wind_neighbors.numel() > 0:
    if vancouver_wind_neighbors.dim() == 0:
        vancouver_wind_neighbors = vancouver_wind_neighbors.unsqueeze(0)
    
    vancouver_wind_weights = A_wind_3h_0[vancouver_idx, vancouver_wind_neighbors]
    
    print(f"\n✓ {target_city} influences (at τ=3h): {len(vancouver_wind_neighbors)} stations")
    
    # Sort by weight
    sorted_indices = torch.argsort(vancouver_wind_weights, descending=True)
    for i in sorted_indices[:5]:  # Top 5
        neighbor_idx = vancouver_wind_neighbors[i].item()
        weight = vancouver_wind_weights[i].item()
        neighbor_name = city_names[neighbor_idx]
        dist = distances[vancouver_idx, neighbor_idx].item()
        bearing = bearings[vancouver_idx, neighbor_idx].item()
        
        print(f"  → {neighbor_name:15s} (weight={weight:.3f}, dist={dist:.0f}km, bearing={bearing:.0f}°)")
else:
    print(f"\n✓ {target_city} has no downwind neighbors at τ=3h")


print("\n" + "="*60)
print("PHASE 5: Testing Metapath Fusion")
print("="*60)

# Collect all metapath outputs
metapath_outputs = {
    'wind': output_wind,  # From Phase 4
    'geo': output_geo,    # From Phase 2
    'temp': output_temp,  # From Phase 3
}

print("Metapath embeddings collected:")
for name, embedding in metapath_outputs.items():
    print(f"  - {name:10s}: {embedding.shape}")

# Initialize fusion module
n_metapaths = len(metapath_outputs)
fusion_module = MetapathFusion(
    hidden_dim=hidden_dim,
    n_metapaths=n_metapaths
)

# Test forward pass
output_fused, metapath_attention = fusion_module(metapath_outputs)

print(f"\n✓ Fused output shape: {output_fused.shape}")
print(f"✓ Fused output range: [{output_fused.min().item():.3f}, {output_fused.max().item():.3f}]")

# Check for NaN
if torch.isnan(output_fused).any():
    print("✗ ERROR: Output contains NaN!")
else:
    print("✓ No NaN in fused output")

# Visualize metapath attention
print(f"\n✓ Learned Metapath Importance:")
print("  " + "─" * 50)
for name, weight in metapath_attention.items():
    bar_length = int(weight * 100)
    bar = "█" * bar_length
    print(f"  {name:10s}: {bar} {weight:.4f}")

# Sort by importance
sorted_metapaths = sorted(metapath_attention.items(), key=lambda x: x[1], reverse=True)
print(f"\n✓ Metapath ranking:")
for i, (name, weight) in enumerate(sorted_metapaths, 1):
    print(f"  {i}. {name:10s} ({weight:.4f})")

print("\n" + "="*60)
print("PHASE 6: Testing Complete MetapathGNN Model")
print("="*60)

from metapath_model import MetapathGNN

# Initialize complete model
model = MetapathGNN(
    n_stations=n_cities,
    n_features=len(input_features),  # This is 4
    hidden_dim=64,
    dropout=0.3,
    max_wind_lag=12,
    k_geo_neighbors=5,
    k_feature_neighbors=8
)

print(f"✓ Model initialized")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# FIXED: Prepare REAL input features (not encoded dummy data!)
window_size_test = 48

# Collect raw features for all cities
all_features = []
for city in city_names:
    city_features = []
    for feature_name in input_features:
        feature_seq = sequences[city][feature_name][:window_size_test]
        city_features.append(feature_seq)
    all_features.append(np.stack(city_features, axis=1))  # [window, n_features]

# Stack: [N_cities, window, n_features]
features_array = np.stack(all_features, axis=0)

# Convert to tensor: [1, N, window, n_features] then expand for batch
test_input = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)
test_input = test_input.expand(batch_size, -1, -1, -1)

print(f"\n✓ Input shape: {test_input.shape}")
print(f"✓ Input should be: [batch={batch_size}, N={n_cities}, lookback={window_size_test}, n_features={len(input_features)}]")

# Feature sequences for metapaths
test_sequences = {
    'temperature': temp_tensor_batch,
    'humidity': torch.tensor(
        np.stack([sequences[city]["humidity"][:window_size_test] for city in city_names], axis=0),
        dtype=torch.float32
    ).unsqueeze(0).expand(batch_size, -1, -1),
    'wind_speed': wind_speed_batch,
    'wind_direction': wind_dir_batch,
}

print(f"✓ Feature sequences prepared")
for name, seq in test_sequences.items():
    print(f"  - {name:15s}: {seq.shape}")

# Forward pass
try:
    predictions, attention_info = model(
        test_input,
        lat_tensor,
        lon_tensor,
        test_sequences
    )
    
    print(f"\n✅ FORWARD PASS SUCCESSFUL!")
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Predictions range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
    
    # Check for NaN
    if torch.isnan(predictions).any():
        print("✗ ERROR: Predictions contain NaN!")
    else:
        print("✓ No NaN in predictions")
    
    # Visualize attention
    print(f"\n✓ Attention Information:")
    print("  " + "─" * 50)
    print("  Lag Attention (top 5):")
    lag_attn = attention_info['lag_attention']
    top_lags = torch.argsort(lag_attn, descending=True)[:5]
    for lag in top_lags:
        bar = "█" * int(lag_attn[lag].item() * 100)
        print(f"    τ={lag.item():2d}h: {bar} {lag_attn[lag].item():.4f}")
    
    print("\n  Metapath Attention:")
    for name, weight in sorted(attention_info['metapath_attention'].items(), 
                               key=lambda x: x[1], reverse=True):
        bar = "█" * int(weight * 100)
        print(f"    {name:12s}: {bar} {weight:.4f}")
    
    # Test backward pass
    print(f"\n✓ Testing backward pass...")
    dummy_target = torch.randn_like(predictions)
    loss = F.mse_loss(predictions, dummy_target)
    loss.backward()
    
    print(f"\n✅ BACKWARD PASS SUCCESSFUL!")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Gradients computed successfully")
    
    # Check gradient magnitudes
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    print(f"✓ Gradient norm range: [{min(grad_norms):.6f}, {max(grad_norms):.6f}]")
    
    if max(grad_norms) > 100:
        print("⚠️  WARNING: Large gradients detected! May need gradient clipping.")
    
except Exception as e:
    print(f"\n✗ ERROR during forward/backward pass:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("MODEL INTEGRATION TEST COMPLETE!")
print("="*60)
print(f"\n✓ Model ready for training")
print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")