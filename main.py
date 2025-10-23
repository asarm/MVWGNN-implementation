from model import DDGNNWind
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup
n_stations = 32
seq_len = 24
hidden_dim = 32

model = DDGNNWind(
    n_stations=n_stations,
    hidden_dim=hidden_dim,
    n_heads=4,
    seq_len=seq_len,
    n_gnn_layers=3,
).to(device)

# Sample data
historical_data = torch.randn(n_stations, seq_len, 5)  # (stations, time, features)
lat = torch.rand(n_stations)  # normalized [0, 1]
lon = torch.rand(n_stations)
wind_direction = torch.rand(n_stations) * 360  # [0, 360) degrees
positions = torch.stack([lat, lon], dim=1)

# Move inputs to device so model and tensors are colocated
historical_data = historical_data.to(device)
lat = lat.to(device)
lon = lon.to(device)
wind_direction = wind_direction.to(device)
positions = positions.to(device)

current_hour = 14  # 2 PM
day_of_year = 150  # May 30th

# Forward pass
predictions = model(
    historical_data,
    lat, lon,
    current_hour, day_of_year,
    wind_direction_deg=wind_direction,
    positions=positions
)

# Print results
print("Predictions:")
for horizon, pred in predictions.items():
    print(f"  {horizon}: shape {pred.shape}, mean {pred.mean():.3f}, std {pred.std():.3f}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")