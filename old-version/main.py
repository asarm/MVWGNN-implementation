from model import DDGNNWind
from dataLoad import load_data
from torchData import GraphWeatherDataset, create_dataloader
import torch
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

df, cities = load_data("hourly-data")
datetime_index = df.index
print(df.sample(5))
print("n stations:", len(cities))
# hours = [int(x.split(" ")[1][:2]) for x in datetime_index[:5]]
hours = datetime_index.hour.tolist()
# day_of_years = [pd.to_datetime(x.split(" ")[0]).timetuple().tm_yday for x in datetime_index[:5]]
day_of_years = datetime_index.dayofyear.tolist()

# Get the first city name for plotting
first_city_name = cities['City'].iloc[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup
n_stations = len(cities)
seq_len = 24
hidden_dim = 64  
train_size = int(0.8 * len(df))

# Multi-horizon prediction: [1] hour ahead
prediction_horizons = [1]
num_horizons = len(prediction_horizons)

# Create weighted loss for different horizons (closer horizons weighted more)
horizon_weights = torch.tensor([1.0], device=device)  
horizon_weights = horizon_weights / horizon_weights.sum()  # Normalize
print(f"Horizon weights: {horizon_weights.cpu().numpy()}")

train_dataset = GraphWeatherDataset(
    df=df.iloc[:train_size],
    cities=cities,
    seq_len=seq_len,
    prediction_window=max(prediction_horizons),  # Changed: max horizon for data preparation
    sliding_window=True
)

test_dataset = GraphWeatherDataset(
    df=df.iloc[train_size:],
    cities=cities,
    seq_len=seq_len,
    prediction_window=max(prediction_horizons),  # Changed: max horizon
    sliding_window=True
)
train_loader = create_dataloader(train_dataset, batch_size=16, shuffle=True)
val_loader = create_dataloader(test_dataset, batch_size=1, shuffle=False)

model = DDGNNWind(
    n_stations=n_stations,
    hidden_dim=hidden_dim,
    n_heads=4,
    seq_len=seq_len,
    n_gnn_layers=1,
    temporal_debug=False

).to(device)

# Multi-horizon loss function
def criterion_multi_horizon(output, target, horizon_weights):
    """
    Compute weighted MSE loss across multiple horizons.
    
    Args:
        output: (B, N, 5) or (N, 5) - predictions for 5 horizons
        target: (B, N, 5) or (N, 5) - ground truth for 5 horizons
        horizon_weights: (5,) - weights for each horizon
    
    Returns:
        weighted_loss: scalar
    """
    # Compute MSE for each horizon
    mse_per_sample = (output - target) ** 2  # (B, N, 5) or (N, 5)
    
    # Average over batch and stations, keep horizons
    if mse_per_sample.dim() == 3:
        # Batched: (B, N, 5) -> (5,)
        mse_per_horizon = mse_per_sample.mean(dim=(0, 1))
    else:
        # Non-batched: (N, 5) -> (5,)
        mse_per_horizon = mse_per_sample.mean(dim=0)
    
    # Apply weights
    weighted_loss = (mse_per_horizon * horizon_weights).sum()
    
    return weighted_loss, mse_per_horizon

criterion = torch.nn.MSELoss(reduction='none')  # For per-element loss
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=0.001,  # Back to 0.001 with MSE loss
                                weight_decay=1e-5)  # Reduced weight decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.003,
    epochs=15,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos'
)
stop_counter = 0
patience = 10

# Create directory for saving models and results
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Initialize loss tracking
train_loss_history = []
val_loss_history = []
val_mae_history = []
# Track per-horizon MAE history
val_horizon_mae_history = {f'horizon_{h}h_mae': [] for h in prediction_horizons}
best_val_loss = float('inf')

def train_epoch(model, epoch):
    model.train()
    total_loss = 0.0
    horizon_losses_sum = {f'horizon_{h}h': 0.0 for h in prediction_horizons}
    
    # Get scaler from the dataset for inverse transform
    scaler = train_loader.dataset.wind_speed_scaler
    scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)  # (n_stations,)
    scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)  # (n_stations,)

    for batch_idx, data in tqdm(enumerate(train_loader)):
        (
            historical_data,
            lat,
            lon,
            current_hour,
            day_of_year,
            wind_dir_deg_last,
            positions,
            target,
        ) = data

        historical_data = historical_data.to(device)
        lat = lat.to(device)
        lon = lon.to(device)
        current_hour = current_hour.to(device)
        day_of_year = day_of_year.to(device)
        wind_dir_deg_last = wind_dir_deg_last.to(device)
        positions = positions.to(device)
        target = target.to(device)

        # print("historical_data.shape (before permute):", historical_data.shape)
        # dataset produces (batch, n_stations, seq, features)
        b, n_st, seq, f = historical_data.shape

        # sanity checks: model expects 5 feature channels and seq_len timesteps
        if f != 5:
            raise ValueError(f"Expected feature dimension 5 but got {f}")
        if seq < seq_len:
            raise ValueError(f"Input sequence length {seq} < required seq_len {seq_len}")
        if seq > seq_len:
            # keep the most recent seq_len timesteps
            historical_data = historical_data[:, :, -seq_len:, :]
            b, n_st, seq, f = historical_data.shape
            print(f"Truncated historical_data to seq_len: new shape {historical_data.shape}")

        # Batched forward/backward: call model once on full batch
        optimizer.zero_grad()

        # Model now accepts batched inputs: (B, N, seq, f)
        # Output: (B, N, 5) - predictions for 5 horizons
        output = model(
            historical_data,
            lat,
            lon,
            current_hour,
            day_of_year,
            wind_dir_deg_last,
            positions,
        )

        # Calculate weighted multi-horizon loss in SCALED space
        # This ensures all stations contribute equally to the loss
        # output/target: (B, N, H)
        weighted_loss, mse_per_horizon = criterion_multi_horizon(output, target, horizon_weights)
        weighted_loss.backward()
        
        # Gradient clipping to prevent exploding gradients and stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        loss_val = weighted_loss.item()
        total_loss += loss_val * b
        
        # Track individual horizon losses
        for i, h in enumerate(prediction_horizons):
            horizon_losses_sum[f'horizon_{h}h'] += mse_per_horizon[i].item() * b

    avg_loss = total_loss / (len(train_loader.dataset))
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.6f}")
    
    # Print individual horizon losses
    for h_name, h_loss_sum in horizon_losses_sum.items():
        avg_h_loss = h_loss_sum / len(train_loader.dataset)
        print(f"  {h_name}: {avg_h_loss:.6f}")
    
    return avg_loss


def eval_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    # Track metrics for each horizon
    horizon_maes = {f'horizon_{h}h': 0.0 for h in prediction_horizons}
    
    # Get scaler from the dataset for inverse transform
    scaler = data_loader.dataset.wind_speed_scaler
    scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)  # (n_stations,)
    scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)  # (n_stations,)
    
    # Store predictions and targets for the first city - for each horizon
    first_city_predictions = {h: [] for h in prediction_horizons}
    first_city_targets = {h: [] for h in prediction_horizons}
    
    for batch_idx, data in enumerate(data_loader):
        (
            historical_data,
            lat,
            lon,
            current_hour,
            day_of_year,
            wind_dir_deg_last,
            positions,
            target,
        ) = data

        historical_data = historical_data.to(device)
        lat = lat.to(device)
        lon = lon.to(device)
        current_hour = current_hour.to(device)
        day_of_year = day_of_year.to(device)
        wind_dir_deg_last = wind_dir_deg_last.to(device)
        positions = positions.to(device)
        target = target.to(device)

        b, n_st, seq, f = historical_data.shape

        if f != 5:
            raise ValueError(f"Expected feature dimension 5 but got {f}")
        if seq < seq_len:
            raise ValueError(f"Input sequence length {seq} < required seq_len {seq_len}")
        if seq > seq_len:
            historical_data = historical_data[:, :, -seq_len:, :]
            b, n_st, seq, f = historical_data.shape
            print(f"Truncated historical_data to seq_len: new shape {historical_data.shape}")

        with torch.no_grad():
            output = model(
                historical_data,
                lat,
                lon,
                current_hour,
                day_of_year,
                wind_dir_deg_last,
                positions,
            )
            
            # Calculate loss in scaled space (for consistency with training)
            loss = criterion(output, target)
            
            # But report metrics in original scale for interpretability
            output_original = output * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)
            target_original = target * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)
            mae = torch.mean(torch.abs(output_original - target_original))

        loss_val = loss.mean().item()
        mae_val = mae.item()
        
        total_loss += loss_val * b
        total_mae += mae_val * b
        total_samples += b
        
        # Track per-horizon MAE and store predictions for first city (station index 1)
        for i, h in enumerate(prediction_horizons):
            mae_h = torch.mean(torch.abs(output_original[:, :, i] - target_original[:, :, i]))
            horizon_maes[f'horizon_{h}h'] += mae_h.item() * b
            
            # Store predictions and targets for the first city
            first_city_predictions[h].append(output_original[:, 1, i].cpu().numpy())  # (B,)
            first_city_targets[h].append(target_original[:, 1, i].cpu().numpy())  # (B,)

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    # Concatenate all predictions and targets for the first city
    import numpy as np
    for h in prediction_horizons:
        first_city_predictions[h] = np.concatenate(first_city_predictions[h])
        first_city_targets[h] = np.concatenate(first_city_targets[h])
    
    print(f"Epoch {epoch}, Validation - Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}")
    
    # Print per-horizon MAE and store in dict for return
    avg_horizon_maes = {}
    for h_name, h_mae in horizon_maes.items():
        avg_h_mae = h_mae / total_samples
        avg_horizon_maes[h_name] = avg_h_mae
        print(f"  {h_name} MAE: {avg_h_mae:.6f}")
    
    return avg_loss, avg_mae, first_city_predictions, first_city_targets, avg_horizon_maes


for epoch in range(1, 15):
    train_loss = train_epoch(model, epoch)
    train_loss_history.append(train_loss)

    if epoch > 0 and epoch % 1 == 0:
        val_loss, val_mae, first_city_preds, first_city_targs, avg_horizon_maes = eval_model(model, val_loader)
        val_loss_history.append(val_loss)
        val_mae_history.append(val_mae)
        
        # Store per-horizon MAE values
        for h_name, h_mae_val in avg_horizon_maes.items():
            val_horizon_mae_history[f'{h_name}_mae'].append(h_mae_val)
        
        # Create subplot for each horizon
        fig, axes = plt.subplots(len(prediction_horizons), 1, figsize=(14, 3*len(prediction_horizons)))
        if len(prediction_horizons) == 1:
            axes = [axes]
        
        for idx, h in enumerate(prediction_horizons):
            preds = first_city_preds[h]
            targs = first_city_targs[h]
            
            plot_length = min(150, len(preds))
            time_steps = range(plot_length)
            
            axes[idx].plot(time_steps, targs[-plot_length:], 'b-', label='Target', linewidth=1.5, alpha=0.7)
            axes[idx].plot(time_steps, preds[-plot_length:], 'r-', label=f'Prediction ({h}h ahead)', linewidth=1.5, alpha=0.7)
            axes[idx].set_xlabel('Time Step', fontsize=10)
            axes[idx].set_ylabel('Wind Speed', fontsize=10)
            axes[idx].set_title(f'{h}-Hour Ahead Prediction', fontsize=11, fontweight='bold')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(f'Epoch {epoch} - 1-Hour Ahead Prediction for {first_city_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pred_plot_path = os.path.join("results", f"prediction_epoch_{epoch:03d}.png")
        plt.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved multi-horizon prediction plot to {pred_plot_path}")
        
        scheduler.step(val_loss)
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join("checkpoints", "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model with validation loss (MAE): {best_val_loss:.6f}")

            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
                
# Save loss history as CSV
loss_df = pd.DataFrame({
    'epoch': list(range(1, len(train_loss_history) + 1)),
    'train_loss': train_loss_history
})

# Add validation metrics (only for epochs where validation was performed)
val_epochs = [i for i in range(1, len(val_loss_history) + 1)]
val_df = pd.DataFrame({
    'epoch': val_epochs,
    'val_loss': val_loss_history,
    'val_mae': val_mae_history
})

# Add per-horizon MAE columns
for h_name, h_mae_list in val_horizon_mae_history.items():
    val_df[h_name] = h_mae_list

# Merge the dataframes
loss_df = loss_df.merge(val_df, on='epoch', how='left')
loss_df.to_csv('results/loss_history.csv', index=False)
print(f"✓ Saved loss history to results/loss_history.csv")

# Plot and save loss curves
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-o', label='Train Loss', linewidth=2)
if val_loss_history:
    plt.plot(val_epochs, val_loss_history, 'r-s', label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MAE)', fontsize=12)
plt.title('Training and Validation Loss History', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/loss_history.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved loss plot to results/loss_history.png")
plt.close()

print(f"\n=== Training Complete ===")
print(f"Best validation loss (MAE): {best_val_loss:.6f}")
print(f"Best model saved to: checkpoints/best_model.pth")