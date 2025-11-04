from model import DDGNNWind
from dataLoad import load_data
from torchData import GraphWeatherDataset, create_dataloader
import torch
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# ============================================================================
# V5 TRAINING: ALL FIXES APPLIED
# ============================================================================
# Changes from V3.1:
# 1. ✅ Model: PatchTST + GATv2 (modern architecture)
# 2. ✅ hidden_dim: 128 → 64 (reduced capacity)
# 3. ✅ seq_len: 48 → 24 (shorter sequences)
# 4. ✅ weight_decay: 5e-5 → 0.001 (stronger regularization)
# 5. ✅ dropout: 0.1-0.25 → 0.35 (higher)
# 6. ✅ batch_size: consistent 16 for train/val
# 7. ✅ EMA: exponential moving average for stability
# 8. ✅ nucleus_p: 0.9 → 0.85 (less aggressive sparsification)
# ============================================================================

df, cities = load_data("hourly-data")
datetime_index = df.index
print(df.sample(5))
print("n stations:", len(cities))

first_city_name = cities['City'].iloc[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_stations = len(cities)

# =========== CRITICAL HYPERPARAMETER CHANGES ===========
seq_len = 24        # 48 → 24: Shorter sequences
hidden_dim = 64     # 128 → 64: Reduce capacity
batch_size = 16     # 32 → 16: Smaller batches, consistent train/val
patch_len = 4       # NEW: For PatchTST (24/4 = 6 patches)
dropout = 0.35      # 0.1-0.25 → 0.35: Stronger regularization
# =======================================================

train_size = int(0.8 * len(df))
epochNum = 100

prediction_horizons = [1]
num_horizons = len(prediction_horizons)

horizon_weights = torch.tensor([1.0], device=device)  
horizon_weights = horizon_weights / horizon_weights.sum()

train_dataset = GraphWeatherDataset(
    df=df.iloc[:train_size],
    cities=cities,
    seq_len=seq_len,
    prediction_window=max(prediction_horizons),
    sliding_window=True
)

test_dataset = GraphWeatherDataset(
    df=df.iloc[train_size:],
    cities=cities,
    seq_len=seq_len,
    prediction_window=max(prediction_horizons),
    sliding_window=True
)

# =========== CONSISTENT BATCH SIZE ===========
train_loader = create_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = create_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
# ============================================

# =========== V5 MODEL: PatchTST + GATv2 ===========
model = DDGNNWind(
    n_stations=n_stations,
    hidden_dim=hidden_dim,
    n_heads=4,
    seq_len=seq_len,
    n_gnn_layers=2,
    temporal_debug=False,
    use_task_aware_adj=True,
    use_cross_attention=True,
    dropout=dropout,
    patch_len=patch_len  # NEW parameter
).to(device)
# =================================================

# ============ EMA for Model Stability ============
class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

ema = EMA(model, decay=0.999)
print("[INFO] EMA initialized")
# ================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

def criterion_multi_horizon(output, target, horizon_weights, model=None, l2_lambda=0.001):
    """Compute weighted MAE loss with STRONGER L2 regularization."""
    mae_per_sample = torch.abs(output - target)
    
    if mae_per_sample.dim() == 3:
        mae_per_horizon = mae_per_sample.mean(dim=(0, 1))
    else:
        mae_per_horizon = mae_per_sample.mean(dim=0)
    
    weighted_loss = (mae_per_horizon * horizon_weights).sum()
    
    # Stronger L2 regularization
    if model is not None and l2_lambda > 0:
        l2_reg = torch.tensor(0., device=output.device)
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2) ** 2
        weighted_loss = weighted_loss + l2_lambda * l2_reg
    
    return weighted_loss, mae_per_horizon

criterion = torch.nn.L1Loss(reduction='none')

# ============ STRONGER REGULARIZATION ============
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.001,              # Conservative starting LR
    weight_decay=0.001,    # 5e-5 → 0.001 (20x stronger!)
    betas=(0.9, 0.999)
)
# =================================================

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    epochs=epochNum,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1000
)

stop_counter = 0
patience = 15

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

train_loss_history = []
val_loss_history = []
val_mae_history = []
val_horizon_mae_history = {f'horizon_{h}h_mae': [] for h in prediction_horizons}
best_val_loss = float('inf')
best_val_mae = float('inf')

def train_epoch(model, epoch):
    model.train()
    total_loss = 0.0
    horizon_losses_sum = {f'horizon_{h}h': 0.0 for h in prediction_horizons}
    
    scaler = train_loader.dataset.wind_speed_scaler
    scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)
    scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)

    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"):
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

        optimizer.zero_grad()
        
        output = model(
            historical_data,
            lat,
            lon,
            current_hour,
            day_of_year,
            wind_dir_deg_last,
            positions,
        )

        output_original = output * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)
        target_original = target * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)
        
        # Stronger L2 regularization
        weighted_loss, mae_per_horizon = criterion_multi_horizon(
            output_original, target_original, horizon_weights, 
            model=model, l2_lambda=0.001
        )
        
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update EMA
        ema.update()

        loss_val = weighted_loss.item()
        total_loss += loss_val * b
        
        for i, h in enumerate(prediction_horizons):
            horizon_losses_sum[f'horizon_{h}h'] += mae_per_horizon[i].item() * b

    avg_loss = total_loss / len(train_loader.dataset)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, Training MAE: {avg_loss:.6f}, LR: {current_lr:.6f}")
    
    for h_name, h_loss_sum in horizon_losses_sum.items():
        avg_h_loss = h_loss_sum / len(train_loader.dataset)
        print(f"  {h_name} MAE: {avg_h_loss:.6f}")
    
    return avg_loss

def eval_model(model, data_loader, use_ema=True):
    """Evaluate with EMA weights"""
    
    if use_ema:
        ema.apply_shadow()
    
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0
    
    horizon_maes = {f'horizon_{h}h': 0.0 for h in prediction_horizons}
    
    scaler = data_loader.dataset.wind_speed_scaler
    scaler_mean = torch.from_numpy(scaler.mean_).float().to(device)
    scaler_scale = torch.from_numpy(scaler.scale_).float().to(device)
    
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
            
            loss = criterion(output, target)
            
            output_original = output * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)
            target_original = target * scaler_scale.unsqueeze(0).unsqueeze(-1) + scaler_mean.unsqueeze(0).unsqueeze(-1)
            mae = torch.mean(torch.abs(output_original - target_original))

        loss_val = loss.mean().item()
        mae_val = mae.item()
        
        total_loss += loss_val * b
        total_mae += mae_val * b
        total_samples += b
        
        for i, h in enumerate(prediction_horizons):
            mae_h = torch.mean(torch.abs(output_original[:, :, i] - target_original[:, :, i]))
            horizon_maes[f'horizon_{h}h'] += mae_h.item() * b
            
            first_city_predictions[h].append(output_original[:, 1, i].cpu().numpy())
            first_city_targets[h].append(target_original[:, 1, i].cpu().numpy())

    avg_loss = total_loss / total_samples
    avg_mae = total_mae / total_samples
    
    if use_ema:
        ema.restore()
    
    import numpy as np
    for h in prediction_horizons:
        first_city_predictions[h] = np.concatenate(first_city_predictions[h])
        first_city_targets[h] = np.concatenate(first_city_targets[h])
    
    print(f"Validation - MAE (loss): {avg_loss:.6f}, MAE (metric): {avg_mae:.6f}")
    
    avg_horizon_maes = {}
    for h_name, h_mae in horizon_maes.items():
        avg_h_mae = h_mae / total_samples
        avg_horizon_maes[h_name] = avg_h_mae
        print(f"  {h_name} MAE: {avg_h_mae:.6f}")
    
    return avg_loss, avg_mae, first_city_predictions, first_city_targets, avg_horizon_maes


# Training loop
print("\n" + "="*60)
print("STARTING V5 TRAINING")
print("="*60)
print(f"Key changes:")
print(f"  - Architecture: PatchTST + GATv2")
print(f"  - hidden_dim: 128 → {hidden_dim}")
print(f"  - seq_len: 48 → {seq_len}")
print(f"  - weight_decay: 5e-5 → 0.001")
print(f"  - dropout: 0.1-0.25 → {dropout}")
print(f"  - batch_size: consistent {batch_size}")
print(f"  - EMA: enabled")
print("="*60 + "\n")

for epoch in range(1, epochNum):
    train_loss = train_epoch(model, epoch)
    train_loss_history.append(train_loss)

    if epoch > 0 and epoch % 1 == 0:
        val_loss, val_mae, first_city_preds, first_city_targs, avg_horizon_maes = eval_model(model, val_loader, use_ema=True)
        val_loss_history.append(val_loss)
        val_mae_history.append(val_mae)
        
        for h_name, h_mae_val in avg_horizon_maes.items():
            val_horizon_mae_history[f'{h_name}_mae'].append(h_mae_val)
        
        # Plot predictions every 5 epochs
        if epoch % 5 == 0 or epoch <= 5:
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
                axes[idx].set_title(f'{h}-Hour Ahead Prediction - Epoch {epoch}', fontsize=11, fontweight='bold')
                axes[idx].legend(fontsize=9)
                axes[idx].grid(True, alpha=0.3)
            
            plt.suptitle(f'V5 Model (PatchTST+GATv2) - Epoch {epoch} - {first_city_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            pred_plot_path = os.path.join("results", f"prediction_v5_epoch_{epoch:03d}.png")
            plt.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved prediction plot to {pred_plot_path}")
        
        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_loss = val_loss
            checkpoint_path = os.path.join("checkpoints", "best_model_v5.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_shadow': ema.shadow,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'best_val_mae': best_val_mae,
                'best_val_loss': best_val_loss,
                'config': {
                    'hidden_dim': hidden_dim,
                    'seq_len': seq_len,
                    'patch_len': patch_len,
                    'dropout': dropout,
                    'weight_decay': 0.001
                }
            }, checkpoint_path)
            print(f"✓ Saved best V5 model with validation MAE: {best_val_mae:.6f}")
            stop_counter = 0
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

# Save loss history
loss_df = pd.DataFrame({
    'epoch': list(range(1, len(train_loss_history) + 1)),
    'train_loss': train_loss_history
})

val_epochs = [i for i in range(1, len(val_loss_history) + 1)]
val_df = pd.DataFrame({
    'epoch': val_epochs,
    'val_loss': val_loss_history,
    'val_mae': val_mae_history
})

for h_name, h_mae_list in val_horizon_mae_history.items():
    val_df[h_name] = h_mae_list

loss_df = loss_df.merge(val_df, on='epoch', how='left')
loss_df.to_csv('results/loss_history_v5.csv', index=False)
print(f"✓ Saved loss history to results/loss_history_v5.csv")

# Plot MAE curves
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, 'b-o', label='Train MAE', linewidth=2, markersize=4, alpha=0.8)
if val_mae_history:
    plt.plot(val_epochs, val_mae_history, 'r-s', label='Validation MAE', linewidth=2, markersize=4, alpha=0.8)
    
    best_epoch = val_epochs[val_mae_history.index(min(val_mae_history))]
    plt.axhline(y=best_val_mae, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Best Val MAE: {best_val_mae:.4f}')
    plt.scatter([best_epoch], [best_val_mae], color='green', s=100, zorder=5, marker='*', edgecolors='darkgreen', linewidths=1.5)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MAE (m/s)', fontsize=12)
plt.title('V5 Model (PatchTST+GATv2) - Training and Validation MAE History', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper right')
plt.grid(True, alpha=0.3, linestyle=':')
plt.tight_layout()
plt.savefig('results/mae_history_v5.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved MAE plot to results/mae_history_v5.png")
plt.close()

print(f"\n{'='*60}")
print(f"V5 TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best validation MAE: {best_val_mae:.6f}")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Best model saved to: checkpoints/best_model_v5.pth")
print(f"{'='*60}\n")