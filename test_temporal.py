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

from temporalEncoders.conv1dEncoder import Conv1DTemporalEncoder
from temporalEncoders.gruConv1dEncoder import GRUConv1DEncoder
from temporalEncoders.sharedGru import SharedGRUEncoder
from temporalEncoders.attentionTemporalEncoder import AttentionTemporalEncoder

df, cities = load_data()
df = df[50:]

# Girdi olarak humidity, temperature, pressure ve wind_speed kullanıyoruz
input_features = ["pressure", "humidity", "temperature", "wind_speed"]
target_feature = "wind_speed"

sequences = {}  # city: {humidity: [], temperature: [], ...}
for city in df.columns.levels[0]:
    imputer = SimpleImputer(strategy="mean")
    imputed_data = imputer.fit_transform(df[city].values)
    sequences[city] = {
        "humidity": imputed_data[:, 0],
        "temperature": imputed_data[:, 1],
        "pressure": imputed_data[:, 2],
        "wind_speed": imputed_data[:, 4],
        "wind_direction": imputed_data[:, 3],
    }

# Önce tüm özellikleri normalize et
scalers = {}
features_to_scale = list(dict.fromkeys(input_features + [target_feature]))
for feature in features_to_scale:
    all_feature_data = sequences["Vancouver"][feature]
    
    scaler = StandardScaler()
    scaler.fit(np.array(all_feature_data).reshape(-1, 1))
    scalers[feature] = scaler
    
    for city in sequences.keys():
        sequences[city][feature] = scaler.transform(
            sequences[city][feature].reshape(-1, 1)
        ).flatten()


def prepare_windowed_data(city, input_features, target_feature, window_size=24, step=1, horizons=6, predDelta=True):
    """
    Her horizon için t₁'den (window sonundaki son gözlem) hedef zamana kadar cumulative delta hesapla
    1-hour: t₁ -> t₂ (delta = t₂ - t₁)
    2-hour: t₁ -> t₃ (delta = t₃ - t₁)
    3-hour: t₁ -> t₄ (delta = t₄ - t₁)
    ...
    """
    data = sequences[city]
    X, y, last_values = [], [], []
    dayofyear_list, hourofday_list = [], []  # Zamansal bilgileri sakla
    total_length = len(data["humidity"])

    for i in range(0, total_length - window_size - horizons, step):
        window_features = []
        for feature in input_features:
            window_features.append(data[feature][i:i+window_size])
        
        # Zamansal bilgi: window'un son noktasındaki (t₁) zaman damgası
        timestamp_idx = i + window_size - 1
        timestamp = df.index[timestamp_idx]
        day_of_year = timestamp.dayofyear
        hour_of_day = timestamp.hour
        
        # t₁ = son gözlem noktası (window'un sonu)
        t1_value = data[target_feature][i + window_size - 1]
        
        if np.isnan(t1_value) or np.isinf(t1_value):
            continue
        
        # Her horizon için hedef değerleri hesapla
        target_values = []
        valid_sample = True
        
        for shift in range(1, horizons+1):  # 1-horizons saat sonrası
            future_idx = i + window_size - 1 + shift  # t₁ + shift
            
            if future_idx >= total_length:
                valid_sample = False
                break
            
            future_value = data[target_feature][future_idx]
            
            if np.isnan(future_value) or np.isinf(future_value):
                valid_sample = False
                break
            
            target_values.append(future_value)
        
        if not valid_sample or len(target_values) != horizons:
            continue

        X.append(np.stack(window_features, axis=1))  # (window_size, n_features)
        dayofyear_list.append(day_of_year)
        hourofday_list.append(hour_of_day)
        
        if predDelta:
            # Delta: future_value - t₁
            target_deltas = [tv - t1_value for tv in target_values]
            y.append(np.array(target_deltas))  # (horizons,)
        else:
            # Doğrudan future_value
            y.append(np.array(target_values))  # (horizons,)

        last_values.append(t1_value)

    return np.array(X), np.array(y), np.array(last_values), np.array(dayofyear_list), np.array(hourofday_list)


class WeatherDataset(Dataset):
    def __init__(self, city, input_features, target_feature, window_size=24, step=1, horizons=6, predDelta=True):
        self.X, self.y, self.last_values, self.dayofyear, self.hourofday = prepare_windowed_data(
            city, input_features, target_feature, window_size, step, horizons, predDelta
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.last_values[idx], dtype=torch.float32),
            torch.tensor(self.dayofyear[idx], dtype=torch.float32),
            torch.tensor(self.hourofday[idx], dtype=torch.float32)
        )


# Hyperparameters
window_size = 24
step = 1
batch_size = 32
learning_rate = 0.001  
num_epochs = 50
horizons = 1
predDelta = False

dataset = WeatherDataset("Vancouver", input_features, target_feature, window_size=window_size, step=step, horizons=horizons, predDelta=predDelta)
 
# Data validation
print(f"Total samples: {len(dataset)}")
sample_X, sample_y, sample_last, sample_day, sample_hour = dataset[0]
print(f"Sample X shape: {sample_X.shape}, Sample y shape: {sample_y.shape}")
print(f"Sample dayofyear: {sample_day.item()}, hourofday: {sample_hour.item()}")
print(f"X has NaN: {torch.isnan(sample_X).any()}, y has NaN: {torch.isnan(sample_y).any()}")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_indices = list(range(0, train_size))
test_indices = list(range(train_size, len(dataset)))        

train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  


MODEL_TYPE = "conv1d" # "conv1d", "gru_conv1d", "shared_gru", "attention"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if MODEL_TYPE == "conv1d":
    model = Conv1DTemporalEncoder(n_features=len(input_features), embedding_dim=64, dropout=0.3, horizons=horizons).to(device)
    print("Using Conv1D Temporal Encoder")
elif MODEL_TYPE == "gru_conv1d":
    model = GRUConv1DEncoder(n_features=len(input_features), lookback=window_size, gru_hidden_dim=32, embedding_dim=64, horizons=horizons).to(device)
    print("Using GRU + Conv1D Encoder")
elif MODEL_TYPE == "shared_gru":
    model = SharedGRUEncoder(n_features=len(input_features), lookback=window_size, gru_hidden_dim=32, embedding_dim=64, horizons=horizons).to(device)
    print("Using Shared GRU Encoder")
elif MODEL_TYPE == "attention":
    model = AttentionTemporalEncoder(n_features=len(input_features), lookback=window_size, embedding_dim=32, dropout=0.3, horizons=horizons).to(device)
    print("Using Attention-Based Temporal Encoder")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training Loop
print(f"\nTraining on {device}")
print(f"Train samples: {train_size}, Test samples: {test_size}\n")

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for batch_X, batch_y, batch_last, batch_day, batch_hour in train_loader:
        batch_X, batch_y, batch_last = batch_X.to(device), batch_y.to(device), batch_last.to(device)
        batch_day, batch_hour = batch_day.to(device), batch_hour.to(device)
        
        # Forward pass - zamansal bilgiyi modele gönder
        predicted = model(batch_X, dayofyear=batch_day, hourofday=batch_hour)
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
    with torch.no_grad():
        for batch_X, batch_y, batch_last, batch_day, batch_hour in test_loader:
            batch_X, batch_y, batch_last = batch_X.to(device), batch_y.to(device), batch_last.to(device)
            batch_day, batch_hour = batch_day.to(device), batch_hour.to(device)
            predicted_deltas = model(batch_X, dayofyear=batch_day, hourofday=batch_hour)
            loss = criterion(predicted_deltas, batch_y)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')
print("\nLoss curve saved as 'training_loss.png'")

# Evaluate on test set
model.eval()
all_predictions = []
all_targets = []
all_last_winds = [] if predDelta else None

with torch.no_grad():
    for batch_X, batch_y, batch_last, batch_day, batch_hour in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        batch_last = batch_last.to(device)
        batch_day, batch_hour = batch_day.to(device), batch_hour.to(device)
        
        # Model prediction - zamansal bilgiyle
        predicted = model(batch_X, dayofyear=batch_day, hourofday=batch_hour)
        
        # t₁ (son gözlem) değerini al
        last_wind = batch_last  # [batch]
        
        # CPU'ya taşı
        predicted_np = predicted.cpu().numpy()
        target_np = batch_y.cpu().numpy()
        last_wind_np = last_wind.cpu().numpy()
        
        all_predictions.append(predicted_np)
        all_targets.append(target_np)
        if predDelta:
            all_last_winds.append(last_wind_np)

# Numpy array'e çevir
all_predictions = np.vstack(all_predictions)  # [n_samples, horizons]
all_targets = np.vstack(all_targets)  # [n_samples, horizons]
if predDelta:
    all_last_winds = np.concatenate(all_last_winds)  # [n_samples]

actual_horizons = all_predictions.shape[1]

# Her horizon için gerçek değerleri hesapla
if predDelta:
    predictions_absolute = all_predictions + all_last_winds
    targets_absolute = all_targets + all_last_winds
else:
    predictions_absolute = all_predictions
    targets_absolute = all_targets

# Clamp predictions
target_min = targets_absolute.min()
target_max = targets_absolute.max()
predictions_absolute = np.clip(predictions_absolute, target_min, target_max)

# Inverse transform to original scale
scaler_wind = scalers[target_feature]
all_predictions_original = np.zeros_like(predictions_absolute)
all_targets_original = np.zeros_like(targets_absolute)

for i in range(actual_horizons):
    all_predictions_original[:, i] = scaler_wind.inverse_transform(
        predictions_absolute[:, i].reshape(-1, 1)
    ).flatten()
    all_targets_original[:, i] = scaler_wind.inverse_transform(
        targets_absolute[:, i].reshape(-1, 1)
    ).flatten()

print(f"\nMax target original: {all_targets_original.max():.2f}")
print(f"Max prediction original: {all_predictions_original.max():.2f}")
print(f"Scaler mean: {scaler_wind.mean_[0]:.2f}, std: {scaler_wind.scale_[0]:.2f}")

# Plot predictions vs actual for each horizon
plt.figure(figsize=(15, 10))
actual_horizons = all_predictions.shape[1]
horizons_list = list(range(1, actual_horizons+1))
n_cols = 3
n_rows = (actual_horizons + n_cols - 1) // n_cols  # Ceiling division
for i, h in enumerate(horizons_list):
    plt.subplot(n_rows, n_cols, i+1)
    plt.plot(all_targets_original[:200, i], label='Actual', alpha=0.7)
    plt.plot(all_predictions_original[:200, i], label='Predicted', alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('Wind Speed')
    plt.title(f'{h}-Hour Ahead Prediction')
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.savefig('predictions_multi_horizon.png')
print("Multi-horizon predictions plot saved as 'predictions_multi_horizon.png'")

# Calculate and print MAE, RMSE, R2 for each horizon
print("\nEvaluation Metrics (Original Scale):")
for i in range(actual_horizons):
    mae = mean_absolute_error(all_targets_original[:, i], all_predictions_original[:, i])
    rmse = np.sqrt(mean_squared_error(all_targets_original[:, i], all_predictions_original[:, i]))
    r2 = r2_score(all_targets_original[:, i], all_predictions_original[:, i])
    print(f"Horizon {i+1}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")