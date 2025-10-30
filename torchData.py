import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


class GraphWeatherDataset(Dataset):
    """
    Each item corresponds to a sliding window in time and contains:
      - historical_data: (n_stations, seq_len, 6) float32
          features: [wind_speed, wind_dir_cos, wind_dir_sin, pressure, temperature, humidity]
      - lat: (n_stations,) float32 (min-max normalized)
      - lon: (n_stations,) float32 (min-max normalized)
      - current_hour: scalar int (prediction time hour)
      - day_of_year: scalar int (prediction time day of year)
      - wind_direction_deg: (n_stations,) float32 (degrees at last history step)
      - positions: (n_stations, 2) float32 (lat, lon normalized)
      - target: (n_stations,) float32 scaled wind_speed at prediction time

    Notes:
      - Expects df with MultiIndex columns (city, feature) and datetime-like index.
      - cities should contain columns 'City', 'Latitude', 'Longitude' and be in the
        same order as the df city columns (this mirrors how `load_data` constructs df).
    """

    def __init__(self, df: pd.DataFrame, cities: pd.DataFrame, seq_len: int = 24, prediction_window: int = 1, sliding_window: bool = True):
        self.df = df.copy()
        self.cities = cities.reset_index(drop=True)
        self.city_names = self.cities['City'].tolist()
        self.seq_len = seq_len
        self.prediction_window = prediction_window
        self.sliding_window = sliding_window
        # sliding step size (advance by half the sequence length)
        # self.step_size = max(1, self.seq_len // 2) if self.sliding_window else (self.seq_len + 1)
        self.step_size = 6

        # feature order we will output per time-step per station (5 features)
        # wind_speed, wind_dir_cos, wind_dir_sin, pressure, temperature
        self.raw_feature_names = ['wind_speed', 'wind_direction', 'pressure', 'temperature', 'humidity']

        # Build raw array: (T, n_stations, raw_features)
        # We assume df columns are MultiIndex (city, feature)
        T = len(self.df)
        self.n_stations = len(self.city_names)

        raw_feats = np.zeros((T, self.n_stations, len(self.raw_feature_names)), dtype=float)
        # Keep original wind_direction degrees array for wind_direction_deg return
        self.orig_wind_dir = np.zeros((T, self.n_stations), dtype=float)

        for si, city in enumerate(self.city_names):
            for fi, feat in enumerate(self.raw_feature_names):
                try:
                    raw_series = self.df[(city, feat)].values
                except Exception:
                    raise KeyError(f"Missing column for city/feature: {(city, feat)} in df")
                raw_feats[:, si, fi] = raw_series
            # store original wind dir
            self.orig_wind_dir[:, si] = raw_feats[:, si, 1]

        # Impute missing values across time & stations per feature vector
        imputer = KNNImputer(n_neighbors=5)
        reshaped = raw_feats.reshape(T * self.n_stations, len(self.raw_feature_names))
        reshaped = imputer.fit_transform(reshaped)
        raw_feats = reshaped.reshape(T, self.n_stations, len(self.raw_feature_names))

        # Compute wind dir sin/cos and build final features (5 dims)
        wind_speed = raw_feats[:, :, 0]
        wind_dir_deg = raw_feats[:, :, 1]
        wind_dir_rad = np.deg2rad(wind_dir_deg)
        wind_dir_cos = np.cos(wind_dir_rad)
        wind_dir_sin = np.sin(wind_dir_rad)
        pressure = raw_feats[:, :, 2]
        temperature = raw_feats[:, :, 3]

        final_feats = np.stack([wind_speed, wind_dir_cos, wind_dir_sin, pressure, temperature], axis=-1)
        # final_feats shape: (T, n_stations, 5)

        # --- Per-city, per-feature scaling ---
        # Compute mean/std per station and per feature (shape: n_stations x n_features)
        # and scale each (city,feature) independently.
        eps = 1e-8
        self.feat_means = final_feats.mean(axis=0).astype(np.float32)   # (n_stations, 5)
        self.feat_stds = final_feats.std(axis=0).astype(np.float32) + eps  # (n_stations, 5)

        # Apply per-city-per-feature z-score scaling
        final_scaled = (final_feats - self.feat_means[None, :, :]) / self.feat_stds[None, :, :]
        self.final_feats = final_scaled.astype(np.float32)

        # Note: target (wind_speed) will use the same per-city-per-feature scaling (col 0)
        
        # Create scaler object for inverse transform (wind_speed is feature index 0)
        self.wind_speed_scaler = StandardScaler()
        self.wind_speed_scaler.mean_ = self.feat_means[:, 0]  # (n_stations,)
        self.wind_speed_scaler.scale_ = self.feat_stds[:, 0]  # (n_stations,)
        self.wind_speed_scaler.var_ = self.feat_stds[:, 0] ** 2
        self.wind_speed_scaler.n_features_in_ = self.n_stations

        # Prepare spatial tensors (min-max normalize lat/lon)
        lats = self.cities['Latitude'].astype(float).values
        lons = self.cities['Longitude'].astype(float).values
        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        self.norm_lats = ((lats - lat_min) / (lat_max - lat_min + 1e-8)).astype(np.float32)
        self.norm_lons = ((lons - lon_min) / (lon_max - lon_min + 1e-8)).astype(np.float32)
        self.positions = np.stack([self.norm_lats, self.norm_lons], axis=1).astype(np.float32)

        # Datetimes index
        try:
            self.datetimes = pd.to_datetime(self.df.index)
        except Exception:
            self.datetimes = pd.to_datetime(self.df.index.astype(str))

        # Number of samples depends on sliding_window mode
        self.T = T
        if self.sliding_window:
            # Sliding windows advance by step_size (default: seq_len // 2)
            # Require target_time = start + seq_len to be <= T-1
            # So max valid start index is (T - seq_len - 1)
            max_start = self.T - self.seq_len - 1
            if max_start < 0:
                self.n_samples = 0
            else:
                self.n_samples = (max_start // self.step_size) + 1
        else:
            # Non-overlapping windows: divide into chunks
            # Each sample uses seq_len timesteps for history + 1 for target
            chunk_size = self.seq_len + 1
            self.n_samples = max(0, self.T // chunk_size)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.n_samples + idx
        if idx < 0 or idx >= self.n_samples:
            raise IndexError('Index out of range')

        # Calculate indices based on sliding_window mode
        if self.sliding_window:
            # Sliding window with step_size shift
            # Historical data: timesteps [start, start+1, ..., start+seq_len-1]
            # Target: timestep [start + seq_len]
            start = idx * self.step_size
            end = start + self.seq_len
            target_time = end
        else:
            # Non-overlapping windows: each sample is a separate chunk
            # Sample 0: hist=[0:24], target=24
            # Sample 1: hist=[25:49], target=49
            # Sample 2: hist=[50:74], target=74
            chunk_size = self.seq_len + 1
            start = idx * chunk_size
            end = start + self.seq_len  # exclusive
            target_time = end

        # historical data: (seq_len, n_stations, 5) -> transpose to (n_stations, seq_len, 5)
        hist = self.final_feats[start:end, :, :].transpose(1, 0, 2)

        # Multi-horizon targets: [1] hour ahead
        horizons = [1]
        targets = []
        
        for h in horizons:
            target_idx = target_time + h - 1  # -1 because target_time is already +1
            if target_idx < self.T:
                target_h = self.final_feats[target_idx, :, 0].astype(np.float32)  # (n_stations,)
            else:
                # If horizon extends beyond data, use last available
                target_h = self.final_feats[-1, :, 0].astype(np.float32)
            targets.append(target_h)

        # Stack targets: (1, n_stations) -> transpose to (n_stations, 1)
        target_scaled = np.stack(targets, axis=0).T.astype(np.float32)  # (n_stations, 1)
        
        # spatial tensors
        lat_t = torch.from_numpy(self.norm_lats).float()
        lon_t = torch.from_numpy(self.norm_lons).float()
        positions_t = torch.from_numpy(self.positions).float()

        # wind_direction_deg at last history step (un-imputed original degrees preferred)
        wind_dir_deg_last = self.orig_wind_dir[end - 1, :].astype(np.float32)

        # cyclic features
        dt = self.datetimes[target_time]
        current_hour = int(dt.hour)
        day_of_year = int(dt.timetuple().tm_yday)

        # save hist and target as tensors
        hist_t = torch.from_numpy(hist).float()
        target_t = torch.from_numpy(target_scaled).float()
        torch.save(hist_t, 'debug_hist.pt')
        torch.save(target_t, 'debug_target.pt')
        return (
            torch.from_numpy(hist).float(),  # (n_stations, seq_len, 5)
            lat_t,  # (n_stations,)
            lon_t,  # (n_stations,)
            torch.tensor(current_hour, dtype=torch.long),
            torch.tensor(day_of_year, dtype=torch.long),
            torch.from_numpy(wind_dir_deg_last).float(),  # (n_stations,)
            positions_t,  # (n_stations, 2)
            torch.from_numpy(target_scaled).float()  # (n_stations, 5) - multi-horizon targets
        )


def create_dataloader(dataset: Dataset, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0):
    """Create a DataLoader for GraphWeatherDataset.

    This uses the default collate which will stack tensors along dim=0. For this dataset,
    batching will produce tensors with an extra leading batch dimension. Users should
    adapt model forward to accept batching, or use batch_size=1 and manually handle batching.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)