"""
Data loading and preprocessing utilities for MVWGNN
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_data(dataName="ceda_data"):
    """
    Load weather data from CSV files.

    Args:
        dataName: Dataset name ('ceda_data' for UK or 'hourly-data' for USA)

    Returns:
        df: DataFrame with multi-index columns (city, feature)
        cities: DataFrame with city attributes (coordinates)
    """
    if dataName == "hourly-data":
        startDate = '2012-10-02'
        endDate = '2017-10-28'
        path = "data/"
    elif dataName == "ceda_data":
        startDate = '2019-01-01'
        endDate = '2024-12-31'
        path = "ceda_data/"
    else:
        raise ValueError(f"Unknown dataset: {dataName}")

    # Load city attributes
    cities = pd.read_csv(f"{path}city_attributes.csv")
    cities = cities[cities.Country != 'Israel']
    city_names = cities['City'].tolist()

    # Load meteorological features
    humidity = pd.read_csv(f"{path}humidity.csv")
    humidity['datetime'] = pd.to_datetime(humidity['datetime'])
    humidity = humidity[(humidity['datetime'] >= startDate) & (humidity['datetime'] <= endDate)]

    pressure = pd.read_csv(f"{path}pressure.csv")
    pressure['datetime'] = pd.to_datetime(pressure['datetime'])
    pressure = pressure[(pressure['datetime'] >= startDate) & (pressure['datetime'] <= endDate)]

    temperature = pd.read_csv(f"{path}temperature.csv")
    temperature['datetime'] = pd.to_datetime(temperature['datetime'])
    temperature = temperature[(temperature['datetime'] >= startDate) & (temperature['datetime'] <= endDate)]

    wind_direction = pd.read_csv(f"{path}wind_direction.csv")
    wind_direction['datetime'] = pd.to_datetime(wind_direction['datetime'])
    wind_direction = wind_direction[(wind_direction['datetime'] >= startDate) & (wind_direction['datetime'] <= endDate)]

    wind_speed = pd.read_csv(f"{path}wind_speed.csv")
    wind_speed['datetime'] = pd.to_datetime(wind_speed['datetime'])
    wind_speed = wind_speed[(wind_speed['datetime'] >= startDate) & (wind_speed['datetime'] <= endDate)]

    # Prepare multi-index dataframe
    data = {}
    for city in city_names:
        data[(city, 'humidity')] = humidity[city].interpolate(method='linear', limit_direction='both')
        data[(city, 'pressure')] = pressure[city].interpolate(method='linear', limit_direction='both')
        data[(city, 'temperature')] = temperature[city].interpolate(method='linear', limit_direction='both')
        data[(city, 'wind_direction')] = wind_direction[city].interpolate(method='linear', limit_direction='both')
        data[(city, 'wind_speed')] = wind_speed[city].interpolate(method='linear', limit_direction='both')

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index = humidity['datetime']
    df.index.name = 'datetime'

    return df, cities


def prepare_sequences(df, cities, input_features, target_feature,
                     train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Prepare sequences with proper temporal splitting and normalization.

    Normalization is done ONLY on training data to prevent data leakage.

    Args:
        df: Input dataframe
        cities: City attributes dataframe
        input_features: List of input feature names
        target_feature: Target feature name
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        sequences: Dict {city: {feature: array}}
        scalers: Dict {feature: StandardScaler} fitted ONLY on train data
        city_names: List of city names
        city_latitudes: List of latitudes
        city_longitudes: List of longitudes
        datetimes: DatetimeIndex
        split_indices: Dict with train/val/test boundaries
    """
    # Get city information
    city_names = cities['City'].tolist()
    city_latitudes = cities['Latitude'].tolist()
    city_longitudes = cities['Longitude'].tolist()

    print(f"Number of cities: {len(city_names)}")

    # Determine temporal split boundaries
    total_length = len(df)
    train_end_idx = int(total_length * train_ratio)
    val_end_idx = int(total_length * (train_ratio + val_ratio))

    split_indices = {
        'train_end': train_end_idx,
        'val_end': val_end_idx,
        'total': total_length,
        'has_val': val_ratio > 0
    }

    print(f"\n{'='*60}")
    print(f"TEMPORAL SPLIT BOUNDARIES")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_length}")
    print(f"Train: 0 to {train_end_idx} ({train_end_idx} timesteps, {train_end_idx/total_length*100:.1f}%)")
    print(f"Val:   {train_end_idx} to {val_end_idx} ({val_end_idx - train_end_idx} timesteps, {(val_end_idx - train_end_idx)/total_length*100:.1f}%)")
    print(f"Test:  {val_end_idx} to {total_length} ({total_length - val_end_idx} timesteps, {(total_length - val_end_idx)/total_length*100:.1f}%)")
    print(f"{'='*60}\n")

    # Prepare raw sequences
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

    # Normalize features using ONLY training data
    scalers = {}
    features_to_scale = [f for f in dict.fromkeys(input_features + [target_feature]) if f != 'wind_direction']

    print(f"{'='*60}")
    print(f"NORMALIZATION (using only training data)")
    print(f"{'='*60}")

    for feature in features_to_scale:
        # Collect ONLY training data
        train_feature_data = []
        for city in city_names:
            train_feature_data.extend(sequences_raw[city][feature][:train_end_idx])

        # Fit scaler on training data
        scaler = StandardScaler()
        scaler.fit(np.array(train_feature_data).reshape(-1, 1))
        scalers[feature] = scaler

        print(f"{feature:15s} - mean: {scaler.mean_[0]:8.4f}, std: {scaler.scale_[0]:8.4f}")

    print(f"{'='*60}\n")

    # Apply scaling to all data
    sequences = {}
    for city in city_names:
        sequences[city] = {}
        for feature in sequences_raw[city].keys():
            if feature == 'wind_direction':
                # Don't scale wind direction (circular variable)
                sequences[city][feature] = sequences_raw[city][feature]
            else:
                # Apply scaler fitted on train data
                sequences[city][feature] = scalers[feature].transform(
                    sequences_raw[city][feature].reshape(-1, 1)
                ).flatten()

    datetimes = df.index

    return sequences, scalers, city_names, city_latitudes, city_longitudes, datetimes, split_indices


def create_data_splits(sequences, city_names, input_features, target_feature,
                      window_size=24, forecast_horizon=1, step=1, batch_size=32, datetimes=None, split_indices=None):
    """
    Create train/validation/test data loaders with temporal splitting.

    Args:
        sequences: Sequences dict from prepare_sequences
        city_names: List of city names
        input_features: List of input feature names
        target_feature: Target feature name
        window_size: Lookback window size
        forecast_horizon: Number of steps ahead to predict (default: 1)
        step: Step size between windows
        batch_size: Batch size for data loaders
        datetimes: DatetimeIndex
        split_indices: Split boundaries dict

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
    """
    train_end = split_indices['train_end']
    val_end = split_indices['val_end']
    total_length = split_indices['total']
    has_val = split_indices['has_val']

    # Calculate valid starting positions (account for forecast_horizon)
    # We need window_size + forecast_horizon timesteps total
    train_positions = list(range(0, train_end - window_size - forecast_horizon, step))
    test_positions = list(range(val_end, total_length - window_size - forecast_horizon, step))

    if has_val:
        val_positions = list(range(train_end, val_end - window_size - forecast_horizon, step))
    else:
        val_positions = test_positions

    print(f"{'='*60}")
    print(f"DATA SPLIT STATISTICS")
    print(f"{'='*60}")
    print(f"Window size: {window_size}, Step size: {step}")
    print(f"Train: {len(train_positions)} samples")
    print(f"Val:   {len(val_positions)} samples")
    print(f"Test:  {len(test_positions)} samples")
    print(f"{'='*60}\n")

    # Create datasets
    train_dataset = WeatherDataset(sequences, city_names, input_features, target_feature,
                                  train_positions, window_size, datetimes, forecast_horizon)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = WeatherDataset(sequences, city_names, input_features, target_feature,
                                 test_positions, window_size, datetimes, forecast_horizon)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if has_val:
        val_dataset = WeatherDataset(sequences, city_names, input_features, target_feature,
                                    val_positions, window_size, datetimes, forecast_horizon)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = test_loader

    return train_loader, val_loader, test_loader


class WeatherDataset(Dataset):
    """
    PyTorch Dataset for weather forecasting.
    """
    def __init__(self, sequences, city_names, input_features, target_feature,
                 indices, window_size, datetimes=None, forecast_horizon=1):
        self.sequences = sequences
        self.city_names = city_names
        self.input_features = input_features
        self.target_feature = target_feature
        self.indices = indices
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.n_cities = len(city_names)
        self.n_features = len(input_features)
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

        # Target: forecast_horizon future values for ALL cities
        # Shape: [n_cities, forecast_horizon]
        y_list = []
        for city in self.city_names:
            city_targets = []
            for h in range(self.forecast_horizon):
                target_idx = i + self.window_size + h
                city_targets.append(self.sequences[city][self.target_feature][target_idx])
            y_list.append(city_targets)
        y = np.array(y_list)  # [n_cities, forecast_horizon]

        # If forecast_horizon is 1, squeeze to keep backward compatibility
        if self.forecast_horizon == 1:
            y = y.squeeze(-1)  # [n_cities]

        # Feature sequences for graph construction
        feature_sequences = {}
        for feature in ['pressure', 'temperature', 'humidity', 'wind_speed', 'wind_direction']:
            feature_sequences[feature] = np.array([
                self.sequences[city][feature][i:i+self.window_size]
                for city in self.city_names
            ])

        # Temporal information
        if self.datetimes is not None:
            window_dt = self.datetimes[i:i+self.window_size]
            hours = np.array([int(t.hour) for t in window_dt])
            try:
                days = np.array([int(t.dayofyear) for t in window_dt])
            except Exception:
                days = np.array([int(t.timetuple().tm_yday) for t in window_dt])
        else:
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
