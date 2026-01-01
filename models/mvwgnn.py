import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.graph_layers import (
    GeographicMetapath,
    FeatureSimilarityMetapath,
    MultiTemporalWindMetapath,
    MetapathFusion
)


class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal information.
    Encodes hour-of-day and day-of-year as cyclical features.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable projection for temporal features (sin/cos for hour and day)
        self.temporal_proj = nn.Linear(4, hidden_dim)

        # Learnable weight for temporal encoding contribution
        self.temporal_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, h, timestamps):
        """
        Args:
            h: Node embeddings [batch, N, lookback, hidden_dim]
            timestamps: Dict with 'hour' and 'day' tensors [batch, lookback]

        Returns:
            h_with_time: Temporally-augmented embeddings [batch, N, lookback, hidden_dim]
        """
        batch_size, N, lookback, hidden_dim = h.shape

        # Extract and normalize temporal features
        hour_of_day = timestamps['hour'] / 24.0  # [0, 1]
        day_of_year = timestamps['day'] / 365.0   # [0, 1]

        # Sinusoidal encoding (captures cyclical nature)
        hour_sin = torch.sin(2 * np.pi * hour_of_day)
        hour_cos = torch.cos(2 * np.pi * hour_of_day)
        day_sin = torch.sin(2 * np.pi * day_of_year)
        day_cos = torch.cos(2 * np.pi * day_of_year)

        # Stack: [batch, lookback, 4]
        temporal_features = torch.stack([hour_sin, hour_cos, day_sin, day_cos], dim=-1)

        # Project to hidden_dim: [batch, lookback, hidden_dim]
        temporal_emb = self.temporal_proj(temporal_features)

        # Expand for stations: [batch, N, lookback, hidden_dim]
        temporal_emb = temporal_emb.unsqueeze(1).expand(batch_size, N, lookback, hidden_dim)

        # Add to node embeddings with learnable weight
        h_with_time = h + self.temporal_weight * temporal_emb

        return h_with_time


class MVWGNN(nn.Module):
    """
    Multi-View Wind Graph Neural Network for wind speed forecasting.

    Architecture:
    1. Feature encoder: Projects each meteorological variable to hidden space
    2. Temporal encoder: 2-layer GRU with positional encoding
    3. Multi-view graphs: Geographic, Feature Similarity (4 features), Wind Propagation
    4. View fusion: Semantic attention mechanism
    5. Prediction: 2-layer MLP
    """

    def __init__(self,
                 n_stations,
                 n_features=4,
                 hidden_dim=128,
                 dropout=0.3,
                 max_wind_lag=12,
                 k_geo_neighbors=5,
                 k_feature_neighbors=5,
                 residual_weight=0.2,
                 layer_type='sage',
                 num_gnn_layers=1,
                 use_feature_view=True,
                 use_geo_view=True,
                 use_prop_view=True,
                 forecast_horizon=1):
        """
        Args:
            n_stations: Number of weather stations
            n_features: Number of input features (default: 4 - pressure, humidity, temperature, wind_speed)
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            max_wind_lag: Maximum temporal lag for wind propagation view (hours)
            k_geo_neighbors: Number of geographic neighbors
            k_feature_neighbors: Number of feature-similar neighbors
            residual_weight: Weight for residual connection (λ_r in paper)
            layer_type: Graph layer type ('sage', 'gcn', or 'gat')
            num_gnn_layers: Number of stacked GNN layers in each metapath
            use_feature_view: Enable feature similarity views
            use_geo_view: Enable geographic proximity view
            use_prop_view: Enable wind propagation view
            forecast_horizon: Number of future timesteps to predict (default: 1)
        """
        super().__init__()

        self.n_stations = n_stations
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.residual_weight = residual_weight
        self.layer_type = layer_type
        self.num_gnn_layers = num_gnn_layers
        self.forecast_horizon = forecast_horizon

        # View configuration
        self.use_feature_view = use_feature_view
        self.use_geo_view = use_geo_view
        self.use_prop_view = use_prop_view

        # Validate at least one view is enabled
        if not (use_feature_view or use_geo_view or use_prop_view):
            raise ValueError("At least one view must be enabled")

        # 1. Feature encoder
        # Projects each scalar feature value to hidden_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU()
        )

        # 2. Temporal positional encoding
        self.temporal_encoding = TemporalPositionalEncoding(hidden_dim)

        # 3. Temporal encoder (2-layer GRU as per paper)
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # 4. Multi-view graph modules

        # Geographic proximity view
        if self.use_geo_view:
            self.geo_metapath = GeographicMetapath(
                hidden_dim=hidden_dim,
                k_neighbors=k_geo_neighbors,
                distance_scale=200.0,
                layer_type=layer_type,
                num_layers=num_gnn_layers
            )

        # Feature similarity views (one per meteorological variable)
        if self.use_feature_view:
            self.pressure_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='pressure',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers
            )

            self.temp_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='temperature',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers
            )

            self.humidity_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='humidity',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers
            )

            self.wind_speed_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='wind_speed',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers
            )

        # Wind propagation view (physics-informed with directional flow)
        if self.use_prop_view and max_wind_lag > 0:
            self.wind_metapath = MultiTemporalWindMetapath(
                hidden_dim=hidden_dim,
                n_stations=n_stations,
                max_lag=max_wind_lag,
                distance_scale=500.0,
                layer_type=layer_type
            )
            self.use_wind_metapath = True
        else:
            self.use_wind_metapath = False

        # 5. Semantic attention-based view fusion
        n_metapaths = 0
        if self.use_geo_view:
            n_metapaths += 1
        if self.use_feature_view:
            n_metapaths += 4  # pressure, temp, humidity, wind_speed
        if self.use_wind_metapath:
            n_metapaths += 1

        self.fusion = MetapathFusion(
            hidden_dim=hidden_dim,
            n_metapaths=n_metapaths
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 6. Prediction head (2-layer MLP as per paper)
        # Output dimension is forecast_horizon (number of future timesteps)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_horizon)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lat, lon, feature_sequences, timestamps):
        """
        Forward pass following MVWGNN architecture.

        Args:
            x: Input features [batch, N_stations, lookback, n_features]
            lat: Station latitudes [N_stations]
            lon: Station longitudes [N_stations]
            feature_sequences: Dict with raw feature values for similarity computation
                {
                    'pressure': [batch, N, lookback],
                    'temperature': [batch, N, lookback],
                    'humidity': [batch, N, lookback],
                    'wind_speed': [batch, N, lookback],
                    'wind_direction': [batch, N, lookback]
                }
            timestamps: Dict with temporal information
                {
                    'hour': [batch, lookback],
                    'day': [batch, lookback]
                }

        Returns:
            predictions: Wind speed predictions [batch, N_stations] if forecast_horizon=1,
                        or [batch, N_stations, forecast_horizon] if forecast_horizon>1
            attention_info: Dict with attention weights for analysis
        """
        batch_size, N, lookback, n_features = x.shape

        # 1. Feature Encoding
        # Encode each feature value to hidden_dim
        x_flat = x.reshape(batch_size * N * lookback * n_features, 1)
        h_encoded = self.feature_encoder(x_flat)  # [batch*N*lookback*n_features, hidden_dim]
        h_encoded = h_encoded.reshape(batch_size, N, lookback, n_features, self.hidden_dim)

        # 2. Temporal Encoding
        # Process each feature separately through temporal encoding + GRU
        feature_names = ['pressure', 'humidity', 'temperature', 'wind_speed']
        h_feature_specific = {}

        for feat_idx, feat_name in enumerate(feature_names):
            # Get feature-specific embeddings: [batch, N, lookback, hidden_dim]
            h_feat = h_encoded[:, :, :, feat_idx, :]

            # Add temporal positional encoding
            h_feat = self.temporal_encoding(h_feat, timestamps)

            # Apply GRU: [batch*N, lookback, hidden_dim]
            h_feat_flat = h_feat.reshape(batch_size * N, lookback, self.hidden_dim)
            gru_out, _ = self.temporal_encoder(h_feat_flat)

            # Take final hidden state: [batch*N, hidden_dim] -> [batch, N, hidden_dim]
            h_feat_final = gru_out[:, -1, :].reshape(batch_size, N, self.hidden_dim)
            h_feature_specific[feat_name] = h_feat_final

        # General temporal representation (mean of all features for geo/wind views)
        h_mean = h_encoded.mean(dim=3)  # [batch, N, lookback, hidden_dim]
        h_mean = self.temporal_encoding(h_mean, timestamps)
        h_mean_flat = h_mean.reshape(batch_size * N, lookback, self.hidden_dim)
        gru_out_mean, _ = self.temporal_encoder(h_mean_flat)
        h_general = gru_out_mean[:, -1, :].reshape(batch_size, N, self.hidden_dim)

        # 3. Multi-View Graph Processing
        metapath_embeddings = {}

        # Geographic proximity view
        if self.use_geo_view:
            metapath_embeddings['geo'] = self.geo_metapath(h_general, lat, lon)

        # Feature similarity views
        if self.use_feature_view:
            metapath_embeddings['pressure'] = self.pressure_metapath(
                h_feature_specific['pressure'],
                feature_sequences['pressure']
            )
            metapath_embeddings['temp'] = self.temp_metapath(
                h_feature_specific['temperature'],
                feature_sequences['temperature']
            )
            metapath_embeddings['humidity'] = self.humidity_metapath(
                h_feature_specific['humidity'],
                feature_sequences['humidity']
            )
            metapath_embeddings['wind_speed'] = self.wind_speed_metapath(
                h_feature_specific['wind_speed'],
                feature_sequences['wind_speed']
            )

        # Wind propagation view
        lag_attention = None
        if self.use_wind_metapath:
            h_wind, lag_attention = self.wind_metapath(
                h_mean,  # Full temporal sequence
                feature_sequences['wind_direction'],
                feature_sequences['wind_speed'],
                lat,
                lon
            )
            metapath_embeddings['wind'] = h_wind

        # 4. Semantic Attention-based View Fusion
        h_fused, metapath_attention = self.fusion(metapath_embeddings)

        # Apply layer normalization and residual connection
        h_fused = self.layer_norm(h_fused)
        h_out = h_fused + self.residual_weight * h_general

        # 5. Prediction
        predictions = self.predictor(h_out)  # [batch, N, forecast_horizon]

        # Squeeze last dimension if forecast_horizon=1 for backward compatibility
        if self.forecast_horizon == 1:
            predictions = predictions.squeeze(-1)  # [batch, N]

        # Package attention information for analysis
        attention_info = {
            'metapath_attention': metapath_attention,
            'lag_attention': lag_attention
        }

        # Add sparsity loss for attention (encourages peaked distributions)
        if metapath_attention is not None and isinstance(metapath_attention, dict):
            # Convert dict of attention weights to tensor for sparsity calculation
            attention_values = torch.tensor(list(metapath_attention.values()),
                                           dtype=torch.float32,
                                           device=predictions.device)
            # Entropy-based sparsity: lower entropy = more peaked
            sparsity_loss = -(attention_values * torch.log(attention_values + 1e-10)).sum()
            attention_info['sparsity_loss'] = -sparsity_loss  # Negative because we want to minimize entropy

        return predictions, attention_info
