import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_layers import (
    GeographicMetapath,
    FeatureSimilarityMetapath,
    MultiTemporalWindMetapath,
    MetapathFusion
)
import numpy as np

class FeatureInteractionLayer(nn.Module):
    """
    Model feature interactions WITHIN each station.
    Similar to HistGNN's local graph.
    """
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        
        # Feature adjacency (learnable)
        self.feature_adj = nn.Parameter(torch.randn(n_features, n_features))
        
        # Feature transformation
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, N_stations, n_features, hidden_dim]
               Each station has embeddings for each feature
        
        Returns:
            x_out: [batch, N_stations, n_features, hidden_dim]
        """
        batch_size, N, n_features, hidden_dim = x.shape
        
        # Compute feature adjacency (softmax per feature)
        adj = F.softmax(self.feature_adj, dim=-1)  # [n_features, n_features]
        
        # Reshape: [batch * N, n_features, hidden_dim]
        x_flat = x.reshape(batch_size * N, n_features, hidden_dim)
        
        # Feature aggregation: f_i' = Σ_j adj[i,j] * f_j
        x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size * N, -1, -1), x_flat)
        
        # Transform
        x_transformed = self.feature_mlp(x_agg)
        
        # Residual
        x_out = 0.5 * x_transformed + 0.5 * x_flat
        
        # Reshape back
        return x_out.reshape(batch_size, N, n_features, hidden_dim)

class TemporalPositionalEncoding(nn.Module):
    """
    Add temporal information using sinusoidal encoding.
    Encodes: hour_of_day, day_of_year
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable projection for temporal features
        self.temporal_proj = nn.Linear(4, hidden_dim)  # 4 = 2 features × 2 (sin, cos)
        
        # Learnable weight for temporal encoding contribution
        self.temporal_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, h, timestamps):
        """
        Args:
            h: [batch, N, lookback, hidden_dim] - node embeddings
            timestamps: [batch, lookback] - datetime objects or indices
        
        Returns:
            h_with_time: [batch, N, lookback, hidden_dim]
        """
        batch_size, N, lookback, hidden_dim = h.shape
        device = h.device
        
        # Extract temporal features
        hour_of_day = timestamps['hour']  # [batch, lookback] in [0, 23]
        day_of_year = timestamps['day']   # [batch, lookback] in [0, 365]
        
        # Normalize to [0, 1]
        hour_norm = hour_of_day / 24.0
        day_norm = day_of_year / 365.0
        
        # Sinusoidal encoding (captures cyclical nature)
        hour_sin = torch.sin(2 * np.pi * hour_norm)
        hour_cos = torch.cos(2 * np.pi * hour_norm)
        day_sin = torch.sin(2 * np.pi * day_norm)
        day_cos = torch.cos(2 * np.pi * day_norm)
        
        # Stack: [batch, lookback, 4]
        temporal_features = torch.stack([hour_sin, hour_cos, day_sin, day_cos], dim=-1)
        
        # Project to hidden_dim: [batch, lookback, hidden_dim]
        temporal_emb = self.temporal_proj(temporal_features)
        
        # Expand for stations: [batch, N, lookback, hidden_dim]
        temporal_emb = temporal_emb.unsqueeze(1).expand(batch_size, N, lookback, hidden_dim)
        
        # ADD to node embeddings with learnable weight
        h_with_time = h + self.temporal_weight * temporal_emb
        
        return h_with_time

class MetapathGNN(nn.Module):
    """
    Metapath-based Heterogeneous Graph Neural Network for Wind Speed Forecasting.
    
    Architecture:
    1. Feature encoder (temporal encoder for input sequences)
    2. Multiple metapaths:
       - Geographic proximity
       - Temperature similarity
       - Humidity similarity
       - Wind speed similarity
       - Multi-temporal wind propagation (CORE NOVELTY!)
    3. Metapath fusion via semantic attention
    4. Temporal decoder (GRU)
    5. Prediction head
    """
    def __init__(self, 
                 n_stations,
                 n_features=4,
                 hidden_dim=64,
                 dropout=0.3,
                 max_wind_lag=12,
                 k_geo_neighbors=5,
                 k_feature_neighbors=8):
        super().__init__()
        
        self.n_stations = n_stations
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # 1. Feature encoder (encode each scalar feature value -> hidden_dim)
        # We encode each feature separately so we can apply feature-level interaction.
        self.feature_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Metapath modules
        # Geographic metapath
        self.geo_metapath = GeographicMetapath(
            hidden_dim=hidden_dim,
            k_neighbors=k_geo_neighbors,
            distance_scale=200.0
        )
        
        # Feature similarity metapaths
        self.temp_metapath = FeatureSimilarityMetapath(
            hidden_dim=hidden_dim,
            feature_name='temperature',
            top_k=k_feature_neighbors,
            lookback_hours=24
        )
        
        self.humidity_metapath = FeatureSimilarityMetapath(
            hidden_dim=hidden_dim,
            feature_name='humidity',
            top_k=k_feature_neighbors,
            lookback_hours=24
        )
        
        self.wind_speed_metapath = FeatureSimilarityMetapath(
            hidden_dim=hidden_dim,
            feature_name='wind_speed',
            top_k=k_feature_neighbors,
            lookback_hours=24
        )
        
        # Multi-temporal wind metapath (CORE NOVELTY!)
        self.wind_metapath = MultiTemporalWindMetapath(
            hidden_dim=hidden_dim,
            max_lag=max_wind_lag,
            distance_scale=500.0
        )

        self.feature_interaction = FeatureInteractionLayer(
            n_features=n_features,
            hidden_dim=hidden_dim
        )
        
        # 3. Metapath fusion
        n_metapaths = 5  # geo + temp + humidity + wind_speed + wind
        self.fusion = MetapathFusion(
            hidden_dim=hidden_dim,
            n_metapaths=n_metapaths
        )
        
        # Layer normalization for stability (only on fusion output)
        self.layer_norm_fusion = nn.LayerNorm(hidden_dim)
        
        # 4. Temporal decoder (GRU over fused representations)
        self.temporal_decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.temporal_encoding = TemporalPositionalEncoding(hidden_dim)
        
        # 5. Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Station-specific bias (each station has different baseline)
        # Initialize small to not disrupt initial training
        self.station_bias = nn.Parameter(torch.zeros(n_stations) * 0.01)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lat, lon, feature_sequences, timestamps):
        """
        Args:
            x: Input features [batch, N_stations, lookback, n_features]
            lat, lon: Station coordinates [N_stations]
            feature_sequences: dict {
                'temperature': [batch, N, lookback],
                'humidity': [batch, N, lookback],
                'wind_speed': [batch, N, lookback],
                'wind_direction': [batch, N, lookback],
            }
        
        Returns:
            predictions: [batch, N_stations]
            attention_info: dict with lag_attention and metapath_attention
        """
        batch_size, N, lookback, n_features = x.shape

        # 1. Encode features at each timestep
        x_feat_flat = x.reshape(batch_size * N * lookback * n_features, 1)
        h_feat = self.feature_encoder(x_feat_flat)
        h_feat = h_feat.reshape(batch_size, N, lookback, n_features, self.hidden_dim)

        # 2. Feature interaction per timestep
        h_interacted = []
        for t in range(lookback):
            h_t = h_feat[:, :, t, :, :]
            h_t_inter = self.feature_interaction(h_t)
            h_interacted.append(h_t_inter)

        h_interacted = torch.stack(h_interacted, dim=2)
        h_encoded = h_interacted.mean(dim=3)  # [batch, N, lookback, hidden_dim]
        h_encoded = self.temporal_encoding(h_encoded, timestamps)

        # 3. GRU OVER ENCODED FEATURES (BEFORE metapaths!)
        h_gru = h_encoded.reshape(batch_size * N, lookback, self.hidden_dim)
        gru_out, _ = self.temporal_decoder(h_gru)  # [batch*N, lookback, hidden_dim]
        h_temporal = gru_out[:, -1, :]  # [batch*N, hidden_dim]
        h_temporal = h_temporal.reshape(batch_size, N, self.hidden_dim)

        # 4. Apply metapaths ONCE on temporal summary
        metapath_embeddings = {}

        metapath_embeddings['geo'] = self.geo_metapath(h_temporal, lat, lon)
        metapath_embeddings['temp'] = self.temp_metapath(
            h_temporal, feature_sequences['temperature']
        )
        metapath_embeddings['humidity'] = self.humidity_metapath(
            h_temporal, feature_sequences['humidity']
        )
        metapath_embeddings['wind_speed'] = self.wind_speed_metapath(
            h_temporal, feature_sequences['wind_speed']
        )

        # Multi-temporal wind: needs full sequence
        h_wind, lag_attention = self.wind_metapath(
            h_encoded,  # Full temporal sequence
            feature_sequences['wind_direction'],
            feature_sequences['wind_speed'],
            lat,
            lon
        )
        metapath_embeddings['wind'] = h_wind

        # 5. Fuse metapaths
        h_fused, metapath_attention = self.fusion(metapath_embeddings)
        
        # Apply layer normalization and add residual temporal info
        h_fused = self.layer_norm_fusion(h_fused)
        h_fused = h_fused + 0.3 * h_temporal  # Residual connection (reduced weight)

        # 6. Predict
        predictions = self.predictor(h_fused)
        predictions = predictions.squeeze(-1)
        
        # Add station-specific bias
        predictions = predictions + self.station_bias.unsqueeze(0)

        attention_info = {
            'lag_attention': lag_attention,
            'metapath_attention': metapath_attention
        }

        return predictions, attention_info