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

class ConditionalFeatureInteraction(nn.Module):
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        # Context-aware adjacency predictor
        self.adj_predictor = nn.Sequential(
            nn.Linear(hidden_dim * n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features * n_features),
            nn.Sigmoid()
        )
        self.feature_transform = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        # x: [batch, N, n_features, hidden_dim]
        batch_size, N, n_features, hidden_dim = x.shape
        
        # Predict adjacency based on current features
        x_context = x.reshape(batch_size, N, -1)  # [batch, N, n_features*hidden_dim]
        adj_logits = self.adj_predictor(x_context)
        adj = adj_logits.view(batch_size, N, n_features, n_features)
        
        # Apply interaction per sample
        x_flat = x.reshape(batch_size * N, n_features, hidden_dim)
        adj_flat = adj.reshape(batch_size * N, n_features, n_features)
        
        x_agg = torch.bmm(adj_flat, x_flat)
        x_out = self.feature_transform(x_agg) + x_flat  # Residual
        
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

    Architecture - SEPARATE FEATURE METAPATHS:
    1. Feature encoder (encode each feature separately)
    2. Shared GRU (single temporal encoder for all features)
    3. Metapaths:
       - Geographic proximity
       - Pressure similarity
       - Temperature similarity
       - Humidity similarity
       - Wind speed similarity
       - Multi-temporal wind propagation (directional)
    4. Metapath fusion via semantic attention
    5. Prediction head
    """
    def __init__(self,
                 n_stations,
                 n_features=4,
                 hidden_dim=64,
                 dropout=0.3,
                 max_wind_lag=12,
                 k_geo_neighbors=5,
                 k_feature_neighbors=8,
                 residual_weight=0.3,
                 layer_type='sage',
                 num_gnn_layers=1,
                 use_feature_view=True,
                 use_geo_view=True,
                 use_prop_view=True):
        """
        Args:
            n_stations: Number of weather stations
            n_features: Number of input features
            hidden_dim: Hidden dimension size
            dropout: Dropout rate
            max_wind_lag: Maximum temporal lag for wind metapath
            k_geo_neighbors: Number of geographic neighbors
            k_feature_neighbors: Number of feature-similar neighbors
            residual_weight: Weight for residual connection
            layer_type: Type of graph layer ('sage', 'gcn', or 'gat')
            num_gnn_layers: Number of stacked GNN layers in each metapath
            use_feature_view: Enable feature similarity metapaths (pressure, temp, humidity, wind_speed)
            use_geo_view: Enable geographic proximity metapath
            use_prop_view: Enable wind propagation metapath (directional)
        """
        super().__init__()

        self.n_stations = n_stations
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.residual_weight = residual_weight
        self.layer_type = layer_type
        self.num_gnn_layers = num_gnn_layers

        # View configuration
        self.use_feature_view = use_feature_view
        self.use_geo_view = use_geo_view
        self.use_prop_view = use_prop_view

        # Validate at least one view is enabled
        if not (use_feature_view or use_geo_view or use_prop_view):
            raise ValueError("At least one view must be enabled (feature_view, geo_view, or prop_view)")
        
        # 1. Feature encoder (encode each scalar feature value -> hidden_dim)
        # We encode each feature separately so we can apply feature-level interaction.
        self.feature_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Metapath modules
        # Geographic metapath
        if self.use_geo_view:
            self.geo_metapath = GeographicMetapath(
                hidden_dim=hidden_dim,
                k_neighbors=k_geo_neighbors,
                distance_scale=200.0,
                layer_type=layer_type,
                num_layers=num_gnn_layers
            )
        else:
            self.geo_metapath = None

        # COMMENTED OUT: Cross-feature spatial metapaths
        # self.pressure_to_wind = CrossFeatureSpatialMetapath2(
        #     hidden_dim=hidden_dim,
        #     source_feature='pressure',
        #     target_feature='wind_speed',
        #     n_stations=n_stations,
        #     top_k=5
        # )
        #
        # self.temp_to_wind = CrossFeatureSpatialMetapath2(
        #     hidden_dim=hidden_dim,
        #     source_feature='temperature',
        #     target_feature='wind_speed',
        #     n_stations=n_stations,
        #     top_k=5
        # )

        # Individual feature similarity metapaths
        if self.use_feature_view:
            self.pressure_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='pressure',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers,
            )

            self.temp_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='temperature',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers,
            )

            self.humidity_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='humidity',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers,
            )

            self.wind_speed_metapath = FeatureSimilarityMetapath(
                hidden_dim=hidden_dim,
                feature_name='wind_speed',
                top_k=k_feature_neighbors,
                lookback_hours=24,
                layer_type=layer_type,
                num_layers=num_gnn_layers,
            )
        else:
            self.pressure_metapath = None
            self.temp_metapath = None
            self.humidity_metapath = None
            self.wind_speed_metapath = None

        # Multi-temporal wind metapath (directional)
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
            self.wind_metapath = None
            self.use_wind_metapath = False

        # 3. Metapath fusion
        # Calculate number of active metapaths
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

        # 2. Create feature-specific temporal representations using SHARED GRU
        # Apply temporal encoding and then use the same GRU for all features
        feature_names_for_metapath = ['pressure', 'humidity', 'temperature', 'wind_speed']
        h_feature_specific = {}

        # Apply temporal encoding to each feature
        h_feat_encoded_list = []
        for feat_idx in range(n_features):
            h_feat_t = h_feat[:, :, :, feat_idx, :]  # [batch, N, lookback, hidden_dim]
            h_feat_t_encoded = self.temporal_encoding(h_feat_t, timestamps)
            h_feat_encoded_list.append(h_feat_t_encoded)

        # Stack and process through shared GRU efficiently
        # [n_features, batch, N, lookback, hidden_dim] -> [n_features * batch * N, lookback, hidden_dim]
        h_feat_encoded_stacked = torch.stack(h_feat_encoded_list, dim=0)
        h_feat_gru = h_feat_encoded_stacked.reshape(n_features * batch_size * N, lookback, self.hidden_dim)

        # Single GRU pass for all features (SHARED GRU)
        feat_gru_out, _ = self.temporal_decoder(h_feat_gru)
        h_feat_temporal = feat_gru_out[:, -1, :]  # [n_features * batch * N, hidden_dim]

        # Reshape back to separate features
        h_feat_temporal = h_feat_temporal.reshape(n_features, batch_size, N, self.hidden_dim)

        # Assign to feature-specific dictionary
        for feat_idx, feat_name in enumerate(feature_names_for_metapath):
            h_feature_specific[feat_name] = h_feat_temporal[feat_idx]  # [batch, N, hidden_dim]

        # Also create general temporal representation (mean of all features)
        h_encoded = h_feat.mean(dim=3)  # [batch, N, lookback, hidden_dim]
        h_encoded = self.temporal_encoding(h_encoded, timestamps)

        # 3. GRU OVER ENCODED FEATURES (for general representation and metapaths)
        h_gru = h_encoded.reshape(batch_size * N, lookback, self.hidden_dim)
        gru_out, _ = self.temporal_decoder(h_gru)  # [batch*N, lookback, hidden_dim]
        h_temporal = gru_out[:, -1, :]  # [batch*N, hidden_dim]
        h_temporal = h_temporal.reshape(batch_size, N, self.hidden_dim)

        # 4. Apply metapaths with feature-specific node representations
        metapath_embeddings = {}

        # Geographic uses general temporal representation
        if self.use_geo_view:
            metapath_embeddings['geo'] = self.geo_metapath(h_temporal, lat, lon)

        # Individual feature similarity metapaths
        if self.use_feature_view:
            metapath_embeddings['pressure'] = self.pressure_metapath(
                h_feature_specific['pressure'], feature_sequences['pressure']
            )
            metapath_embeddings['temp'] = self.temp_metapath(
                h_feature_specific['temperature'], feature_sequences['temperature']
            )
            metapath_embeddings['humidity'] = self.humidity_metapath(
                h_feature_specific['humidity'], feature_sequences['humidity']
            )
            metapath_embeddings['wind_speed'] = self.wind_speed_metapath(
                h_feature_specific['wind_speed'], feature_sequences['wind_speed']
            )

        # Multi-temporal wind: needs full sequence
        if self.use_wind_metapath:
            h_wind, lag_attention = self.wind_metapath(
                h_encoded,  # Full temporal sequence
                feature_sequences['wind_direction'],
                feature_sequences['wind_speed'],
                lat,
                lon
            )
            metapath_embeddings['wind'] = h_wind
        else:
            lag_attention = None  # No lag attention when disabled

        # COMMENTED OUT: Cross-feature spatial metapaths
        # # Cross-feature spatial metapaths
        # # Pressure → Wind Speed etkisi
        # h_pressure_to_wind, lag_attn_p2w = self.pressure_to_wind(
        #     source_node_features=h_feature_specific['pressure'],
        #     target_node_features=h_feature_specific['wind_speed'],
        #     source_values=feature_sequences['pressure'],
        #     target_values=feature_sequences['wind_speed']
        # )
        # metapath_embeddings['pressure_to_wind'] = h_pressure_to_wind
        #
        # # Temperature → Wind Speed etkisi
        # h_temp_to_wind, lag_attn_t2w = self.temp_to_wind(
        #     source_node_features=h_feature_specific['temperature'],
        #     target_node_features=h_feature_specific['wind_speed'],
        #     source_values=feature_sequences['temperature'],
        #     target_values=feature_sequences['wind_speed']
        # )
        # metapath_embeddings['temp_to_wind'] = h_temp_to_wind
        lag_attn_p2w = None
        lag_attn_t2w = None

        # 5. Fuse metapaths
        h_fused, metapath_attention = self.fusion(metapath_embeddings)

        # Apply layer normalization and add residual temporal info
        # SINGLE residual connection at post-fusion level (no residuals in metapath layers)
        h_fused = self.layer_norm_fusion(h_fused)
        h_fused = h_fused + self.residual_weight * h_temporal

        # 6. Predict
        predictions = self.predictor(h_fused)
        predictions = predictions.squeeze(-1)
        
        # Add station-specific bias
        predictions = predictions + self.station_bias.unsqueeze(0)

        attention_info = {
            'lag_attention': lag_attention,
            'metapath_attention': metapath_attention,
            'cross_feature_lag_attention': {
                'pressure_to_wind': lag_attn_p2w,
                'temp_to_wind': lag_attn_t2w
            }
        }

        return predictions, attention_info
