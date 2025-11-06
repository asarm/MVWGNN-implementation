import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
#                         UTILITY FUNCTIONS
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between points (vectorized)."""
    if not isinstance(lat1, torch.Tensor):
        lat1 = torch.tensor(lat1, dtype=torch.float32)
        lon1 = torch.tensor(lon1, dtype=torch.float32)
        lat2 = torch.tensor(lat2, dtype=torch.float32)
        lon2 = torch.tensor(lon2, dtype=torch.float32)
    
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    
    if lat1.dim() == 1:
        lat1_rad = lat1_rad.unsqueeze(1)
        lon1_rad = lon1_rad.unsqueeze(1)
        lat2_rad = lat2_rad.unsqueeze(0)
        lon2_rad = lon2_rad.unsqueeze(0)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
    
    R = 6371.0  # Earth radius in km
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing from point 1 to point 2 (degrees)."""
    if not isinstance(lat1, torch.Tensor):
        lat1 = torch.tensor(lat1, dtype=torch.float32)
        lon1 = torch.tensor(lon1, dtype=torch.float32)
        lat2 = torch.tensor(lat2, dtype=torch.float32)
        lon2 = torch.tensor(lon2, dtype=torch.float32)
    
    lat1_rad = torch.deg2rad(lat1)
    lat2_rad = torch.deg2rad(lat2)
    dlon_rad = torch.deg2rad(lon2 - lon1)
    
    if lat1.dim() == 1:
        lat1_rad = lat1_rad.unsqueeze(1)
        lat2_rad = lat2_rad.unsqueeze(0)
        dlon_rad = dlon_rad.unsqueeze(0)
    
    y = torch.sin(dlon_rad) * torch.cos(lat2_rad)
    x = torch.cos(lat1_rad) * torch.sin(lat2_rad) - \
        torch.sin(lat1_rad) * torch.cos(lat2_rad) * torch.cos(dlon_rad)
    
    bearing = torch.atan2(y, x)
    bearing = torch.rad2deg(bearing) % 360
    
    return bearing


# ============================================================================
#                         SHARED GNN LAYER
# ============================================================================

class SharedGraphSAGE(nn.Module):
    """
    Shared GraphSAGE layer used across multiple metapaths.
    Separates self and neighbor transformations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.W_self = nn.Linear(in_features, out_features, bias=False)
        self.W_neigh = nn.Linear(in_features, out_features, bias=False)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_self.weight)
        nn.init.xavier_uniform_(self.W_neigh.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features [batch, N, in_features]
            adj: Adjacency matrix [batch, N, N] or [N, N]
        """
        h_self = self.W_self(x)
        
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
        h_neigh = torch.bmm(adj, x)
        h_neigh = self.W_neigh(h_neigh)
        
        output = h_self + h_neigh
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


# ============================================================================
#                    ADJACENCY CONSTRUCTION MODULES
# ============================================================================

class GeographicAdjacency(nn.Module):
    """Construct geographic k-NN adjacency (static)."""
    def __init__(self, k_neighbors=5, distance_scale=200.0):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.distance_scale = distance_scale
        self._cached_adjacency = None
        
    def construct(self, lat, lon):
        """Returns [N, N] normalized adjacency."""
        if self._cached_adjacency is not None:
            return self._cached_adjacency
        
        N = len(lat)
        device = lat.device
        
        distances = haversine_distance(lat, lon, lat, lon)
        A_geo = torch.zeros(N, N, device=device)
        
        for i in range(N):
            _, topk_indices = torch.topk(distances[i], k=self.k_neighbors + 1, largest=False)
            neighbors = topk_indices[1:]
            neighbor_distances = distances[i, neighbors]
            weights = torch.exp(-neighbor_distances / self.distance_scale)
            A_geo[i, neighbors] = weights
        
        row_sum = A_geo.sum(dim=-1, keepdim=True)
        A_geo = A_geo / (row_sum + 1e-8)
        
        self._cached_adjacency = A_geo
        return A_geo


class FeatureSimilarityAdjacency(nn.Module):
    """Construct feature similarity adjacency based on correlation."""
    def __init__(self, feature_name, top_k=8, lookback_hours=24):
        super().__init__()
        self.feature_name = feature_name
        self.top_k = top_k
        self.lookback_hours = lookback_hours
        
    def construct(self, feature_values):
        """
        Args:
            feature_values: [batch, N, lookback]
        Returns:
            [batch, N, N] normalized adjacency
        """
        batch_size, N, lookback = feature_values.shape
        device = feature_values.device
        
        H = min(self.lookback_hours, lookback)
        recent_values = feature_values[:, :, -H:]
        
        A_feat = torch.zeros(batch_size, N, N, device=device)
        
        for b in range(batch_size):
            values_b = recent_values[b]
            
            mean = values_b.mean(dim=1, keepdim=True)
            std = values_b.std(dim=1, keepdim=True)
            values_normalized = (values_b - mean) / (std + 1e-8)
            
            correlation = torch.mm(values_normalized, values_normalized.T) / H
            correlation = torch.clamp(correlation, min=0)
            
            if self.top_k < N:
                for i in range(N):
                    _, topk_indices = torch.topk(correlation[i], k=self.top_k + 1)
                    mask = torch.zeros(N, device=device)
                    mask[topk_indices] = 1.0
                    correlation[i] = correlation[i] * mask
            
            correlation = correlation * (1 - torch.eye(N, device=device))
            A_feat[b] = correlation
        
        row_sum = A_feat.sum(dim=-1, keepdim=True)
        A_feat = A_feat / (row_sum + 1e-8)
        
        return A_feat


class WindPropagationAdjacency(nn.Module):
    """Construct wind-based adjacency for specific temporal lag."""
    def __init__(self, distance_scale=500.0):
        super().__init__()
        self.distance_scale = distance_scale
        
    def construct(self, wind_dirs, wind_speeds, lat, lon, tau):
        """
        Args:
            wind_dirs: [batch, N, lookback]
            wind_speeds: [batch, N, lookback]
            tau: Time lag (hours)
        Returns:
            [batch, N, N] normalized adjacency
        """
        batch_size, N, lookback = wind_dirs.shape
        device = wind_dirs.device
        
        if tau >= lookback:
            return torch.zeros(batch_size, N, N, device=device)
        
        time_idx = lookback - tau - 1
        wind_dir_t = wind_dirs[:, :, time_idx]
        
        distances = haversine_distance(lat, lon, lat, lon)
        bearings = calculate_bearing(lat, lon, lat, lon)
        
        A_tau = torch.zeros(batch_size, N, N, device=device)
        
        for b in range(batch_size):
            wind_i = wind_dir_t[b:b+1, :].T
            
            alignment = torch.cos(torch.deg2rad(wind_i - bearings))
            alignment_positive = torch.clamp(alignment, min=0)
            
            distance_decay = torch.exp(-distances / self.distance_scale)
            # Ensure lag_decay is a tensor on the correct device/dtype before calling torch.exp
            lag_decay = torch.exp(torch.tensor(-tau / 6.0, device=device, dtype=distances.dtype))
            
            edge_weight = alignment_positive * distance_decay * lag_decay
            
            edge_weight = torch.where(
                alignment_positive > 0.5,
                edge_weight,
                torch.zeros_like(edge_weight)
            )
            
            edge_weight = edge_weight * (1 - torch.eye(N, device=device))
            A_tau[b] = edge_weight
        
        row_sum = A_tau.sum(dim=-1, keepdim=True)
        A_tau = A_tau / (row_sum + 1e-8)
        
        return A_tau


# ============================================================================
#                    FEATURE INTERACTION LAYER
# ============================================================================

class FeatureInteractionLayer(nn.Module):
    """Model feature interactions WITHIN each station."""
    def __init__(self, n_features, hidden_dim):
        super().__init__()
        self.feature_adj = nn.Parameter(torch.randn(n_features, n_features))
        self.feature_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, N_stations, n_features, hidden_dim]
        Returns:
            [batch, N_stations, n_features, hidden_dim]
        """
        batch_size, N, n_features, hidden_dim = x.shape
        
        adj = F.softmax(self.feature_adj, dim=-1)
        x_flat = x.reshape(batch_size * N, n_features, hidden_dim)
        
        x_agg = torch.bmm(adj.unsqueeze(0).expand(batch_size * N, -1, -1), x_flat)
        x_transformed = self.feature_mlp(x_agg)
        
        x_out = 0.5 * x_transformed + 0.5 * x_flat
        
        return x_out.reshape(batch_size, N, n_features, hidden_dim)


# ============================================================================
#                    METAPATH FUSION
# ============================================================================

class MetapathFusion(nn.Module):
    """Fuse representations from multiple metapaths using semantic attention."""
    def __init__(self, hidden_dim, n_metapaths):
        super().__init__()
        self.n_metapaths = n_metapaths
        
        self.semantic_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, metapath_embeddings):
        """
        Args:
            metapath_embeddings: dict {name: [batch, N, hidden_dim]}
        Returns:
            h_fused: [batch, N, hidden_dim]
            attention_weights: dict {name: float}
        """
        metapath_names = list(metapath_embeddings.keys())
        h_stacked = torch.stack([metapath_embeddings[name] 
                                 for name in metapath_names], dim=2)
        
        attention_logits = self.semantic_attention(h_stacked)
        attention_logits_avg = attention_logits.mean(dim=[0, 1])
        attention_weights = F.softmax(attention_logits_avg, dim=0)
        
        h_fused = torch.sum(
            attention_weights.view(1, 1, -1, 1) * h_stacked,
            dim=2
        )
        
        h_fused = self.output_proj(h_fused)
        h_fused = F.relu(h_fused)
        
        attention_dict = {name: attention_weights[i].item() 
                         for i, name in enumerate(metapath_names)}
        
        return h_fused, attention_dict


# ============================================================================
#                    MAIN MODEL (SHARED GNN VERSION)
# ============================================================================

class MetapathGNN_Shared(nn.Module):
    """
    Metapath-based GNN with SHARED GraphSAGE layers for parameter efficiency.
    
    Key differences from original:
    - Feature similarity metapaths share 1 GNN
    - Wind lag metapaths share 1 GNN + lag-specific projections
    - Geographic metapath has dedicated GNN (static, special)
    
    Parameter reduction: ~82% (16 GNN layers → 3 GNN layers)
    """
    def __init__(self, 
                 n_stations,
                 n_features=4,
                 hidden_dim=64,
                 dropout=0.2,
                 max_wind_lag=12,
                 k_geo_neighbors=5,
                 k_feature_neighbors=5):
        super().__init__()
        
        self.n_stations = n_stations
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.max_wind_lag = max_wind_lag
        
        # ===== FEATURE ENCODER =====
        self.feature_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)  # Lower dropout for encoder
        )
        
        # ===== FEATURE INTERACTION =====
        self.feature_interaction = FeatureInteractionLayer(n_features, hidden_dim)
        
        # ===== TEMPORAL DECODER (GRU) =====
        self.temporal_decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # ===== ADJACENCY CONSTRUCTORS (NO GNN INSIDE!) =====
        self.geo_adj = GeographicAdjacency(
            k_neighbors=k_geo_neighbors,
            distance_scale=200.0
        )
        
        self.temp_adj = FeatureSimilarityAdjacency(
            feature_name='temperature',
            top_k=k_feature_neighbors,
            lookback_hours=24
        )
        
        self.humidity_adj = FeatureSimilarityAdjacency(
            feature_name='humidity',
            top_k=k_feature_neighbors,
            lookback_hours=24
        )
        
        self.wind_speed_adj = FeatureSimilarityAdjacency(
            feature_name='wind_speed',
            top_k=k_feature_neighbors,
            lookback_hours=24
        )
        
        self.wind_prop_adj = WindPropagationAdjacency(distance_scale=500.0)
        
        # ===== SHARED GNN LAYERS =====
        # 1. Geographic GNN (dedicated, static nature)
        self.geo_gnn = SharedGraphSAGE(hidden_dim, hidden_dim)
        
        # 2. Feature similarity GNN (shared across temp/humidity/wind_speed)
        self.feature_similarity_gnn = SharedGraphSAGE(hidden_dim, hidden_dim)
        
        # 3. Wind propagation GNN (shared across all lags)
        self.wind_gnn = SharedGraphSAGE(hidden_dim, hidden_dim)
        
        # Lag-specific projections (lightweight, preserves expressiveness)
        self.wind_lag_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(max_wind_lag)
        ])
        
        # Wind lag attention
        self.lag_attention_logits = nn.Parameter(torch.randn(max_wind_lag))
        self.wind_output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # ===== METAPATH FUSION =====
        n_metapaths = 5  # geo + temp + humidity + wind_speed + wind
        self.fusion = MetapathFusion(hidden_dim, n_metapaths)
        
        # ===== PREDICTION HEAD =====
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, lat, lon, feature_sequences):
        """
        Args:
            x: [batch, N, lookback, n_features]
            lat, lon: [N]
            feature_sequences: dict {
                'temperature': [batch, N, lookback],
                'humidity': [batch, N, lookback],
                'wind_speed': [batch, N, lookback],
                'wind_direction': [batch, N, lookback]
            }
        """
        batch_size, N, lookback, n_features = x.shape
        
        # ===== 1. FEATURE ENCODING =====
        x_feat_flat = x.reshape(batch_size * N * lookback * n_features, 1)
        h_feat = self.feature_encoder(x_feat_flat)
        h_feat = h_feat.reshape(batch_size, N, lookback, n_features, self.hidden_dim)
        
        # ===== 2. FEATURE INTERACTION =====
        h_interacted = []
        for t in range(lookback):
            h_t = h_feat[:, :, t, :, :]
            h_t_inter = self.feature_interaction(h_t)
            h_interacted.append(h_t_inter)
        
        h_interacted = torch.stack(h_interacted, dim=2)
        h_encoded = h_interacted.mean(dim=3)  # [batch, N, lookback, hidden_dim]
        
        # ===== 3. TEMPORAL MODELING (GRU) =====
        h_gru = h_encoded.reshape(batch_size * N, lookback, self.hidden_dim)
        gru_out, _ = self.temporal_decoder(h_gru)
        h_temporal = gru_out[:, -1, :]
        h_temporal = h_temporal.reshape(batch_size, N, self.hidden_dim)
        
        # ===== 4. METAPATH EMBEDDINGS =====
        metapath_embeddings = {}
        
        # Geographic metapath (dedicated GNN)
        A_geo = self.geo_adj.construct(lat, lon)
        h_geo = self.geo_gnn(h_temporal, A_geo)
        metapath_embeddings['geo'] = F.relu(h_geo)
        
        # Temperature metapath (shared feature GNN)
        A_temp = self.temp_adj.construct(feature_sequences['temperature'])
        h_temp = self.feature_similarity_gnn(h_temporal, A_temp)
        metapath_embeddings['temp'] = F.relu(h_temp)
        
        # Humidity metapath (shared feature GNN)
        A_humidity = self.humidity_adj.construct(feature_sequences['humidity'])
        h_humidity = self.feature_similarity_gnn(h_temporal, A_humidity)
        metapath_embeddings['humidity'] = F.relu(h_humidity)
        
        # Wind speed metapath (shared feature GNN)
        A_ws = self.wind_speed_adj.construct(feature_sequences['wind_speed'])
        h_ws = self.feature_similarity_gnn(h_temporal, A_ws)
        metapath_embeddings['wind_speed'] = F.relu(h_ws)
        
        # Multi-temporal wind metapath (shared wind GNN + lag projections)
        h_lags = []
        for tau in range(self.max_wind_lag):
            A_tau = self.wind_prop_adj.construct(
                feature_sequences['wind_direction'],
                feature_sequences['wind_speed'],
                lat, lon, tau
            )
            
            if tau < lookback:
                h_tau_source = h_encoded[:, :, lookback - tau - 1, :]
            else:
                h_tau_source = h_temporal
            
            # Shared GNN
            h_agg = self.wind_gnn(h_tau_source, A_tau)
            
            # Lag-specific projection
            h_transformed = self.wind_lag_projs[tau](h_agg)
            h_transformed = F.relu(h_transformed)
            
            h_lags.append(h_transformed)
        
        # Aggregate lags with attention
        h_stacked = torch.stack(h_lags, dim=2)
        lag_attention = F.softmax(self.lag_attention_logits, dim=0)
        h_wind_agg = torch.sum(
            lag_attention.view(1, 1, -1, 1) * h_stacked,
            dim=2
        )
        h_wind = self.wind_output_proj(h_wind_agg)
        metapath_embeddings['wind'] = F.relu(h_wind)
        
        # ===== 5. METAPATH FUSION =====
        h_fused, metapath_attention = self.fusion(metapath_embeddings)
        
        # ===== 6. PREDICTION =====
        predictions = self.predictor(h_fused)
        predictions = predictions.squeeze(-1)
        
        attention_info = {
            'lag_attention': lag_attention,
            'metapath_attention': metapath_attention
        }
        
        return predictions, attention_info


# ============================================================================
#                         UTILITY: PARAMETER COUNT
# ============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Breakdown
    breakdown = {}
    breakdown['feature_encoder'] = sum(p.numel() for p in model.feature_encoder.parameters())
    breakdown['feature_interaction'] = sum(p.numel() for p in model.feature_interaction.parameters())
    breakdown['temporal_decoder'] = sum(p.numel() for p in model.temporal_decoder.parameters())
    breakdown['geo_gnn'] = sum(p.numel() for p in model.geo_gnn.parameters())
    breakdown['feature_similarity_gnn'] = sum(p.numel() for p in model.feature_similarity_gnn.parameters())
    breakdown['wind_gnn'] = sum(p.numel() for p in model.wind_gnn.parameters())
    breakdown['wind_lag_projs'] = sum(p.numel() for p in model.wind_lag_projs.parameters())
    breakdown['fusion'] = sum(p.numel() for p in model.fusion.parameters())
    breakdown['predictor'] = sum(p.numel() for p in model.predictor.parameters())
    
    return total, breakdown


if __name__ == "__main__":
    # Test model
    print("="*60)
    print("TESTING MetapathGNN_Shared")
    print("="*60)
    
    n_stations = 30
    n_features = 4
    hidden_dim = 64
    batch_size = 8
    lookback = 24
    
    model = MetapathGNN_Shared(
        n_stations=n_stations,
        n_features=n_features,
        hidden_dim=hidden_dim,
        dropout=0.2,
        max_wind_lag=12,
        k_geo_neighbors=5,
        k_feature_neighbors=5
    )
    
    total_params, breakdown = count_parameters(model)
    
    print(f"\n✓ Model initialized")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"\nParameter breakdown:")
    for name, count in breakdown.items():
        print(f"  {name:25s}: {count:8,} ({count/total_params*100:5.2f}%)")
    
    # Create dummy data
    x = torch.randn(batch_size, n_stations, lookback, n_features)
    lat = torch.randn(n_stations) * 45
    lon = torch.randn(n_stations) * 90
    feature_sequences = {
        'temperature': torch.randn(batch_size, n_stations, lookback),
        'humidity': torch.randn(batch_size, n_stations, lookback),
        'wind_speed': torch.randn(batch_size, n_stations, lookback),
        'wind_direction': torch.rand(batch_size, n_stations, lookback) * 360
    }
    
    # Forward pass
    predictions, attention_info = model(x, lat, lon, feature_sequences)
    
    print(f"\n✓ Forward pass successful")
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {predictions.shape}")
    print(f"✓ Output range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
    
    print(f"\n✓ Attention Info:")
    print(f"  Lag attention shape: {attention_info['lag_attention'].shape}")
    print(f"  Metapath attention: {list(attention_info['metapath_attention'].keys())}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)