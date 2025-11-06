import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer with mean aggregation.
    Separates self and neighbor transformations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Separate transformations for self and neighbors
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
                 Should be pre-normalized (row-sum = 1)
        
        Returns:
            Node features [batch, N, out_features]
        """
        # Self transformation
        h_self = self.W_self(x)  # [batch, N, out_features]
        
        # Neighbor aggregation (mean)
        if adj.dim() == 2:
            # Static adjacency: expand for batch
            adj = adj.unsqueeze(0).expand(x.size(0), -1, -1)
        
        h_neigh = torch.bmm(adj, x)  # [batch, N, in_features]
        h_neigh = self.W_neigh(h_neigh)  # [batch, N, out_features]
        
        # Combine
        output = h_self + h_neigh
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate haversine distance between points (vectorized).
    
    Args:
        lat1, lon1: [N] or scalar
        lat2, lon2: [N] or scalar
    
    Returns:
        Distance in kilometers [N, N] if inputs are [N]
    """
    # Convert to tensors if needed
    if not isinstance(lat1, torch.Tensor):
        lat1 = torch.tensor(lat1, dtype=torch.float32)
        lon1 = torch.tensor(lon1, dtype=torch.float32)
        lat2 = torch.tensor(lat2, dtype=torch.float32)
        lon2 = torch.tensor(lon2, dtype=torch.float32)
    
    # Convert to radians
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    
    # Expand for pairwise computation if needed
    if lat1.dim() == 1:
        lat1_rad = lat1_rad.unsqueeze(1)
        lon1_rad = lon1_rad.unsqueeze(1)
        lat2_rad = lat2_rad.unsqueeze(0)
        lon2_rad = lon2_rad.unsqueeze(0)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
    c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
    
    R = 6371.0  # Earth radius in km
    return R * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing from point 1 to point 2 (degrees).
    
    Returns:
        Bearing in degrees [0, 360)
    """
    if not isinstance(lat1, torch.Tensor):
        lat1 = torch.tensor(lat1, dtype=torch.float32)
        lon1 = torch.tensor(lon1, dtype=torch.float32)
        lat2 = torch.tensor(lat2, dtype=torch.float32)
        lon2 = torch.tensor(lon2, dtype=torch.float32)
    
    lat1_rad = torch.deg2rad(lat1)
    lat2_rad = torch.deg2rad(lat2)
    dlon_rad = torch.deg2rad(lon2 - lon1)
    
    # Expand for pairwise if needed
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

class GeographicMetapath(nn.Module):
    """
    Static geographic proximity metapath.
    Connects stations based on spatial distance using k-NN.
    """
    def __init__(self, hidden_dim, k_neighbors=5, distance_scale=200.0):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.distance_scale = distance_scale
        
        self.sage = GraphSAGELayer(hidden_dim, hidden_dim)
        self._cached_adjacency = None
        
    def construct_geo_adjacency(self, lat, lon):
        """
        Construct k-NN geographic adjacency (static).
        
        Args:
            lat, lon: [N] tensors
        
        Returns:
            A_geo: [N, N] normalized adjacency
        """
        N = len(lat)
        device = lat.device
        
        # Compute pairwise distances
        distances = haversine_distance(lat, lon, lat, lon)  # [N, N]
        
        # Keep only k-nearest neighbors per node
        A_geo = torch.zeros(N, N, device=device)
        
        for i in range(N):
            # Get k+1 nearest (including self at k=0)
            _, topk_indices = torch.topk(distances[i], k=self.k_neighbors + 1, largest=False)
            
            # Exclude self (first element)
            neighbors = topk_indices[1:]
            neighbor_distances = distances[i, neighbors]
            
            # Distance decay weights
            weights = torch.exp(-neighbor_distances / self.distance_scale)
            A_geo[i, neighbors] = weights
        
        # Row-normalize
        row_sum = A_geo.sum(dim=-1, keepdim=True)
        A_geo = A_geo / (row_sum + 1e-8)
        
        return A_geo
    
    def forward(self, node_features, lat, lon):
        """
        Args:
            node_features: [batch, N, hidden_dim]
            lat, lon: [N] tensors
        
        Returns:
            h_geo: [batch, N, hidden_dim]
        """
        # Cache static adjacency
        if self._cached_adjacency is None:
            self._cached_adjacency = self.construct_geo_adjacency(lat, lon)
        
        A_geo = self._cached_adjacency
        
        # GraphSAGE aggregation
        h_geo = self.sage(node_features, A_geo)
        h_geo = F.relu(h_geo)
        
        return h_geo
    
class FeatureSimilarityMetapath(nn.Module):
    """
    Dynamic feature similarity metapath.
    Connects stations with similar feature values (e.g., temperature, wind speed).
    """
    def __init__(self, hidden_dim, feature_name, top_k=10, lookback_hours=24):
        super().__init__()
        self.feature_name = feature_name
        self.top_k = top_k
        self.lookback_hours = lookback_hours
        
        self.sage = GraphSAGELayer(hidden_dim, hidden_dim)
        
    def construct_similarity_adjacency(self, feature_values):
        """
        Construct adjacency based on feature similarity (correlation).
        
        Args:
            feature_values: [batch, N_stations, lookback] - historical values
        
        Returns:
            A_feat: [batch, N, N] - normalized adjacency
        """
        batch_size, N, lookback = feature_values.shape
        device = feature_values.device
        
        # Use recent history
        H = min(self.lookback_hours, lookback)
        recent_values = feature_values[:, :, -H:]  # [batch, N, H]
        
        A_feat = torch.zeros(batch_size, N, N, device=device)
        
        for b in range(batch_size):
            values_b = recent_values[b]  # [N, H]
            
            # Normalize (z-score)
            mean = values_b.mean(dim=1, keepdim=True)
            std = values_b.std(dim=1, keepdim=True)
            values_normalized = (values_b - mean) / (std + 1e-8)
            
            # Pearson correlation: [N, N]
            correlation = torch.mm(values_normalized, values_normalized.T) / H
            
            # Keep only positive correlations
            correlation = torch.clamp(correlation, min=0)
            
            # Top-k filtering
            if self.top_k < N:
                # For each station, keep only top-k correlated stations
                for i in range(N):
                    _, topk_indices = torch.topk(correlation[i], k=self.top_k + 1)
                    # Create mask: keep only top-k
                    mask = torch.zeros(N, device=device)
                    mask[topk_indices] = 1.0
                    correlation[i] = correlation[i] * mask
            
            # Remove self-loops
            correlation = correlation * (1 - torch.eye(N, device=device))
            
            A_feat[b] = correlation
        
        # Row-normalize
        row_sum = A_feat.sum(dim=-1, keepdim=True)
        A_feat = A_feat / (row_sum + 1e-8)
        
        return A_feat
    
    def forward(self, node_features, feature_values):
        """
        Args:
            node_features: [batch, N, hidden_dim]
            feature_values: [batch, N, lookback] - raw feature history
        
        Returns:
            h_feat: [batch, N, hidden_dim]
        """
        A_feat = self.construct_similarity_adjacency(feature_values)
        
        # GraphSAGE aggregation
        h_feat = self.sage(node_features, A_feat)
        h_feat = F.relu(h_feat)
        
        return h_feat
    
class MultiTemporalWindMetapath(nn.Module):
    """
    Multi-temporal wind-informed metapath.
    Captures wind propagation effects across multiple time lags.
    
    This is the CORE NOVELTY of the model!
    """
    def __init__(self, hidden_dim, max_lag=12, distance_scale=500.0):
        super().__init__()
        self.max_lag = max_lag
        self.distance_scale = distance_scale
        
        # Learnable lag attention weights
        self.lag_attention_logits = nn.Parameter(torch.randn(max_lag))
        
        # GraphSAGE layers for each lag (can share or separate)
        self.sage_layers = nn.ModuleList([
            GraphSAGELayer(hidden_dim, hidden_dim) for _ in range(max_lag)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def construct_lag_adjacency(self, wind_dirs, wind_speeds, lat, lon, tau):
        """
        Construct adjacency matrix for specific temporal lag tau.
        """
        batch_size, N, lookback = wind_dirs.shape
        device = wind_dirs.device
        
        # Use wind data from (lookback - tau - 1) timestep
        if tau >= lookback:
            return torch.zeros(batch_size, N, N, device=device)
        
        time_idx = lookback - tau - 1
        wind_dir_t = wind_dirs[:, :, time_idx]  # [batch, N]
        wind_speed_t = wind_speeds[:, :, time_idx]  # [batch, N] (NORMALIZED!)
        
        # Compute pairwise distances and bearings
        distances = haversine_distance(lat, lon, lat, lon)  # [N, N]
        bearings = calculate_bearing(lat, lon, lat, lon)  # [N, N]
        
        A_tau = torch.zeros(batch_size, N, N, device=device)
        
        for b in range(batch_size):
            wind_i = wind_dir_t[b:b+1, :].T  # [N, 1]
            
            # Alignment: cos(wind_direction - bearing)
            alignment = torch.cos(torch.deg2rad(wind_i - bearings))  # [N, N]
            
            # Use alignment directly as weight 
            # Positive alignment = downwind, negative = upwind
            # Keep only positive (downwind) but use soft weighting
            alignment_positive = torch.clamp(alignment, min=0)  # [N, N]
            
            # Distance decay
            distance_decay = torch.exp(-distances / self.distance_scale)
            
            # Simplified temporal component
            # Don't try to match exact travel time, just use lag-based decay
            lag_decay = torch.exp(torch.tensor(-tau / 6.0, device=device))  # Decay over 6 hours
            
            # Edge weight: alignment × distance × lag
            edge_weight = alignment_positive * distance_decay * lag_decay
            
            # keep edges where alignment > 0.75
            edge_weight = torch.where(
                alignment_positive > 0.5, 
                edge_weight, 
                torch.zeros_like(edge_weight)
            )
            
            # Remove self-loops
            edge_weight = edge_weight * (1 - torch.eye(N, device=device))
            
            A_tau[b] = edge_weight
        
        # Row-normalize
        row_sum = A_tau.sum(dim=-1, keepdim=True)
        A_tau = A_tau / (row_sum + 1e-8)
        
        return A_tau
    
    def forward(self, node_features, wind_dirs, wind_speeds, lat, lon):
        """
        Args:
            node_features: [batch, N, lookback, hidden_dim]
            wind_dirs: [batch, N, lookback] - wind directions (degrees)
            wind_speeds: [batch, N, lookback] - wind speeds
            lat, lon: [N] - station coordinates
        
        Returns:
            h_wind: [batch, N, hidden_dim]
            lag_attention: [max_lag] - learned importance weights
        """
        batch_size, N, lookback, hidden_dim = node_features.shape
        
        # Use most recent features for current state
        h_current = node_features[:, :, -1, :]  # [batch, N, hidden_dim]
        
        # Aggregate across multiple lags
        h_lags = []
        
        for tau in range(self.max_lag):
            # Construct adjacency for this lag
            A_tau = self.construct_lag_adjacency(
                wind_dirs, wind_speeds, lat, lon, tau
            )  # [batch, N, N]
            
            # Select features from (t - tau) if available
            if tau < lookback:
                h_tau_source = node_features[:, :, lookback - tau - 1, :]
            else:
                h_tau_source = h_current  # Fall back to current
            
            # GraphSAGE aggregation
            h_transformed = self.sage_layers[tau](h_tau_source, A_tau)
            h_transformed = F.relu(h_transformed)
            
            h_lags.append(h_transformed)
        
        # Stack: [batch, N, max_lag, hidden_dim]
        h_stacked = torch.stack(h_lags, dim=2)
        
        # Compute lag attention weights
        lag_attention = F.softmax(self.lag_attention_logits, dim=0)  # [max_lag]
        
        # Weighted aggregation
        h_aggregated = torch.sum(
            lag_attention.view(1, 1, -1, 1) * h_stacked,
            dim=2
        )  # [batch, N, hidden_dim]
        
        # Final projection
        h_out = self.output_proj(h_aggregated)
        h_out = F.relu(h_out)
        
        return h_out, lag_attention
    
class MetapathFusion(nn.Module):
    def __init__(self, hidden_dim, n_metapaths):
        super().__init__()
        
        # ADD: Multi-head attention (more expressive)
        self.n_heads = 4
        self.semantic_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, metapath_embeddings):
        """
        Args:
            metapath_embeddings: dict {
                'wind': [batch, N, hidden_dim],
                'geo': [batch, N, hidden_dim],
                'temp': [batch, N, hidden_dim],
                ...
            }
        
        Returns:
            h_fused: [batch, N, hidden_dim]
            attention_weights: dict {metapath_name: float}
        """
        # Stack metapath embeddings: [batch, N, n_metapaths, hidden_dim]
        metapath_names = list(metapath_embeddings.keys())
        h_stacked = torch.stack([metapath_embeddings[name]
                                 for name in metapath_names], dim=2)

        # Prepare for multi-head attention:
        # treat metapath dimension as sequence length S = n_metapaths,
        # and attend across those tokens for each (batch, station).
        batch_size, N, S, hidden_dim = h_stacked.shape

        # reshape to (batch*N, S, hidden_dim) because MultiheadAttention with batch_first=True
        h_reshaped = h_stacked.reshape(batch_size * N, S, hidden_dim)

        # MultiheadAttention expects (query, key, value)
        # attn_out: (batch*N, S, hidden_dim)
        # attn_weights: (batch*N, S, S) -- averaged over heads by default
        attn_out, attn_weights = self.semantic_attention(h_reshaped, h_reshaped, h_reshaped)

        # attn_weights: (batch*N, query_len=S, key_len=S)
        # Aggregate to per-metapath (source/token) importance:
        # 1) average over queries -> (batch*N, S)
        # 2) average over batch*N -> (S,)
        attn_per_source = attn_weights.mean(dim=1).mean(dim=0)  # (S,)

        # Re-normalize to ensure a valid distribution across metapaths
        attention_weights = F.softmax(attn_per_source, dim=0)  # (S,)

        # Weighted sum across metapaths -> [batch, N, hidden_dim]
        h_fused = torch.sum(
            attention_weights.view(1, 1, S, 1) * h_stacked,
            dim=2
        )
        
        # Final projection
        h_fused = self.output_proj(h_fused)
        h_fused = F.relu(h_fused)

        # Return attention weights as dict for interpretability
        attention_dict = {name: float(attention_weights[i].item())
                          for i, name in enumerate(metapath_names)}

        return h_fused, attention_dict