import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from temporalEncoders.conv1dEncoder import date2cyclic_features
from temporalEncoders.positionalEncoder import PositionalEncoder

class FeatureGraphLayer(nn.Module):
    """
    Learn feature interactions via graph structure.
    Captures pairwise interactions between features (e.g., temp ↔ humidity ↔ wind_speed)
    using a learnable feature adjacency matrix.
    """
    def __init__(self, n_features=4):
        super().__init__()
        # Learnable adjacency for features
        self.feature_adj = nn.Parameter(torch.randn(n_features, n_features))
        
        # Feature transformation
        self.feature_mlp = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, n_stations, window, n_features]
        Returns:
            x_out: [batch, n_stations, window, n_features]
        """
        batch, n_stations, window, n_features = x.shape
        
        # Reshape: [batch*n_stations*window, n_features]
        x_flat = x.reshape(-1, n_features)
        
        # Compute feature adjacency (softmax per feature for normalization)
        adj = F.softmax(self.feature_adj, dim=-1)  # [n_features, n_features]
        
        # Feature aggregation: f_i' = Σ_j adj[i,j] * f_j
        x_agg = torch.matmul(x_flat, adj.T)  # [batch*n_stations*window, n_features]
        
        # Transform aggregated features
        x_transformed = self.feature_mlp(x_agg)
        
        # Residual connection to preserve original features
        x_out = 0.5 * x_transformed + 0.5 * x_flat
        
        # Reshape back
        return x_out.reshape(batch, n_stations, window, n_features)

class SimplifiedGate(nn.Module):
    """
    Single learnable weight per layer (shared across all stations)
    """
    def __init__(self):
        super().__init__()
        # Single scalar weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, gcn_output, temporal_proj):
        # Normalize to [0, 1]
        alpha = torch.sigmoid(self.alpha)
        
        # Simple weighted combination
        output = alpha * gcn_output + (1 - alpha) * temporal_proj
        
        return output

class GatedResidualFusion(nn.Module):
    """
    Learnable fusion between GCN output and temporal embedding
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, gcn_output, temporal_proj):
        # Concatenate features
        combined = torch.cat([gcn_output, temporal_proj], dim=-1)
        
        # Learn gate weights [0, 1]
        alpha = self.gate(combined)  # [batch, n_stations, hidden_dim]
        
        # Adaptive fusion
        output = alpha * gcn_output + (1 - alpha) * temporal_proj
        
        return output

class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolutional Layer (GCN).
    Performs message passing: H' = σ(D^-1/2 A D^-1/2 H W)
    For simplicity, we'll use: H' = σ(A H W) with A normalized
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Args:
            x: Node features [batch_size, n_nodes, in_features]
            adj: Adjacency matrix [batch_size, n_nodes, n_nodes]
        Returns:
            Node features [batch_size, n_nodes, out_features]
        """
        # Linear transformation: [batch, n_nodes, in_features] @ [in_features, out_features]
        support = torch.matmul(x, self.weight)  # [batch, n_nodes, out_features]
        
        # Message passing: [batch, n_nodes, n_nodes] @ [batch, n_nodes, out_features]
        output = torch.matmul(adj, support)  # [batch, n_nodes, out_features]
        
        if self.bias is not None:
            output = output + self.bias
            
        return output

class GraphTemporalModel(nn.Module):
    """
    Graph-based temporal prediction model.
    
    Architecture:
    1. Conv1DTemporalEncoder encodes temporal features for each station
    2. PositionalEncoder learns positional embeddings for each station
    3. Adjacency matrix is computed from positional embeddings (outer product)
    4. Multiple GCN layers process node features (temporal embeddings) using the adjacency
    5. Predict only the target station's (Vancouver) next wind speed
    """
    def __init__(self, n_stations, n_features=4, temporal_embed_dim=64, 
                 positional_embed_dim=32, gcn_hidden_dim=64, num_gcn_layers=1,
                 dropout=0.3, target_station_idx=0, temporal_encoder_type='conv1d'):
        """
        Args:
            n_stations: Number of weather stations (cities)
            n_features: Number of input features per station
            temporal_embed_dim: Embedding dimension for temporal encoder
            positional_embed_dim: Embedding dimension for positional encoder
            gcn_hidden_dim: Hidden dimension for GCN layers
            num_gcn_layers: Number of GCN layers (default: 1)
            dropout: Dropout rate
            target_station_idx: Index of the target station (Vancouver)
            temporal_encoder_type: Type of temporal encoder to use
                Options: 'conv1d' (default), 'dilated', 'stacked', 'wavenet'
        """
        super().__init__()
        
        self.n_stations = n_stations
        self.n_features = n_features  # Store n_features for reference
        self.target_station_idx = target_station_idx
        self.temporal_embed_dim = temporal_embed_dim
        self.positional_embed_dim = positional_embed_dim
        self.num_gcn_layers = num_gcn_layers
        self.temporal_encoder_type = temporal_encoder_type
        
        self.temporal_embeddings = None
        self.positional_embeddings = None   

        # Feature graph layer to capture pairwise feature interactions
        self.feature_graph = FeatureGraphLayer(n_features=n_features)
        
        # Temporal encoder (shared across all stations)
        # Original Conv1D encoder (ModuleDict)
        self.temporal_encoder = nn.ModuleDict({
            'conv1': nn.Conv1d(n_features, temporal_embed_dim//2, kernel_size=3, padding=1),
            'conv2': nn.Conv1d(n_features, temporal_embed_dim//2, kernel_size=7, padding=3),
            'gru': nn.GRU(input_size=temporal_embed_dim, hidden_size=temporal_embed_dim, 
                         batch_first=True, bidirectional=False),
            'pool': nn.AdaptiveAvgPool1d(1),
            'dropout': nn.Dropout(dropout),
            'cyclic_embedder': nn.Linear(4, temporal_embed_dim//4),  # 4 cyclic features
            'fusion': nn.Linear(temporal_embed_dim + temporal_embed_dim//4, temporal_embed_dim)
        })

        # Positional encoder for learning station embeddings
        self.positional_encoder = PositionalEncoder(
            n_stations=n_stations,
            embedding_dim=positional_embed_dim,
            dropout=dropout
        )
        
        # Geographic bias parameters
        # alpha: learnable weight for distance bias (how important is geography?)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        # sigma: scale parameter for distance bias (in km)
        self.sigma = nn.Parameter(torch.tensor(250.0))
        
        # Edge prediction MLP
        # Takes concatenated node embeddings and predicts edge strength
        self.edge_predictor = nn.Sequential(
            nn.Linear(positional_embed_dim * 2, positional_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(positional_embed_dim, 1)
        )
        
        # Multiple GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            if i == 0:
                # First layer: temporal_embed_dim -> gcn_hidden_dim
                self.gcn_layers.append(GraphConvLayer(temporal_embed_dim, gcn_hidden_dim))
            else:
                # Subsequent layers: gcn_hidden_dim -> gcn_hidden_dim
                self.gcn_layers.append(GraphConvLayer(gcn_hidden_dim, gcn_hidden_dim))
        
        # Project temporal embeddings to GCN hidden dim for residual averaging
        self.temporal_proj = nn.Linear(temporal_embed_dim, gcn_hidden_dim)
        
        # Gated fusion module
        # self.gated_fusion = GatedResidualFusion(gcn_hidden_dim)

        # Final prediction layer (per-node)
        self.predictor = nn.Sequential(
            nn.Linear(gcn_hidden_dim, gcn_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gcn_hidden_dim // 2, 1)  # Predict next wind speed
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def encode_temporal_features(self, x, dayofyear=None, hourofday=None):
        """
        Encode temporal features using the selected temporal encoder.
        
        Args:
            x: [batch_size, n_stations, lookback, n_features]
            dayofyear: Day of year for temporal encoding (optional)
            hourofday: Hour of day for temporal encoding (optional)
        Returns:
            Temporal embeddings: [batch_size, n_stations, temporal_embed_dim]
        """
        batch_size, n_stations, lookback, n_features = x.shape
        
        # Apply feature graph FIRST to learn pairwise feature interactions
        x = self.feature_graph(x)  # [batch, n_stations, lookback, n_features]
        
        # ========== ORIGINAL CONV1D ENCODER ==========
        # Reshape to process all stations in batch
        x_reshaped = x.view(batch_size * n_stations, lookback, n_features)
        x_reshaped = x_reshaped.transpose(1, 2)  # [batch*n_stations, n_features, lookback]
        
        # Multi-scale feature extraction
        feat1 = F.relu(self.temporal_encoder['conv1'](x_reshaped))
        feat1 = self.temporal_encoder['dropout'](feat1)
        
        feat2 = F.relu(self.temporal_encoder['conv2'](x_reshaped))
        feat2 = self.temporal_encoder['dropout'](feat2)
        
        # Combine
        combined = torch.cat([feat1, feat2], dim=1)  # [batch*n_stations, temporal_embed_dim, lookback]
        
        # GRU processing: [batch*n_stations, lookback, temporal_embed_dim]
        combined = combined.transpose(1, 2)  # [batch*n_stations, lookback, temporal_embed_dim]
        gru_out, _ = self.temporal_encoder['gru'](combined)  # [batch*n_stations, lookback, temporal_embed_dim]
        gru_out = self.temporal_encoder['dropout'](gru_out)
        
        # Transpose back for pooling: [batch*n_stations, temporal_embed_dim, lookback]
        gru_out = gru_out.transpose(1, 2)
        
        # Global pooling
        pooled = self.temporal_encoder['pool'](gru_out).squeeze(-1)  # [batch*n_stations, temporal_embed_dim]
        pooled = self.temporal_encoder['dropout'](pooled)
        
        # Add cyclic temporal features if provided
        if dayofyear is not None and hourofday is not None:
            # Convert to cyclic features
            day_sin, day_cos, hour_sin, hour_cos = date2cyclic_features(dayofyear, hourofday)
            
            # Stack and convert to tensor: [batch, 4]
            cyclical_features = np.stack([day_sin, day_cos, hour_sin, hour_cos], axis=-1)
            cyclical_features = torch.tensor(cyclical_features, dtype=pooled.dtype, device=pooled.device)
            
            # Expand for all stations: [batch*n_stations, 4]
            cyclical_features = cyclical_features.unsqueeze(1).expand(batch_size, n_stations, -1)
            cyclical_features = cyclical_features.reshape(batch_size * n_stations, -1)
            
            # Embed cyclic features
            cyclical_embed = F.relu(self.temporal_encoder['cyclic_embedder'](cyclical_features))
            cyclical_embed = self.temporal_encoder['dropout'](cyclical_embed)
            
            # Concatenate and fuse
            pooled = torch.cat([pooled, cyclical_embed], dim=-1)  # [batch*n_stations, temporal_embed_dim + temporal_embed_dim//4]
            pooled = F.relu(self.temporal_encoder['fusion'](pooled))  # [batch*n_stations, temporal_embed_dim]
            pooled = self.temporal_encoder['dropout'](pooled)
        
        # Reshape back
        temporal_embeddings = pooled.view(batch_size, n_stations, self.temporal_embed_dim)

        return temporal_embeddings
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate haversine distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point (in degrees)
            lat2, lon2: Latitude and longitude of second point (in degrees)
        Returns:
            Distance in kilometers
        """
        # Convert to radians
        lat1_rad = torch.deg2rad(lat1)
        lon1_rad = torch.deg2rad(lon1)
        lat2_rad = torch.deg2rad(lat2)
        lon2_rad = torch.deg2rad(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = torch.sin(dlat/2)**2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon/2)**2
        c = 2 * torch.asin(torch.sqrt(a))
        
        # Earth radius in kilometers
        R = 6371.0
        
        return R * c
    
    def compute_distance_bias(self, latitude, longitude):
        """
        Compute geographic distance bias matrix using haversine distance.
        
        Args:
            latitude: [batch_size, n_stations] or [n_stations]
            longitude: [batch_size, n_stations] or [n_stations]
        Returns:
            Distance bias matrix: [batch_size, n_stations, n_stations]
        """
        # Ensure batch dimension
        if latitude.dim() == 1:
            latitude = latitude.unsqueeze(0)
            longitude = longitude.unsqueeze(0)
        
        batch_size, n_stations = latitude.shape
        
        # Expand dimensions for pairwise computation
        # [batch, n_stations, 1]
        lat1 = latitude.unsqueeze(2)
        lon1 = longitude.unsqueeze(2)
        
        # [batch, 1, n_stations]
        lat2 = latitude.unsqueeze(1)
        lon2 = longitude.unsqueeze(1)
        
        # Compute pairwise distances [batch, n_stations, n_stations]
        distances = self.haversine_distance(lat1, lon1, lat2, lon2)
        
        # Convert distance to similarity bias using exponential decay
        # Clamp sigma to avoid division by very small values
        sigma = torch.clamp(self.sigma, min=10.0)
        distance_bias = torch.exp(-distances / sigma)
        
        return distance_bias
    
    def _apply_wind_modulation(self, adj_logits, wind_directions, latitude, longitude):
        """
        Modulate edge strength based on wind direction (vectorized).
        
        Args:
            adj_logits: [batch_size, n_stations, n_stations] - adjacency logits before softmax
            wind_directions: [batch_size, n_stations] or [n_stations] - wind direction in degrees
            latitude: [batch_size, n_stations] or [n_stations] - latitudes
            longitude: [batch_size, n_stations] or [n_stations] - longitudes
        
        Returns:
            modulated_logits: [batch_size, n_stations, n_stations] - wind-modulated adjacency logits
        """
        device = adj_logits.device
        is_batched = adj_logits.dim() == 3
        
        # Ensure inputs have batch dimension
        if wind_directions.dim() == 1:
            wind_directions = wind_directions.unsqueeze(0)  # (1, N)
        if latitude.dim() == 1:
            latitude = latitude.unsqueeze(0)  # (1, N)
            longitude = longitude.unsqueeze(0)  # (1, N)
        
        B, N = wind_directions.shape
        
        # Stack positions: (B, N, 2) - [lon, lat]
        positions = torch.stack([longitude, latitude], dim=-1)  # (B, N, 2)
        
        # Convert wind direction to unit vectors
        wind_rad = torch.deg2rad(wind_directions)  # (B, N)
        wind_vectors = torch.stack([
            torch.cos(wind_rad),  # x-component (east-west)
            torch.sin(wind_rad)   # y-component (north-south)
        ], dim=-1)  # (B, N, 2)
        
        # Pairwise spatial offsets: (B, N, N, 2)
        # offsets[b, i, j] = position[j] - position[i]  (direction from i to j)
        offsets = positions[:, None, :, :] - positions[:, :, None, :]  # (B, N, N, 2)
        
        # Compute distances
        dists = torch.norm(offsets, dim=-1)  # (B, N, N)
        
        # Normalize to unit vectors (spatial direction from i to j)
        eps = 1e-6
        spatial_dir = offsets / (dists[..., None] + eps)  # (B, N, N, 2)
        
        # Compute alignment: wind[i] · spatial_dir[i→j]
        # wind_vectors: (B, N, 2) → (B, N, 1, 2)
        wind_exp = wind_vectors[:, :, None, :]  # (B, N, 1, 2)
        
        # Dot product: alignment[i,j] = wind[i] · direction[i→j]
        alignment = (wind_exp * spatial_dir).sum(dim=-1)  # (B, N, N)
        # alignment ∈ [-1, 1]:
        #   +1 = perfect downwind (i → j same direction as wind at i)
        #   -1 = perfect upwind (i → j opposite to wind at i)
        #    0 = perpendicular
        
        # Wind modulation formula
        # Downwind: boost (alignment > 0)
        # Upwind: reduce (alignment < 0)
        # Formula: 0.6 + 0.4 * alignment
        # Range: [0.2, 1.0]
        #   - Perfect downwind: 0.6 + 0.4*1 = 1.0 (no change)
        #   - Perfect upwind: 0.6 + 0.4*(-1) = 0.2 (reduce by 80%)
        #   - Perpendicular: 0.6 + 0.4*0 = 0.6 (reduce by 40%)
        
        wind_modulation = 0.6 + 0.4 * alignment  # (B, N, N)
        
        # Self-connections always have modulation = 1 (no effect)
        eye = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)
        wind_modulation = wind_modulation * (1.0 - eye) + eye  # (B, N, N)
        
        # Remove batch dimension if input was not batched
        if not is_batched:
            wind_modulation = wind_modulation[0]  # (N, N)
        
        # Apply modulation to logits (multiplicative)
        return adj_logits * wind_modulation
    
    def compute_adjacency(self, positional_embeddings, remove_self_loops=True, top_k=5,
                         latitude=None, longitude=None, wind_directions=None):
        """
        Compute adjacency matrix using MLP-based edge prediction with geographic bias and wind modulation.
        For each pair of nodes (i,j), concatenates embeddings and predicts edge strength via MLP.
        Adds geographic bias based on haversine distance if lat/lon provided.
        Applies wind direction modulation to create directed edges if wind_directions provided.
        Keeps only top-k connections for each node.
        
        Args:
            positional_embeddings: [batch_size, n_stations, positional_embed_dim]
            remove_self_loops: If True, completely remove self-loops by masking diagonal
            top_k: Number of top connections to keep for each node (default: 5)
            latitude: Latitude values [batch_size, n_stations] or [n_stations] (optional)
            longitude: Longitude values [batch_size, n_stations] or [n_stations] (optional)
            wind_directions: Wind direction in degrees [batch_size, n_stations] or [n_stations] (optional)
        Returns:
            Adjacency matrix: [batch_size, n_stations, n_stations]
        """
        batch_size, n_stations, embed_dim = positional_embeddings.shape
        
        # Expand embeddings for pairwise concatenation
        # [batch, n_stations, 1, embed_dim] - source nodes
        emb_i = positional_embeddings.unsqueeze(2).expand(batch_size, n_stations, n_stations, embed_dim)
        # [batch, 1, n_stations, embed_dim] - target nodes
        emb_j = positional_embeddings.unsqueeze(1).expand(batch_size, n_stations, n_stations, embed_dim)
        
        # Concatenate source and target embeddings
        # [batch, n_stations, n_stations, embed_dim * 2]
        edge_features = torch.cat([emb_i, emb_j], dim=-1)
        
        # Predict edge strength using MLP
        # [batch, n_stations, n_stations, embed_dim * 2] -> [batch, n_stations, n_stations, 1]
        adj_logits = self.edge_predictor(edge_features).squeeze(-1)

        # Add geographic bias if lat/lon provided
        if latitude is not None and longitude is not None:
            distance_bias = self.compute_distance_bias(latitude, longitude)
            # Combine learned edge predictions with geographic bias
            adj_logits = adj_logits + self.alpha * torch.log(distance_bias + 1e-8)
            
            # adj_logits = adj_logits * (distance_bias ** self.alpha) # best result

        # Apply wind direction modulation if provided
        if wind_directions is not None:
            if latitude is not None and longitude is not None:
                adj_logits = self._apply_wind_modulation(adj_logits, wind_directions, latitude, longitude)
        
        if remove_self_loops:
            # Completely remove self-connections by setting diagonal to -inf before softmax
            mask = torch.eye(n_stations, device=adj_logits.device, dtype=torch.bool)
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            adj_logits = adj_logits.masked_fill(mask, float('-inf'))
        
        # Normalize adjacency matrix (row-wise softmax for attention-like weights)
        adj = F.softmax(adj_logits, dim=-1)
        
        # Keep only top-k connections for each node
        # Find top-k values and indices for each row
        top_k_values, top_k_indices = torch.topk(adj, k=min(top_k, n_stations), dim=-1, largest=True)
        
        # Create a new adjacency matrix with only top-k connections
        adj_topk = torch.zeros_like(adj)
        # Scatter top-k values back to their positions
        adj_topk.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        adj = adj_topk
        
        # Re-normalize rows to sum to 1 (important for message passing)
        # This maintains proper probability distribution after top-k filtering
        row_sums = adj.sum(dim=-1, keepdim=True)
        adj = adj / (row_sums + 1e-8)
        
        return adj
    
    def forward(self, x, station_ids=None, latitude=None, longitude=None, 
                dayofyear=None, hourofday=None, wind_directions=None):
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, n_stations, lookback, n_features]
            station_ids: Station indices [batch_size, n_stations] or [n_stations]
            latitude: Latitude values for each station (optional)
            longitude: Longitude values for each station (optional)
            dayofyear: Day of year for temporal encoding (optional)
            hourofday: Hour of day for temporal encoding (optional)
            wind_directions: Wind direction in degrees [batch_size, n_stations] or [n_stations] (optional)
        Returns:
            Predictions for target station: [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # 1. Encode temporal features for all stations (with cyclic temporal features)
        temporal_embeddings = self.encode_temporal_features(
            x, dayofyear=dayofyear, hourofday=hourofday
        )  # [batch, n_stations, temporal_embed_dim]
        # 2. Learn positional embeddings
        if station_ids is None:
            # Create default station IDs [0, 1, 2, ..., n_stations-1]
            station_ids = torch.arange(self.n_stations, device=x.device)
        
        # Ensure station_ids has batch dimension
        if station_ids.dim() == 1:
            station_ids = station_ids.unsqueeze(0).expand(batch_size, -1)
        
        positional_embeddings = self.positional_encoder(
            station_ids, latitude=latitude, longitude=longitude
        )  # [batch, n_stations, positional_embed_dim]

        # 3. Compute directed adjacency matrix from positional embeddings (no self-loops)
        # with geographic bias and wind modulation if provided
        adj = self.compute_adjacency(
            positional_embeddings,
            remove_self_loops=True,
            latitude=latitude,
            longitude=longitude,
            wind_directions=wind_directions,
            top_k=10
        )  # [batch, n_stations, n_stations]
        
        # 4. Apply multiple GCN layers with residual averaging to original temporal embedding
        node_features = temporal_embeddings
        # project original temporal embeddings once for averaging
        temporal_proj = self.temporal_proj(temporal_embeddings)  # [batch, n_stations, gcn_hidden_dim]
        for i, gcn_layer in enumerate(self.gcn_layers):
            # GCN aggregation from neighbors
            node_features = gcn_layer(node_features, adj)  # [batch, n_stations, gcn_hidden_dim]
            node_features = F.relu(node_features)
            node_features = self.dropout(node_features)

            node_features = node_features*0.5 + temporal_proj*0.5

        # 5. Predict next wind speed for all stations (per-node)
        # node_features: [batch, n_stations, gcn_hidden_dim]
        pred_per_node = self.predictor(node_features)  # [batch, n_stations, 1]
        pred_per_node = pred_per_node.squeeze(-1)  # [batch, n_stations]

        return pred_per_node
