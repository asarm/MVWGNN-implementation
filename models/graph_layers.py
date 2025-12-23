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

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network (GCN) Layer.

    Implements the spectral graph convolution from Kipf & Welling (2017).
    Uses symmetric normalization: D^(-1/2) A D^(-1/2)

    Key differences from GraphSAGE:
    - Single transformation matrix (no separate self/neighbor weights)
    - Expects symmetrically normalized adjacency
    - More parameter efficient
    - Better for homophilic graphs (similar neighbors)
    """
    def __init__(self, in_features, out_features, bias=True, add_self_loops=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.add_self_loops = add_self_loops

        # Single weight matrix (parameter efficient)
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

    def normalize_adjacency(self, adj):
        """
        Symmetric normalization: D^(-1/2) A D^(-1/2)

        Args:
            adj: Adjacency matrix [batch, N, N] or [N, N]

        Returns:
            Normalized adjacency matrix
        """
        # Handle both batched and unbatched
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size, N, _ = adj.shape
        device = adj.device

        # Add self-loops if requested
        if self.add_self_loops:
            identity = torch.eye(N, device=device).unsqueeze(0).expand(batch_size, -1, -1)
            adj = adj + identity

        # Compute degree matrix
        # D[i,i] = sum_j A[i,j]
        degree = adj.sum(dim=-1)  # [batch, N]

        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree + 1e-8, -0.5)  # [batch, N]
        degree_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # [batch, N, N]

        # D^(-1/2) A D^(-1/2)
        adj_normalized = torch.bmm(torch.bmm(degree_inv_sqrt, adj), degree_inv_sqrt)

        if squeeze_output:
            adj_normalized = adj_normalized.squeeze(0)

        return adj_normalized

    def forward(self, x, adj):
        """
        Args:
            x: Node features [batch, N, in_features]
            adj: Adjacency matrix [batch, N, N] or [N, N]
                 Can be unnormalized - will be normalized internally

        Returns:
            Node features [batch, N, out_features]
        """
        # Normalize adjacency
        if adj.dim() == 2:
            adj_norm = self.normalize_adjacency(adj)
            adj_norm = adj_norm.unsqueeze(0).expand(x.size(0), -1, -1)
        else:
            adj_norm = self.normalize_adjacency(adj)

        # GCN propagation: A_norm @ X @ W
        # First: aggregate neighbors
        h = torch.bmm(adj_norm, x)  # [batch, N, in_features]

        # Then: linear transformation
        # h: [batch, N, in_features], weight: [in_features, out_features]
        output = torch.matmul(h, self.weight)  # [batch, N, out_features]

        if self.bias is not None:
            output = output + self.bias

        return output

class GATLayer(nn.Module):
    """
    Graph Attention Network (GAT) Layer.

    Implements multi-head attention mechanism from Velickovic et al. (2018).

    Key differences from GraphSAGE/GCN:
    - Learns edge weights dynamically via attention
    - Multi-head attention for diverse relationship modeling
    - Handles heterophily better (dissimilar neighbors)
    - More parameters but more expressive

    Good for:
    - Dynamic graphs with varying importance
    - When you don't trust pre-computed edge weights
    - Feature similarity metapaths (can refine correlations)
    """
    def __init__(self, in_features, out_features, num_heads=4,
                 dropout=0.2, concat_heads=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.dropout = dropout

        # If concatenating heads, each head outputs out_features/num_heads
        # If averaging, each head outputs out_features
        if concat_heads:
            assert out_features % num_heads == 0, \
                f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"
            self.head_dim = out_features // num_heads
        else:
            self.head_dim = out_features

        # Linear transformation for each head
        # [in_features] -> [num_heads, head_dim]
        self.W = nn.Parameter(torch.FloatTensor(in_features, num_heads, self.head_dim))

        # Attention parameters (a in the paper)
        # For each head, we have a 2*head_dim -> 1 attention scoring
        self.attention = nn.Parameter(torch.FloatTensor(num_heads, 2 * self.head_dim, 1))

        if bias and concat_heads:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        elif bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.attention)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        Args:
            x: Node features [batch, N, in_features]
            adj: Adjacency matrix [batch, N, N] or [N, N]
                 Used as attention mask (0 = no connection, >0 = connection)

        Returns:
            Node features [batch, N, out_features]
        """
        batch_size, N, _ = x.shape
        device = x.device

        # Handle static adjacency
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

        # Linear transformation: [batch, N, in_features] @ [in_features, num_heads, head_dim]
        # Result: [batch, N, num_heads, head_dim]
        h = torch.einsum('bni,ihd->bnhd', x, self.W)

        # Compute attention scores
        # For each pair (i, j), compute attention e_ij

        # Repeat h for all pairs
        # h_i: [batch, N, 1, num_heads, head_dim] - source nodes
        # h_j: [batch, 1, N, num_heads, head_dim] - target nodes
        h_i = h.unsqueeze(2).expand(batch_size, N, N, self.num_heads, self.head_dim)
        h_j = h.unsqueeze(1).expand(batch_size, N, N, self.num_heads, self.head_dim)

        # Concatenate [h_i || h_j]: [batch, N, N, num_heads, 2*head_dim]
        h_cat = torch.cat([h_i, h_j], dim=-1)

        # Attention scores: [batch, N, N, num_heads, 2*head_dim] @ [num_heads, 2*head_dim, 1]
        # Result: [batch, N, N, num_heads]
        e = torch.einsum('bnmhd,hdo->bnmh', h_cat, self.attention).squeeze(-1)
        e = self.leaky_relu(e)

        # Mask attention scores based on adjacency
        # Where adj == 0, set attention to -inf (will become 0 after softmax)
        mask = (adj > 0).unsqueeze(-1).expand_as(e)  # [batch, N, N, num_heads]
        e = e.masked_fill(~mask, float('-inf'))

        # Softmax normalization over neighbors (dim=2)
        attention_weights = F.softmax(e, dim=2)  # [batch, N, N, num_heads]

        # Handle isolated nodes (no neighbors): softmax of all -inf gives NaN
        # Replace NaN with 0 (isolated nodes output will be 0)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        attention_weights = self.dropout_layer(attention_weights)

        # Apply attention: weighted sum of neighbor features
        # attention_weights: [batch, N, N, num_heads]
        # h: [batch, N, num_heads, head_dim]
        # We want: for each node i, head h: sum_j (attention[i,j,h] * h[j,h,:])

        # Expand h for neighbors: [batch, 1, N, num_heads, head_dim]
        h_expanded = h.unsqueeze(1).expand(batch_size, N, N, self.num_heads, self.head_dim)

        # Apply attention: [batch, N, N, num_heads, 1] * [batch, N, N, num_heads, head_dim]
        weighted_h = attention_weights.unsqueeze(-1) * h_expanded  # [batch, N, N, num_heads, head_dim]

        # Sum over neighbors: [batch, N, num_heads, head_dim]
        output = weighted_h.sum(dim=2)

        # Concatenate or average heads
        if self.concat_heads:
            # [batch, N, num_heads, head_dim] -> [batch, N, num_heads * head_dim]
            output = output.reshape(batch_size, N, self.num_heads * self.head_dim)
        else:
            # Average across heads
            output = output.mean(dim=2)  # [batch, N, head_dim]

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

    Args:
        hidden_dim: Hidden feature dimension
        k_neighbors: Number of nearest neighbors to connect
        distance_scale: Distance decay scale in km
        layer_type: 'sage', 'gcn', or 'gat'
        num_layers: Number of stacked graph layers
    """
    def __init__(self, hidden_dim, k_neighbors=5, distance_scale=200.0, layer_type='sage', num_layers=1):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.distance_scale = distance_scale
        self.layer_type = layer_type
        self.num_layers = num_layers

        # Stack multiple graph layers
        self.graph_layers = nn.ModuleList()
        for _ in range(num_layers):
            if layer_type == 'sage':
                self.graph_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            elif layer_type == 'gcn':
                self.graph_layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif layer_type == 'gat':
                self.graph_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4, concat_heads=True))
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}. Choose from ['sage', 'gcn', 'gat']")

        # Add unique projection to make this metapath distinctive
        self.metapath_proj = nn.Linear(hidden_dim, hidden_dim)
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

        # Stack multiple graph layers (no residuals - handled at fusion level)
        h_geo = node_features
        for layer in self.graph_layers:
            h_geo = layer(h_geo, A_geo)
            h_geo = F.relu(h_geo)

        # Apply metapath-specific projection
        h_geo = self.metapath_proj(h_geo)

        return h_geo

class FeatureSimilarityMetapathV2(nn.Module):
    def __init__(self, hidden_dim, feature_name, top_k=10, lookback_hours=24, layer_type='sage', num_layers=1, n_stations=None):
        super().__init__()
        self.feature_name = feature_name
        self.top_k = top_k
        self.lookback_hours = lookback_hours
        self.layer_type = layer_type
        self.num_layers = num_layers

        if n_stations is not None:
            emb_dim = hidden_dim // 4
            self.node_emb1 = nn.Parameter(torch.randn(n_stations, emb_dim) * 0.01)
            self.node_emb2 = nn.Parameter(torch.randn(n_stations, emb_dim) * 0.01)
            # Blending weight: computed vs learned adjacency
            self.adj_blend = nn.Parameter(torch.tensor(0.7))  # Başta computed'a ağırlık ver
            self.use_learnable = True
        else:
            self.use_learnable = False

        # Stack multiple graph layers
        self.graph_layers = nn.ModuleList()
        for _ in range(num_layers):
            if layer_type == 'sage':
                self.graph_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            elif layer_type == 'gcn':
                self.graph_layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif layer_type == 'gat':
                self.graph_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4, concat_heads=True))
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}. Choose from ['sage', 'gcn', 'gat']")

        # Add unique projection to make this metapath distinctive
        self.metapath_proj = nn.Linear(hidden_dim, hidden_dim)

    def construct_similarity_adjacency(self, feature_values):
        """
        Construct adjacency based on feature similarity (correlation).
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

    def get_learned_adjacency(self):
        """Tamamen öğrenilebilir adjacency matrix"""
        # E1 @ E2^T -> her istasyon çifti için bir skor
        raw_adj = self.node_emb1 @ self.node_emb2.T  # [N, N]
        # Row-wise softmax: her satır için olasılık dağılımı
        learned_adj = F.softmax(raw_adj, dim=-1)
        return learned_adj  # [N, N]

    def blend_adjacencies(self, A_computed):
        """Computed ve learned adjacency'yi birleştir"""
        if not self.use_learnable:
            return A_computed
            
        A_learned = self.get_learned_adjacency()  # [N, N]
        
        # Batch boyutuna genişlet
        batch_size = A_computed.size(0)
        A_learned = A_learned.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Sigmoid ile [0,1] arasına sıkıştır
        alpha = torch.sigmoid(self.adj_blend)
        
        # Blend: alpha * computed + (1-alpha) * learned
        A_final = alpha * A_computed + (1 - alpha) * A_learned
        
        # Re-normalize rows
        row_sum = A_final.sum(dim=-1, keepdim=True)
        A_final = A_final / (row_sum + 1e-8)
        
        return A_final
    
    def forward(self, node_features, feature_values):
        # 1. Correlation-based adjacency hesapla
        A_computed = self.construct_similarity_adjacency(feature_values)
        
        # 2. Learnable component ile birleştir (YENİ)
        A_final = self.blend_adjacencies(A_computed)
        
        # 3. Graph convolution (mevcut kod)
        h_feat = node_features
        for i, layer in enumerate(self.graph_layers):
            h_prev = h_feat
            h_feat = layer(h_feat, A_final)  # A_computed yerine A_final
            h_feat = F.relu(h_feat)
            if i > 0:
                h_feat = h_feat + h_prev
        
        h_feat = self.metapath_proj(h_feat)
        return h_feat

class FeatureSimilarityMetapath(nn.Module):
    """
    Dynamic feature similarity metapath.
    Connects stations with similar feature values (e.g., temperature, wind speed).

    Args:
        hidden_dim: Hidden feature dimension
        feature_name: Name of the feature for similarity computation
        top_k: Number of most similar stations to connect
        lookback_hours: Hours of history to use for correlation
        layer_type: 'sage', 'gcn', or 'gat'
        num_layers: Number of stacked graph layers
    """
    def __init__(self, hidden_dim, feature_name, top_k=10, lookback_hours=24, layer_type='sage', num_layers=1):
        super().__init__()
        self.feature_name = feature_name
        self.top_k = top_k
        self.lookback_hours = lookback_hours
        self.layer_type = layer_type
        self.num_layers = num_layers

        # Stack multiple graph layers
        self.graph_layers = nn.ModuleList()
        for _ in range(num_layers):
            if layer_type == 'sage':
                self.graph_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            elif layer_type == 'gcn':
                self.graph_layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif layer_type == 'gat':
                self.graph_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4, concat_heads=True))
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}. Choose from ['sage', 'gcn', 'gat']")

        # Add unique projection to make this metapath distinctive
        self.metapath_proj = nn.Linear(hidden_dim, hidden_dim)

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

        # Stack multiple graph layers (no residuals - handled at fusion level)
        h_feat = node_features
        for layer in self.graph_layers:
            h_feat = layer(h_feat, A_feat)
            h_feat = F.relu(h_feat)

        # Apply metapath-specific projection
        h_feat = self.metapath_proj(h_feat)

        return h_feat


class CrossFeatureSpatialMetapath1(nn.Module):
    """
    A istasyonundaki feature_i'nin B istasyonundaki feature_j'ye etkisini modeller.
    Örnek: pressure(A) → wind_speed(B)
    """
    def __init__(self, hidden_dim, source_feature, target_feature, 
                 n_stations, top_k=5, layer_type='sage'):
        super().__init__()
        self.source_feature = source_feature  # 'pressure'
        self.target_feature = target_feature  # 'wind_speed'
        
        # Cross-feature ilişki öğrenmek için
        self.cross_feature_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable spatial adjacency for this specific cross-feature relation
        self.node_emb_source = nn.Parameter(torch.randn(n_stations, hidden_dim // 4) * 0.01)
        self.node_emb_target = nn.Parameter(torch.randn(n_stations, hidden_dim // 4) * 0.01)
        
        # Graph layer
        if layer_type == 'sage':
            self.graph_layer = GraphSAGELayer(hidden_dim, hidden_dim)
        
        self.top_k = top_k
        self.n_station = n_stations
        self.lag_weights = nn.Parameter(
            torch.zeros(n_stations, n_stations, 10)  # max lag 10
        )
        
    def construct_cross_feature_adjacency(self, source_values, target_values):
        """
        Source feature ile target feature arasındaki lagged correlation'a dayalı adjacency.
        
        Args:
            source_values: [batch, N, lookback] - örn: pressure
            target_values: [batch, N, lookback] - örn: wind_speed
        """
        batch_size, N, lookback = source_values.shape
        device = source_values.device
        
        A_cross = torch.zeros(batch_size, N, N, device=device)
        
        for b in range(batch_size):
            # Her source-target istasyon çifti için cross-correlation
            for lag in range(1, min(7, lookback)):  # 1-6 saat lag
                source_lagged = source_values[b, :, :-lag]  # [N, lookback-lag]
                target_current = target_values[b, :, lag:]   # [N, lookback-lag]
                
                # Source i, Target j için correlation
                # source_lagged[i] ile target_current[j] arasındaki ilişki
                src_norm = (source_lagged - source_lagged.mean(dim=1, keepdim=True)) / (source_lagged.std(dim=1, keepdim=True) + 1e-8)
                tgt_norm = (target_current - target_current.mean(dim=1, keepdim=True)) / (target_current.std(dim=1, keepdim=True) + 1e-8)
                
                # Cross-correlation matrix [N_source, N_target]
                cross_corr = torch.mm(src_norm, tgt_norm.T) / (lookback - lag)
                A_cross[b] += torch.clamp(cross_corr, min=0) / 6  # Positive correlations, averaged
        
        # Learnable refinement
        learned_adj = F.softmax(self.node_emb_source @ self.node_emb_target.T, dim=-1)
        alpha = 0.7
        A_cross = alpha * A_cross + (1 - alpha) * learned_adj.unsqueeze(0)
        
        # Top-k sparsification
        if self.top_k < N:
            for b in range(batch_size):
                for i in range(N):
                    _, topk_idx = torch.topk(A_cross[b, i], self.top_k)
                    mask = torch.zeros(N, device=device)
                    mask[topk_idx] = 1
                    A_cross[b, i] = A_cross[b, i] * mask
        
        # Row normalize
        row_sum = A_cross.sum(dim=-1, keepdim=True)
        A_cross = A_cross / (row_sum + 1e-8)
        
        return A_cross
    
    def forward(self, source_node_features, target_node_features, 
                source_values, target_values):
        """
        Args:
            source_node_features: [batch, N, hidden_dim] - pressure encoding
            target_node_features: [batch, N, hidden_dim] - wind_speed encoding  
            source_values: [batch, N, lookback] - raw pressure values
            target_values: [batch, N, lookback] - raw wind_speed values
        """
        # Cross-feature adjacency
        A_cross = self.construct_cross_feature_adjacency(source_values, target_values)
        
        # Source features'ı cross-feature space'e project et
        h_source_proj = self.cross_feature_proj(source_node_features)

        # Graph convolution: source'dan target'a bilgi akışı
        # A_cross[i,j] = "source i'nin target j'yi ne kadar etkilediği"
        h_cross = self.graph_layer(h_source_proj, A_cross)

        # No residual - handled at fusion level
        return F.relu(h_cross)

class CrossFeatureSpatialMetapath2(nn.Module):
    """
    SIMPLIFIED: Cross-feature spatial metapath with global lag weights.
    Models how source_feature at station A affects target_feature at station B.

    Key simplification: Uses global lag weights (shared across all station pairs)
    instead of learning pairwise lag attention, reducing parameters and overfitting risk.

    Example: pressure(A) → wind_speed(B) with global lag preference
    """
    def __init__(self, hidden_dim, source_feature, target_feature,
                 n_stations, max_lag=6, top_k=5, layer_type='sage'):
        super().__init__()
        self.source_feature = source_feature
        self.target_feature = target_feature
        self.max_lag = max_lag
        self.top_k = top_k
        self.n_stations = n_stations

        # Cross-feature projection
        self.cross_feature_proj = nn.Linear(hidden_dim, hidden_dim)

        # ============================================================
        # SIMPLIFIED: Global lag weights (shared across all pairs)
        # ============================================================
        self.lag_weights = nn.Parameter(torch.randn(max_lag))

        # Graph layer
        if layer_type == 'sage':
            self.graph_layer = GraphSAGELayer(hidden_dim, hidden_dim)
        elif layer_type == 'gcn':
            self.graph_layer = GCNLayer(hidden_dim, hidden_dim)
        elif layer_type == 'gat':
            self.graph_layer = GATLayer(hidden_dim, hidden_dim, num_heads=4, concat_heads=True)
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def get_lag_attention(self):
        """
        SIMPLIFIED: Global lag attention (same weights for all station pairs).

        Returns:
            lag_attention: [max_lag] - softmax normalized global weights
        """
        return F.softmax(self.lag_weights, dim=0)
        
    def construct_cross_feature_adjacency(self, source_values, target_values):
        """
        SIMPLIFIED: Compute cross-feature adjacency using global lag weights.

        Args:
            source_values: [batch, N, lookback] - source feature (e.g., pressure)
            target_values: [batch, N, lookback] - target feature (e.g., wind_speed)

        Returns:
            A_cross: [batch, N, N] - weighted adjacency
            lag_attention: [max_lag] - global lag importance weights
        """
        batch_size, N, lookback = source_values.shape
        device = source_values.device

        # ============================================================
        # Compute correlation matrix for each lag
        # ============================================================
        lag_correlations = []

        for tau in range(1, self.max_lag + 1):
            if tau >= lookback:
                lag_correlations.append(torch.zeros(batch_size, N, N, device=device))
                continue

            # Lagged values
            source_lagged = source_values[:, :, :-tau]  # [batch, N, lookback-tau]
            target_current = target_values[:, :, tau:]   # [batch, N, lookback-tau]

            A_tau = torch.zeros(batch_size, N, N, device=device)

            for b in range(batch_size):
                src = source_lagged[b]  # [N, T]
                tgt = target_current[b]  # [N, T]

                # Z-score normalization
                src_norm = (src - src.mean(dim=1, keepdim=True)) / (src.std(dim=1, keepdim=True) + 1e-8)
                tgt_norm = (tgt - tgt.mean(dim=1, keepdim=True)) / (tgt.std(dim=1, keepdim=True) + 1e-8)

                # Cross-correlation: source[i] -> target[j]
                cross_corr = torch.mm(src_norm, tgt_norm.T) / (lookback - tau)
                A_tau[b] = torch.clamp(cross_corr, min=0)  # Only positive correlations

            lag_correlations.append(A_tau)

        # Stack: [batch, N, N, max_lag]
        lag_correlations = torch.stack(lag_correlations, dim=-1)

        # ============================================================
        # Global lag attention (shared across all pairs)
        # ============================================================
        lag_attention = self.get_lag_attention()  # [max_lag]

        # Weighted sum over lags using global weights
        # lag_correlations: [batch, N, N, max_lag]
        # lag_attention: [max_lag]
        A_cross = torch.sum(
            lag_correlations * lag_attention.view(1, 1, 1, self.max_lag),
            dim=-1
        )  # [batch, N, N]

        # ============================================================
        # Top-k sparsification (keep only top-k neighbors per node)
        # ============================================================
        if self.top_k < N:
            for b in range(batch_size):
                for i in range(N):
                    _, topk_idx = torch.topk(A_cross[b, i], min(self.top_k, N))
                    mask = torch.zeros(N, device=device)
                    mask[topk_idx] = 1
                    A_cross[b, i] = A_cross[b, i] * mask

        # Row normalize
        row_sum = A_cross.sum(dim=-1, keepdim=True)
        A_cross = A_cross / (row_sum + 1e-8)

        return A_cross, lag_attention
    
    def forward(self, source_node_features, target_node_features,
                source_values, target_values):
        """
        Args:
            source_node_features: [batch, N, hidden_dim] - source feature encoding
            target_node_features: [batch, N, hidden_dim] - target feature encoding
            source_values: [batch, N, lookback] - raw source values
            target_values: [batch, N, lookback] - raw target values

        Returns:
            h_out: [batch, N, hidden_dim] - output node features
            lag_attention: [max_lag] - global lag weights for interpretability
        """
        # Cross-feature adjacency with global lag weights
        A_cross, lag_attention = self.construct_cross_feature_adjacency(
            source_values, target_values
        )

        # Project source features to cross-feature space
        h_source_proj = self.cross_feature_proj(source_node_features)

        # Graph convolution: source → target information flow
        h_cross = self.graph_layer(h_source_proj, A_cross)
        h_cross = F.relu(h_cross)

        # Output projection (no residual - handled at fusion level)
        h_out = self.output_proj(h_cross)

        return F.relu(h_out), lag_attention

    def get_dominant_lag(self):
        """
        SIMPLIFIED: Returns the single most dominant lag (global).

        Returns:
            dominant_lag: scalar int - the most important lag (1-indexed)
            lag_attention: [max_lag] - full attention distribution
        """
        with torch.no_grad():
            lag_attention = self.get_lag_attention()  # [max_lag]
            dominant_lag = torch.argmax(lag_attention).item() + 1  # 1-indexed

        return dominant_lag, lag_attention

class MultiTemporalWindMetapath(nn.Module):
    """
    Multi-temporal wind-informed metapath.
    Captures wind propagation effects across multiple time lags.

    Args:
        hidden_dim: Hidden feature dimension
        n_stations: Number of weather stations
        max_lag: Maximum number of temporal lags to consider
        distance_scale: Distance decay scale in km
        layer_type: 'sage', 'gcn', or 'gat'
    """
    def __init__(self, hidden_dim, n_stations, max_lag=12, distance_scale=500.0, layer_type='sage'):
        super().__init__()
        self.max_lag = max_lag
        self.distance_scale = distance_scale
        self.layer_type = layer_type
        self.n_stations = n_stations

        # Learnable lag attention weights
        self.lag_attention_logits = nn.Parameter(torch.randn(max_lag))

        # Graph layers for each lag (separate layers per lag)
        self.graph_layers = nn.ModuleList()
        for _ in range(max_lag):
            if layer_type == 'sage':
                self.graph_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            elif layer_type == 'gcn':
                self.graph_layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif layer_type == 'gat':
                self.graph_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4, concat_heads=True))
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}. Choose from ['sage', 'gcn', 'gat']")

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable alignment threshold
        # self.alignment_threshold = nn.Parameter(torch.tensor(0.5))

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
        wind_speed_t = wind_speeds[:, :, time_idx]  # [batch, N] (NORMALIZED)
        
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

            # Edge weight: alignment × distance × lag × terrain
            edge_weight = alignment_positive * distance_decay * lag_decay
            
            # keep edges where alignment > thtreshold
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
            
            # Graph layer aggregation
            h_transformed = self.graph_layers[tau](h_tau_source, A_tau)
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
       
class MultiTemporalWindMetapath2(nn.Module):
    """
    Multi-temporal wind-informed metapath with pairwise lag learning.
    Her istasyon çifti (i,j) için optimal lag'i öğrenir.
    """
    def __init__(self, hidden_dim, n_stations, max_lag=12, distance_scale=500.0, layer_type='sage'):
        super().__init__()
        self.max_lag = max_lag
        self.distance_scale = distance_scale
        self.layer_type = layer_type
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim

        # ============================================================
        # Pairwise lag attention
        # ============================================================
        self.source_lag_emb = nn.Parameter(torch.randn(n_stations, hidden_dim // 8) * 0.01)
        self.target_lag_emb = nn.Parameter(torch.randn(n_stations, hidden_dim // 8) * 0.01)
        self.lag_query = nn.Parameter(torch.randn(max_lag, hidden_dim // 4) * 0.01)

        # Graph layers for each lag
        self.graph_layers = nn.ModuleList()
        for _ in range(max_lag):
            if layer_type == 'sage':
                self.graph_layers.append(GraphSAGELayer(hidden_dim, hidden_dim))
            elif layer_type == 'gcn':
                self.graph_layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif layer_type == 'gat':
                self.graph_layers.append(GATLayer(hidden_dim, hidden_dim, num_heads=4, concat_heads=True))
            else:
                raise ValueError(f"Unknown layer_type: {layer_type}")

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable alignment threshold
        self.alignment_threshold = nn.Parameter(torch.tensor(0.5))

    def compute_pairwise_lag_attention(self):
        """
        Her istasyon çifti (i,j) için lag attention weights hesapla.
        
        Returns:
            lag_attention: [N, N, max_lag] - softmax normalized per pair
        """
        N = self.n_stations
        
        src_expanded = self.source_lag_emb.unsqueeze(1).expand(N, N, -1)  # [N, N, d]
        tgt_expanded = self.target_lag_emb.unsqueeze(0).expand(N, N, -1)  # [N, N, d]
        pair_emb = torch.cat([src_expanded, tgt_expanded], dim=-1)  # [N, N, 2d]
        
        lag_scores = torch.einsum('ijd,ld->ijl', pair_emb, self.lag_query)  # [N, N, max_lag]
        lag_attention = F.softmax(lag_scores, dim=-1)  # [N, N, max_lag]
        
        return lag_attention

    def construct_lag_adjacency(self, wind_dirs, wind_speeds, lat, lon, tau):
        """
        Construct adjacency matrix for specific temporal lag tau.
        """
        batch_size, N, lookback = wind_dirs.shape
        device = wind_dirs.device
        
        if tau >= lookback:
            return torch.zeros(batch_size, N, N, device=device)
        
        time_idx = lookback - tau - 1
        wind_dir_t = wind_dirs[:, :, time_idx]
        
        # Pairwise distances and bearings
        distances = haversine_distance(lat, lon, lat, lon)
        bearings = calculate_bearing(lat, lon, lat, lon)
        
        A_tau = torch.zeros(batch_size, N, N, device=device)
        
        for b in range(batch_size):
            wind_i = wind_dir_t[b:b+1, :].T  # [N, 1]
            
            # Alignment: cos(wind_direction - bearing)
            alignment = torch.cos(torch.deg2rad(wind_i - bearings))
            alignment_positive = torch.clamp(alignment, min=0)
            
            # Distance decay
            distance_decay = torch.exp(-distances / self.distance_scale)
            
            # Lag decay
            lag_decay = torch.exp(torch.tensor(-tau / 6.0, device=device))
            
            # Edge weight
            edge_weight = alignment_positive * distance_decay * lag_decay
            
            # Threshold
            threshold = torch.sigmoid(self.alignment_threshold)
            edge_weight = torch.where(
                alignment_positive > threshold,
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
            lag_attention: [N, N, max_lag] - pairwise lag importance
        """
        batch_size, N, lookback, hidden_dim = node_features.shape
        device = node_features.device
        
        # Current state features
        h_current = node_features[:, :, -1, :]  # [batch, N, hidden_dim]
        
        # ============================================================
        # Her lag için adjacency ve transformed features hesapla
        # ============================================================
        lag_adjacencies = []
        lag_features = []
        
        for tau in range(self.max_lag):
            # Adjacency for this lag
            A_tau = self.construct_lag_adjacency(wind_dirs, wind_speeds, lat, lon, tau)
            lag_adjacencies.append(A_tau)
            
            # Source features from (t - tau)
            if tau < lookback:
                h_tau_source = node_features[:, :, lookback - tau - 1, :]
            else:
                h_tau_source = h_current
            
            # Graph layer for this lag
            h_transformed = self.graph_layers[tau](h_tau_source, A_tau)
            h_transformed = F.relu(h_transformed)
            lag_features.append(h_transformed)
        
        # Stack features: [batch, N, max_lag, hidden_dim]
        h_stacked = torch.stack(lag_features, dim=2)
        
        # Stack adjacencies: [batch, N, N, max_lag]
        A_stacked = torch.stack(lag_adjacencies, dim=-1)
        
        # ============================================================
        # Pairwise lag attention
        # ============================================================
        lag_attention = self.compute_pairwise_lag_attention()  # [N, N, max_lag]
        
        # ============================================================
        # İki aşamalı aggregation:
        # 1) Her lag için spatial aggregation (graph conv zaten yaptı)
        # 2) Lag'ler arası aggregation (pairwise weights ile)
        # ============================================================
        
        # Basit ve etkili yaklaşım:
        # Her hedef node için, incoming edge'lerin lag importance'ını kullan
        
        # Weighted adjacency: her (i,j) edge'ini optimal lag'e göre ağırlıklandır
        # A_stacked: [batch, N, N, max_lag]
        # lag_attention: [N, N, max_lag]
        A_weighted = torch.sum(
            A_stacked * lag_attention.unsqueeze(0),
            dim=-1
        )  # [batch, N, N]
        
        # Re-normalize
        row_sum = A_weighted.sum(dim=-1, keepdim=True)
        A_weighted = A_weighted / (row_sum + 1e-8)
        
        # Her node için lag-weighted feature aggregation
        # Diagonal attention kullan (her node'un kendi lag preference'ı)
        # lag_attention diagonal: [N, max_lag]
        diag_lag_attention = torch.diagonal(lag_attention, dim1=0, dim2=1).T  # [N, max_lag]
        
        # Weighted sum over lags per node
        # h_stacked: [batch, N, max_lag, hidden_dim]
        # diag_lag_attention: [N, max_lag]
        h_aggregated = torch.einsum(
            'bnlh,nl->bnh',
            h_stacked,
            diag_lag_attention
        )  # [batch, N, hidden_dim]
        
        # Final projection
        h_out = self.output_proj(h_aggregated)
        h_out = F.relu(h_out)
        
        return h_out, lag_attention

    def get_dominant_lags(self):
        """
        Her istasyon çifti için en dominant lag'i döndür.
        """
        with torch.no_grad():
            lag_attention = self.compute_pairwise_lag_attention()
            dominant_lags = torch.argmax(lag_attention, dim=-1) + 1
        
        return dominant_lags, lag_attention

class MetapathFusion(nn.Module):
    def __init__(self, hidden_dim, n_metapaths):
        super().__init__()

        # Semantic-level attention (learn importance of each metapath)
        # Use a simple attention mechanism based on metapath embeddings
        self.attention_query = nn.Parameter(torch.randn(hidden_dim) * 0.01)  # Smaller init
        self.attention_transform = nn.Linear(hidden_dim, hidden_dim)

        self.temperature = torch.tensor(1.0)

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

        batch_size, N, n_metapaths, hidden_dim = h_stacked.shape

        # Compute attention scores for each metapath
        # Transform each metapath embedding
        h_transformed = self.attention_transform(h_stacked)  # [batch, N, n_metapaths, hidden_dim]

        # Compute attention scores by dot product with learnable query vector
        # attention_query: [hidden_dim]
        # h_transformed: [batch, N, n_metapaths, hidden_dim]
        # We want to compute attention score for each metapath
        attention_scores = torch.einsum('bnmd,d->bnm', h_transformed, self.attention_query)
        # Result: [batch, N, n_metapaths]

        # Apply temperature scaling (allows model to control sharpness)
        attention_scores = attention_scores / torch.clamp(self.temperature, min=0.1)

        # Softmax to get attention weights (per sample and station)
        attention_weights_per_sample = F.softmax(attention_scores, dim=-1)  # [batch, N, n_metapaths]

        # Average attention weights across all samples and stations for interpretability
        # This gives us the global learned importance of each metapath
        attention_weights_global = attention_weights_per_sample.mean(dim=0).mean(dim=0)  # [n_metapaths]

        # Weighted sum across metapaths using per-sample attention
        # [batch, N, n_metapaths, 1] * [batch, N, n_metapaths, hidden_dim]
        h_fused = torch.sum(
            attention_weights_per_sample.unsqueeze(-1) * h_stacked,
            dim=2
        )  # [batch, N, hidden_dim]

        # Final projection
        h_fused = self.output_proj(h_fused)
        h_fused = F.relu(h_fused)
        
        # Return attention weights as dict for interpretability
        attention_dict = {name: float(attention_weights_global[i].item())
                          for i, name in enumerate(metapath_names)}

        return h_fused, attention_dict