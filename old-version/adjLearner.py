import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicAdjacencyLearner(nn.Module):
    """
    V2: Dynamic adjacency that changes based on BOTH spatial and temporal patterns.
    
    Key improvements:
    - Uses temporal features to modulate spatial adjacency
    - Wind patterns affect connectivity strength
    - Learns when to connect distant stations (teleconnections)
    """
    
    def __init__(self, hidden_dim=64, n_stations=50, embedding_dim=32,
                 sparsify_mode: str = "top_p", nucleus_p: float = 0.9,
                 temperature_scale: float = 0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        self.sparsify_mode = sparsify_mode
        self.nucleus_p = nucleus_p
        
        # ========== Spatial Embeddings (static) ==========
        self.spatial_node_embedding = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.spatial_node_embedding)
        
        # ========== Temporal Modulation Network ==========
        # This learns how temporal patterns affect connectivity
        self.temporal_modulator = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh()  # Bounded output for stability
        )
        
        # ========== Directional Influence ==========
        self.directional_influence = nn.Sequential(
                                    nn.Linear(3, 16),
                                    nn.ReLU(),
                                    nn.Linear(16, 8),
                                    nn.ReLU(),
                                    nn.Linear(8, 1)
                                    )
        
        # ========== Edge Weight Predictor ==========
        # Predicts edge strength from combined node features
        self.edge_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )
        
        # Learnable temperature
        self.temperature_scale = nn.Parameter(
            torch.tensor(temperature_scale)
        )
        
        # Station-specific biases
        self.station_bias = nn.Parameter(
            torch.randn(n_stations, 1) * 0.01
        )
    
    def forward(self, spatial_features, temporal_features, 
                positions=None, wind_directions=None):
        """
        Learn dynamic adjacency from spatial and temporal features.
        
        Args:
            spatial_features: (N, H) or (B, N, H) - spatial embeddings
            temporal_features: (N, H) or (B, N, H) - temporal features
            positions: (N, 2) or (B, N, 2) - lat/lon
            wind_directions: (N,) or (B, N) - wind direction in degrees
        
        Returns:
            adjacency: (N, N) or (B, N, N) - dynamic adjacency matrix
        """
        device = spatial_features.device
        is_batched = spatial_features.dim() == 3
        
        if is_batched:
            B, N, H = spatial_features.shape
        else:
            N = spatial_features.shape[0]
        
        # ========== STEP 1: Spatial base adjacency ==========
        # Use static spatial embeddings as base
        if is_batched:
            spatial_emb = self.spatial_node_embedding.unsqueeze(0).expand(B, -1, -1)
        else:
            spatial_emb = self.spatial_node_embedding
        
        # Compute pairwise spatial affinity (using dot product)
        if is_batched:
            # (B, N, D) @ (B, D, N) -> (B, N, N)
            spatial_adj = torch.bmm(spatial_emb, spatial_emb.transpose(1, 2))
        else:
            # (N, D) @ (D, N) -> (N, N)
            spatial_adj = torch.mm(spatial_emb, spatial_emb.T)
        
        # Scale by embedding dimension
        spatial_adj = spatial_adj / math.sqrt(self.embedding_dim)
        
        # ========== STEP 2: Temporal modulation ==========
        # Learn how current temporal state affects connectivity
        temporal_mod = self.temporal_modulator(temporal_features)
        # (B, N, D) or (N, D)
        
        # Compute temporal similarity (stations with similar temporal patterns connect)
        if is_batched:
            temporal_adj = torch.bmm(temporal_mod, temporal_mod.transpose(1, 2))
        else:
            temporal_adj = torch.mm(temporal_mod, temporal_mod.T)
        
        temporal_adj = temporal_adj / math.sqrt(self.embedding_dim)
        
        # ========== STEP 3: Combine spatial and temporal ==========
        # Learnable combination (start with 70% spatial, 30% temporal)
        # Use safer normalization: clamp to prevent magnitude explosion
        spatial_std = spatial_adj.std().clamp(min=0.1)  # Prevent division by very small numbers
        temporal_std = temporal_adj.std().clamp(min=0.1)
        spatial_adj_norm = spatial_adj / spatial_std
        temporal_adj_norm = temporal_adj / temporal_std
        combined_adj = 0.7 * spatial_adj_norm + 0.3 * temporal_adj_norm
        
        # Add station biases
        if is_batched:
            bias = self.station_bias.squeeze(-1).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
            combined_adj = combined_adj + bias  # Broadcast
        else:
            combined_adj = combined_adj + self.station_bias
        
        # ========== STEP 4: Directional modulation (if provided) ==========
        if wind_directions is not None and positions is not None:
            combined_adj = self._apply_directional_modulation(
                combined_adj, wind_directions, positions
            )
        
        # ========== STEP 5: Sparsify and normalize ==========
        adjacency = self._sparsify_adjacency(combined_adj)
        adjacency = self._add_self_loops(adjacency, self_loop_weight=0.1)
        adjacency = self._normalize_adjacency(adjacency)
        
        # Debug statistics (properly handles batched case)
        if torch.rand(1).item() < 0.01:  # Print 1% of the time to avoid spam
            self._print_adjacency_stats(adjacency)
        
        return adjacency
    
    def _print_adjacency_stats(self, adjacency):
        """Print adjacency statistics (handles both batched and non-batched)."""
        with torch.no_grad():
            if adjacency.dim() == 3:
                # Batched: (B, N, N) - analyze first sample only
                B, N, _ = adjacency.shape
                adj = adjacency[0]  # First sample
            else:
                # Non-batched: (N, N)
                N = adjacency.shape[0]
                adj = adjacency

            # Edge statistics
            non_zero = (adj > 1e-8).sum().item()
            
            # Row normalization check
            row_sums = adj.sum(dim=-1)
            
            # Degree distribution
            degrees = (adj > 1e-8).sum(dim=-1).float()
    
    def _apply_directional_modulation(self, adjacency, wind_directions, positions):
        """Modulate adjacency by wind direction (vectorized)."""
        device = adjacency.device
        is_batched = adjacency.dim() == 3
        
        # Ensure inputs have batch dimension
        if wind_directions.dim() == 1:
            wind_directions = wind_directions.unsqueeze(0)
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        B = wind_directions.shape[0]
        N = wind_directions.shape[1]
        
        # Convert wind direction to vectors
        wind_rad = torch.deg2rad(wind_directions)
        wind_vectors = torch.stack([torch.cos(wind_rad), torch.sin(wind_rad)], dim=-1)
        # (B, N, 2)
        
        # Pairwise spatial offsets
        offsets = positions[:, :, None, :] - positions[:, None, :, :]  # (B, N, N, 2)
        dists = torch.norm(offsets, dim=-1)  # (B, N, N)
        
        eps = 1e-6
        spatial_dir = offsets / (dists[..., None] + eps)  # (B, N, N, 2)
        
        # Alignment: how well wind aligns with spatial direction
        wind_exp = wind_vectors[:, :, None, :]  # (B, N, 1, 2)
        alignment = (wind_exp * spatial_dir).sum(dim=-1)  # (B, N, N)
        
        # ========== FIX: Use alignment directly with proper reshape ==========
        # alignment is already [-1, +1] where:
        # +1 = perfect downwind
        # -1 = perfect upwind
        #  0 = perpendicular
        
        # Flatten for computation
        alignment_flat = alignment.view(-1)  # (B*N*N,)
        
        # Apply modulation formula
        mod_flat = 0.6 + 0.4 * alignment_flat  # (B*N*N,)
        
        # Reshape back to (B, N, N)
        mod = mod_flat.view(B, N, N)
        
        # Self-connections always have modulation = 1
        eye = torch.eye(N, device=device).unsqueeze(0)
        mod = mod * (1.0 - eye) + eye
        
        # Apply modulation
        if not is_batched:
            mod = mod[0]
        
        return adjacency * mod
    
    def _sparsify_adjacency(self, adjacency):
        """Sparsify using top-p (nucleus sampling)."""
        temp = self.temperature_scale.clamp(min=0.01)
        probs = torch.softmax(adjacency / temp, dim=-1)
        
        is_batched = probs.dim() == 3
        p = float(self.nucleus_p)
        
        if is_batched:
            # (B, N, N)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            shifted = cumsum - sorted_probs
            keep_sorted = (shifted < p)
            keep_sorted[..., 0] = True  # Always keep top-1
            
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, sorted_idx, keep_sorted.float())
        else:
            # (N, N)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            shifted = cumsum - sorted_probs
            keep_sorted = (shifted < p)
            keep_sorted[:, 0] = True
            
            mask = torch.zeros_like(probs)
            mask.scatter_(1, sorted_idx, keep_sorted.float())
        
        # Renormalize
        sparsified = probs * mask
        row_sums = sparsified.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return sparsified / row_sums
    
    def _add_self_loops(self, adjacency, self_loop_weight=0.1):
        """Add self-loops."""
        if adjacency.dim() == 2:
            N = adjacency.shape[0]
            device = adjacency.device
            identity = torch.eye(N, device=device)
            return adjacency + self_loop_weight * identity
        else:
            B, N, _ = adjacency.shape
            device = adjacency.device
            identity = torch.eye(N, device=device).unsqueeze(0)
            return adjacency + self_loop_weight * identity
    
    def _normalize_adjacency(self, adjacency):
        """Row-wise normalization."""
        row_sums = adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return adjacency / row_sums