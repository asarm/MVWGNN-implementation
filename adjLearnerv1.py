import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class TemporalAdjacencyLearner(nn.Module):
    """
    Learn dynamic graph structure from temporal patterns.
    
    KEY FIX: Now receives temporal_features directly (not raw data)
    No redundant wind speed extraction!
    """
    
    def __init__(self, hidden_dim=64, n_stations=50, embedding_dim=32,
                 sparsity_threshold=0.1, temperature_scale=0.1,
                 sparsify_mode: str = "quantile", nucleus_p: float = 0.9,
                 prob_threshold: float = 0.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        self.sparsity_threshold = sparsity_threshold
        # Sparsification controls
        # modes: 'quantile' (keep top-fraction per row), 'top_p' (nucleus), 'magnitude' (keep >= prob_threshold)
        self.sparsify_mode = sparsify_mode
        self.nucleus_p = nucleus_p
        self.prob_threshold = prob_threshold
        
        # Node embeddings (station-specific properties)
        self.node_embedding_1 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        self.node_embedding_2 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embedding_1)
        nn.init.xavier_uniform_(self.node_embedding_2)
        
        # Project features to embedding space
        self.feature_to_embedding = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
        # Directional influence
        self.directional_influence = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Learnable temperature
        self.temperature_scale = nn.Parameter(
            torch.tensor(temperature_scale)
        )
        
        # Station biases
        self.station_bias_out = nn.Parameter(
            torch.randn(n_stations, 1) * 0.1
        )
        self.station_bias_in = nn.Parameter(
            torch.randn(n_stations, 1) * 0.1
        )
    
    def _compute_base_adjacency(self, station_representations):
        """Compute an asymmetric, learnable base adjacency.

        Uses a learned projection of the input features plus two distinct
        station embeddings (out/in) and station-specific biases to form
        scaled dot-product logits. This avoids over-smoothing when temporal
        features are nearly identical (which made cosine similarity saturate).

        Supports inputs of shape (N, E) or (B, N, E).
        """
        # Project to shared embedding space
        # nn.Linear supports arbitrary leading dims; apply directly.
        h = self.feature_to_embedding(station_representations)  # (..., N, D)

        D = self.embedding_dim
        scale = math.sqrt(D)

        if h.dim() == 2:
            # (N, D)
            # Create distinct query/key representations for asymmetry
            Q = h + self.node_embedding_1  # (N, D)
            K = h + self.node_embedding_2  # (N, D)

            # Scaled dot-product logits + station biases
            logits = (Q @ K.T) / (scale + 1e-8)
            logits = logits + self.station_bias_out + self.station_bias_in.T  # broadcast (N, N)

            # Ensure positive scores to play well with later multiplicative modulation
            adj = F.elu(logits) + 1.0  # >= 0
            return adj
        else:
            # (B, N, D)
            B, N, _ = h.shape
            # Broadcast node embeddings and biases to batch
            emb_out = self.node_embedding_1.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)
            emb_in = self.node_embedding_2.unsqueeze(0).expand(B, -1, -1)   # (B, N, D)

            Q = h + emb_out  # (B, N, D)
            K = h + emb_in   # (B, N, D)

            # Batched matmul for logits
            logits = torch.matmul(Q, K.transpose(-2, -1)) / (scale + 1e-8)  # (B, N, N)

            b_out = self.station_bias_out.squeeze(-1)  # (N,)
            b_in = self.station_bias_in.squeeze(-1)    # (N,)
            logits = logits + b_out.unsqueeze(0).unsqueeze(-1) + b_in.unsqueeze(0).unsqueeze(-2)

            adj = F.elu(logits) + 1.0  # (B, N, N), non-negative
            return adj
    
    def _apply_directional_modulation(self, adjacency, wind_directions, positions):
        """Modulate adjacency by wind direction."""
        # Vectorized implementation supporting adjacency shape (N,N) or (B,N,N)
        device = adjacency.device

        # Ensure wind_directions and positions have batch dim
        if wind_directions.dim() == 1:
            wind_directions = wind_directions.unsqueeze(0)
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)

        # wind_directions: (B, N)
        wind_dir_rad = torch.deg2rad(wind_directions)
        wind_vectors = torch.stack([torch.cos(wind_dir_rad), torch.sin(wind_dir_rad)], dim=-1)
        # (B, N, 2)

        B = wind_vectors.shape[0]
        N = wind_vectors.shape[1]

        # positions: (B, N, 2)
        pos = positions

        # pairwise offsets: (B, N, N, 2)
        offsets = pos[:, :, None, :] - pos[:, None, :, :]
        dists = torch.norm(offsets, dim=-1)  # (B, N, N)

        eps = 1e-6
        spatial_dir = offsets / (dists[..., None] + eps)

        # wind_vectors: (B, N, 1, 2) -> align with spatial_dir
        wind_exp = wind_vectors[:, :, None, :]
        alignment = (wind_exp * spatial_dir).sum(dim=-1)  # (B, N, N)
        # directional input: (B, N, N, 3) -> [wind_x, wind_y, alignment]
        # wind_exp: (B, N, 1, 2) -> wind_x/wind_y: (B, N)
        wind_x = wind_exp[..., 0].squeeze(-1)
        wind_y = wind_exp[..., 1].squeeze(-1)

        # expand to pairwise shape (B, N, N)
        wind_x_mat = wind_x.unsqueeze(2).expand(-1, -1, N)
        wind_y_mat = wind_y.unsqueeze(2).expand(-1, -1, N)

        dir_input = torch.stack([wind_x_mat, wind_y_mat, alignment], dim=-1)
        # Flatten to feed through directional_influence
        flat_input = dir_input.view(-1, 3)
        mod_flat = self.directional_influence(flat_input)
        mod = mod_flat.view(B, N, N)

        # For self-connection, set modulation to 1 (or a fixed value)
        eye = torch.eye(N, device=device).unsqueeze(0)
        mod = mod * (1.0 - eye) + 0.5 * eye

        # If adjacency was non-batched, reduce back
        if adjacency.dim() == 2:
            mod = mod[0]

        modulated_adjacency = adjacency * mod

        return modulated_adjacency
    
    def _sparsify_adjacency(self, adjacency):
        """Sparsify adjacency matrix by keeping only top connections."""
        # Support adjacency: (N,N) or (B,N,N)
        temp = self.temperature_scale.clamp(min=0.01)
        probs = torch.softmax(adjacency / temp, dim=-1)

        mode = getattr(self, 'sparsify_mode', 'quantile')

        def renorm(x):
            s = x.sum(dim=-1, keepdim=True)
            s = torch.clamp(s, min=1e-8)
            return x / s

        if mode == 'quantile':
            if probs.dim() == 2:
                threshold = torch.quantile(probs, 1 - self.sparsity_threshold, dim=-1, keepdim=True)
                mask = (probs >= threshold).float()
                return renorm(probs * mask)
            else:
                threshold = torch.quantile(probs, 1 - self.sparsity_threshold, dim=-1, keepdim=True)
                mask = (probs >= threshold).float()
                return renorm(probs * mask)

        elif mode == 'magnitude':
            thr = float(getattr(self, 'prob_threshold', 0.0))
            if probs.dim() == 2:
                mask = (probs >= thr).float()
                # ensure at least one per row
                ensure = torch.zeros_like(mask)
                argmax = probs.argmax(dim=-1)
                ensure[torch.arange(probs.shape[0]), argmax] = 1.0
                mask = torch.maximum(mask, ensure)
                return renorm(probs * mask)
            else:
                B, N, _ = probs.shape
                mask = (probs >= thr).float()
                ensure = torch.zeros_like(mask)
                argmax = probs.argmax(dim=-1)
                br = torch.arange(B).unsqueeze(1).expand(B, N)
                nr = torch.arange(N).unsqueeze(0).expand(B, N)
                ensure[br, nr, argmax] = 1.0
                mask = torch.maximum(mask, ensure)
                return renorm(probs * mask)

        elif mode == 'top_p':
            p = float(getattr(self, 'nucleus_p', 0.9))
            if probs.dim() == 2:
                # (N, N)
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                shifted = cumsum - sorted_probs
                keep_sorted = (shifted < p)
                # always keep the top-1
                keep_sorted[:, 0] = True
                # scatter back to original order
                mask = torch.zeros_like(probs, dtype=torch.float32)
                mask.scatter_(1, sorted_idx, keep_sorted.float())
                return renorm(probs * mask)
            else:
                # (B, N, N)
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                shifted = cumsum - sorted_probs
                keep_sorted = (shifted < p)
                keep_sorted[..., 0] = True
                mask = torch.zeros_like(probs, dtype=torch.float32)
                mask.scatter_(-1, sorted_idx, keep_sorted.float())
                return renorm(probs * mask)

        else:
            # Fallback to quantile if unknown mode
            if probs.dim() == 2:
                threshold = torch.quantile(probs, 1 - self.sparsity_threshold, dim=-1, keepdim=True)
                mask = (probs >= threshold).float()
                return renorm(probs * mask)
            else:
                threshold = torch.quantile(probs, 1 - self.sparsity_threshold, dim=-1, keepdim=True)
                mask = (probs >= threshold).float()
                return renorm(probs * mask)
    
    def _add_self_loops(self, adjacency, self_loop_weight=0.1):
        """Add self-loops to adjacency matrix."""
        if adjacency.dim() == 2:
            n_stations = adjacency.shape[0]
            device = adjacency.device
            identity = torch.eye(n_stations, device=device)
            adjacency_with_loops = adjacency + self_loop_weight * identity
            return adjacency_with_loops
        else:
            # (B, N, N)
            B, N, _ = adjacency.shape
            device = adjacency.device
            identity = torch.eye(N, device=device).unsqueeze(0)
            adjacency_with_loops = adjacency + self_loop_weight * identity
            return adjacency_with_loops
    
    def _normalize_adjacency(self, adjacency):
        """Normalize adjacency matrix (row-wise stochastic)."""
        row_sums = adjacency.sum(dim=-1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-8)
        normalized = adjacency / row_sums
        return normalized
    
    def forward(self, station_representations, 
                positions=None, wind_directions=None):
        """
        Learn adaptive adjacency matrix.
        
        Args:
            station_representations: (n_stations, hidden_dim) combined repr.
                                     Should be based on spatial/static features.
            positions: (n_stations, 2) lat/lon
            wind_directions: (n_stations,) degrees
        
        Returns:
            adjacency: (n_stations, n_stations) sparse, normalized
        """
        
        # Support batched or non-batched station_representations
        # station_representations: (N, E) or (B, N, E)
        device = station_representations.device
        
        # If station_representations are provided per-sample, they may be batched too.
        adjacency = self._compute_base_adjacency(station_representations)

        # Prepare default wind_directions / positions if not provided
        if wind_directions is None:
            if adjacency.dim() == 2:
                N = adjacency.shape[0]
                wind_directions = torch.ones(N, device=device) * 270.0
            else:
                N = adjacency.shape[1]
                B = adjacency.shape[0]
                wind_directions = torch.ones(B, N, device=device) * 270.0

        if positions is None:
            if adjacency.dim() == 2:
                grid_size = int(math.sqrt(adjacency.shape[0])) + 1
                x_pos = torch.arange(adjacency.shape[0], device=device) % grid_size
                y_pos = torch.arange(adjacency.shape[0], device=device) // grid_size
                positions = torch.stack([x_pos, y_pos], dim=1).float()
                positions = positions / positions.max()
            else:
                B, N, _ = adjacency.shape
                grid_size = int(math.sqrt(N)) + 1
                x_pos = torch.arange(N, device=device) % grid_size
                y_pos = torch.arange(N, device=device) // grid_size
                pos = torch.stack([x_pos, y_pos], dim=1).float()
                pos = pos / pos.max()
                positions = pos.unsqueeze(0).expand(B, -1, -1)

        # Apply directional modulation (vectorized)
        adjacency = self._apply_directional_modulation(adjacency, wind_directions, positions)
        '''
        with torch.no_grad():
            print("[ADJ_DEBUG] after modulation mean/std:", float(adjacency.mean()), float(adjacency.std()))
            print("Edge count before sparsify:", int((adjacency > 0).sum().item()))
        '''
        # Sparsify
        adjacency = self._sparsify_adjacency(adjacency)
        '''
        with torch.no_grad():
            kept = (adjacency > 0).sum(dim=-1)
            kept_mean = float(kept.float().mean())
            kept_min = int(kept.min())
            kept_max = int(kept.max())
            print(f"[ADJ_DEBUG] after sparsify nnz per-row mean/min/max: {kept_mean:.2f}/{kept_min}/{kept_max}")
        '''

        # Add self-loops
        adjacency = self._add_self_loops(adjacency, self_loop_weight=0.1)

        # Normalize
        adjacency = self._normalize_adjacency(adjacency)
        '''
        with torch.no_grad():
            row_sums = adjacency.sum(dim=-1)
            print("[ADJ_DEBUG] final row-sum mean/std:", float(row_sums.mean()), float(row_sums.std()))
        '''
        return adjacency