import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalAdjacencyLearner(nn.Module):
    """
    Learn dynamic graph structure from temporal patterns.
    
    KEY FIX: Now receives temporal_features directly (not raw data)
    No redundant wind speed extraction!
    """
    
    def __init__(self, hidden_dim=64, n_stations=50, embedding_dim=32,
                 sparsity_threshold=0.1, temperature_scale=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        self.sparsity_threshold = sparsity_threshold
        
        # Node embeddings (station-specific properties)
        self.node_embedding_1 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        self.node_embedding_2 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embedding_1)
        nn.init.xavier_uniform_(self.node_embedding_2)
        
        # Project temporal features to embedding space
        self.temporal_to_embedding = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
        # Horizon-specific modulation
        self.horizon_modulation = nn.ModuleDict({
            '3h': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()
            ),
            '6h': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()
            ),
            '12h': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()
            ),
            '24h': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
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
        """Compute asymmetric base adjacency matrix."""
        adj = station_representations @ station_representations.T
        '''
        M1 = torch.tanh(self.node_embedding_1 + temporal_embedding)
        M2 = torch.tanh(self.node_embedding_2)
        
        forward = M1 @ M2.T
        backward = M2 @ M1.T
        
        base_adjacency = torch.relu(torch.tanh(forward - backward))
        
        base_adjacency = (
            base_adjacency + 
            self.station_bias_out +
            self.station_bias_in.T
        )
        '''
        return adj
    
    def _apply_directional_modulation(self, adjacency, wind_directions, positions):
        """Modulate adjacency by wind direction."""
        
        device = adjacency.device
        n_stations = adjacency.shape[0]
        
        wind_dir_rad = torch.deg2rad(wind_directions)
        wind_vectors = torch.stack([
            torch.cos(wind_dir_rad),
            torch.sin(wind_dir_rad)
        ], dim=1)
        
        modulation = torch.ones(n_stations, n_stations, device=device)
        
        for i in range(n_stations):
            for j in range(n_stations):
                if i == j:
                    continue
                
                spatial_offset = positions[j] - positions[i]
                spatial_dist = torch.norm(spatial_offset)
                
                if spatial_dist < 1e-6:
                    mod_factor = torch.tensor(0.5, device=device)
                else:
                    spatial_dir = spatial_offset / spatial_dist
                    alignment = torch.dot(wind_vectors[i], spatial_dir)
                    
                    dir_input = torch.tensor([
                        wind_vectors[i, 0].item(),
                        wind_vectors[i, 1].item(),
                        alignment.item()
                    ], device=device)
                    
                    mod_factor = self.directional_influence(dir_input)
                
                modulation[i, j] = mod_factor
        
        modulated_adjacency = adjacency * modulation
        
        return modulated_adjacency
    
    def _sparsify_adjacency(self, adjacency):
        """Sparsify adjacency matrix by keeping only top connections."""
        
        device = adjacency.device
        n_stations = adjacency.shape[0]
        
        sparse_adjacency = torch.zeros_like(adjacency)
        
        for i in range(n_stations):
            row = adjacency[i, :]
            
            temp = self.temperature_scale.clamp(min=0.01)
            row_softmax = torch.softmax(row / temp, dim=0)
            
            threshold = torch.quantile(
                row_softmax, 
                1 - self.sparsity_threshold
            )
            
            row_sparse = row_softmax * (row_softmax >= threshold).float()
            
            row_sum = row_sparse.sum()
            if row_sum > 1e-8:
                row_sparse = row_sparse / row_sum
            
            sparse_adjacency[i, :] = row_sparse
        
        return sparse_adjacency
    
    def _add_self_loops(self, adjacency, self_loop_weight=0.1):
        """Add self-loops to adjacency matrix."""
        
        n_stations = adjacency.shape[0]
        device = adjacency.device
        
        identity = torch.eye(n_stations, device=device)
        adjacency_with_loops = adjacency + self_loop_weight * identity
        
        return adjacency_with_loops
    
    def _normalize_adjacency(self, adjacency):
        """Normalize adjacency matrix (row-wise stochastic)."""
        
        row_sums = adjacency.sum(dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-8)
        
        normalized = adjacency / row_sums
        
        return normalized
    
    def forward(self, temporal_features, station_representations, 
                positions=None, wind_directions=None):
        """
        Learn adaptive adjacency matrix.
        
        Args:
            temporal_features: (n_stations, hidden_dim) from TemporalEncoder
                             NO REDUNDANT EXTRACTION! Already has wind patterns
            learned_representations: (n_stations, hidden_dim) combined repr
            horizon: '3h', '6h', '12h', '24h'
            positions: (n_stations, 2) lat/lon
            wind_directions: (n_stations,) degrees
        
        Returns:
            adjacency: (n_stations, n_stations) sparse, normalized
        """
        
        device = temporal_features.device
        n_stations = temporal_features.shape[0]
        
        # Project temporal features to embedding space
        # temporal_embedding = self.temporal_to_embedding(temporal_features)
        
        # Compute base asymmetric adjacency
        adjacency = self._compute_base_adjacency(temporal_features+station_representations)

        # Apply directional modulation
        if wind_directions is None:
            wind_directions = torch.ones(n_stations, device=device) * 270.0
        
        if positions is None:
            grid_size = int(math.sqrt(n_stations)) + 1
            x_pos = torch.arange(n_stations, device=device) % grid_size
            y_pos = torch.arange(n_stations, device=device) // grid_size
            positions = torch.stack([x_pos, y_pos], dim=1).float()
            positions = positions / positions.max()
        
        adjacency = self._apply_directional_modulation(
            adjacency, wind_directions, positions
        )
        
        # Sparsify
        adjacency = self._sparsify_adjacency(adjacency)
        
        # Add self-loops
        adjacency = self._add_self_loops(adjacency, self_loop_weight=0.1)
        
        # Normalize
        adjacency = self._normalize_adjacency(adjacency)
        
        return adjacency