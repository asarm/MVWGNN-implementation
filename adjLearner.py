import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAdjacencyLearner(nn.Module):
    """
    Learn dynamic graph structure from historical data and learned representations.
    
    Simplified version focusing on essential components.
    """
    
    def __init__(self, hidden_dim=64, n_stations=50, seq_len=168, 
                 embedding_dim=32, sparsity_threshold=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.sparsity_threshold = sparsity_threshold
        
        # Learnable node embeddings (fixed properties of each station)
        self.node_embedding_1 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        self.node_embedding_2 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embedding_1)
        nn.init.xavier_uniform_(self.node_embedding_2)
        
        # Extract temporal dynamics from sequences
        self.temporal_feature_extractor = nn.Sequential(
            nn.Linear(seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        
        # Learn directional influence on adjacency
        self.directional_influence = nn.Sequential(
            nn.Linear(3, 16),  # [dir_cos, dir_sin, alignment_to_neighbor]
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Temperature for sparsification
        self.temperature_scale = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, historical_data, learned_representations, horizon='6h', 
                positions=None, wind_directions=None):
        """
        Learn adjacency matrix for this horizon.
        
        Args:
            historical_data: (n_stations, seq_len, n_features)
            learned_representations: (n_stations, hidden_dim) from combined encoders
            horizon: '3h', '6h', '12h', '24h'
            positions: (n_stations, 2) lat/lon if available
            wind_directions: (n_stations,) wind direction at last timestep
        
        Returns:
            adjacency: (n_stations, n_stations) sparse adjacency matrix
        """
        
        device = historical_data.device
        n_stations = historical_data.shape[0]
        
        # Extract wind speed from historical data (first feature)
        wind_speeds = historical_data[:, :, 0]  # (n_stations, seq_len)
        
        # Extract temporal dynamics
        temporal_dynamics = self.temporal_feature_extractor(wind_speeds)
        # (n_stations, embedding_dim)
        
        # Combine node information
        node_repr = (
            self.node_embedding_1 + 
            temporal_dynamics +
            learned_representations[:, :self.node_embedding_1.shape[1]]
        )
        
        # Create asymmetric base adjacency
        M1 = torch.tanh(self.node_embedding_1)
        M2 = torch.tanh(self.node_embedding_2)
        
        base_adjacency = torch.relu(
            torch.tanh(M1 @ M2.T - M2 @ M1.T)
        )
        # (n_stations, n_stations)
        
        # Modulate by wind direction if available
        if wind_directions is not None and positions is not None:
            directional_modulation = torch.ones(n_stations, n_stations, device=device)
            
            wind_dir_rad = torch.deg2rad(wind_directions)
            wind_dir_vec = torch.stack([
                torch.cos(wind_dir_rad),
                torch.sin(wind_dir_rad)
            ], dim=1)  # (n_stations, 2)
            
            positions_norm = positions / (positions.norm(dim=0, keepdim=True) + 1e-8)
            
            for i in range(n_stations):
                for j in range(n_stations):
                    if i == j:
                        continue
                    
                    spatial_offset = positions[j] - positions[i]
                    dist = torch.norm(spatial_offset)
                    if dist > 1e-6:
                        spatial_dir = spatial_offset / dist
                        alignment = torch.dot(wind_dir_vec[i], spatial_dir)
                    else:
                        alignment = 0.0
                    
                    # Learn how direction affects adjacency
                    dir_input = torch.tensor(
                        [wind_dir_vec[i, 0].item(), wind_dir_vec[i, 1].item(), alignment],
                        device=device
                    )
                    mod_factor = self.directional_influence(dir_input)
                    directional_modulation[i, j] = mod_factor
            
            adjacency = base_adjacency * directional_modulation
        else:
            adjacency = base_adjacency
        
        # Sparsify: keep only top connections per row
        adjacency_sparse = torch.zeros_like(adjacency)
        
        for i in range(n_stations):
            row = adjacency[i, :]
            
            # Temperature-scaled softmax for sharpening
            row_sparse = torch.softmax(
                row / self.temperature_scale.clamp(min=0.01),
                dim=0
            )
            
            # Keep only top-k
            threshold = torch.quantile(row_sparse, 1 - self.sparsity_threshold)
            row_sparse = row_sparse * (row_sparse >= threshold).float()
            
            # Renormalize
            row_sum = row_sparse.sum()
            if row_sum > 0:
                row_sparse = row_sparse / row_sum
            
            adjacency_sparse[i, :] = row_sparse
        
        # Add self-loops
        adjacency_sparse = adjacency_sparse + 0.1 * torch.eye(n_stations, device=device)
        
        # Normalize
        row_sums = adjacency_sparse.sum(dim=1, keepdim=True)
        adjacency_sparse = adjacency_sparse / (row_sums + 1e-8)
        
        return adjacency_sparse
