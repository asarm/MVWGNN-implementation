import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAdjacencyLearner(nn.Module):
    """
    Learn dynamic, time-dependent spatial relationships from historical data.
    
    Philosophy:
    - Adjacency is NOT static (like geographic distance)
    - Adjacency is NOT hard-coded (like pressure→wind physics)
    - Adjacency is LEARNED from what actually happens in the data
    
    Key features:
    1. Learns from full historical sequences (not single timesteps)
    2. Generates DIFFERENT adjacency matrices for different horizons
    3. Incorporates directional information as INDUCTIVE BIAS (not hard constraint)
    4. Discovers what relationships matter for forecasting
    """
    
    def __init__(
        self,
        hidden_dim=64,
        n_stations=50,
        seq_len=168,
        embedding_dim=32,
        n_horizons=4,
        sparsity_threshold=0.1,  # Keep only top connections
        use_directional_bias=True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.n_horizons = n_horizons
        self.sparsity_threshold = sparsity_threshold
        self.use_directional_bias = use_directional_bias
        
        # ========== FOUNDATION: Learnable Node Embeddings ==========
        # Each station has a FIXED unique embedding that represents its inherent properties
        # (geographic location, terrain, urban/rural nature, etc.)
        
        self.node_embedding_1 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        self.node_embedding_2 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        
        nn.init.xavier_uniform_(self.node_embedding_1)
        nn.init.xavier_uniform_(self.node_embedding_2)
        
        
        # ========== TEMPORAL FEATURE EXTRACTION ==========
        # Extract what's happening at each station over the sequence
        # These capture local dynamics that influence spatial correlations
        
        self.temporal_feature_extractor = nn.Sequential(
            nn.Linear(seq_len, 64),  # Compress temporal sequence
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        
        
        # ========== DIRECTIONAL INFLUENCE MODULATOR ==========
        # If use_directional_bias=True, learn how wind direction affects adjacency
        # NOT forcing pressure→wind, but providing INDUCTIVE BIAS
        
        if self.use_directional_bias:
            self.directional_influence = nn.Sequential(
                nn.Linear(4, 32),  # [dir_cos, dir_sin, spatial_alignment, temporal_alignment]
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # Output: [0, 1] modulation factor
            )
        
        
        # ========== PRESSURE GRADIENT INFLUENCE ==========
        # Learn how pressure patterns affect correlations
        # Provide inductive bias without hard constraint
        
        self.pressure_influence = nn.Sequential(
            nn.Linear(2, 32),  # [pressure_i, pressure_j] or pressure difference
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # Output: [-1, 1]
        )
        
        
        # ========== TIME-LAG DETECTION ==========
        # Discover if correlations exist with time delays
        # (wind at station i might influence station j 2 hours later)
        
        self.lag_detector = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),  # Compare features of two stations
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),  # Predict lag: 0h, 1h, 2h, 3h, 4h
            nn.Softmax(dim=-1)
        )
        
        
        # ========== HORIZON-SPECIFIC GRAPH STRUCTURE ==========
        # Different adjacency matrices for 3h, 6h, 12h, 24h forecasts
        # Because physical processes operate at different timescales
        
        self.horizon_specific_projection = nn.ModuleDict({
            '3h': nn.Linear(embedding_dim * 2, embedding_dim),
            '6h': nn.Linear(embedding_dim * 2, embedding_dim),
            '12h': nn.Linear(embedding_dim * 2, embedding_dim),
            '24h': nn.Linear(embedding_dim * 2, embedding_dim)
        })
        
        
        # ========== SPARSIFICATION ==========
        # Make the graph sparse (keep only important connections)
        # This improves interpretability and reduces computation
        
        self.temperature_scale = nn.Parameter(
            torch.tensor(0.1)  # Controls how concentrated adjacency is
        )
        
    
    def forward(
        self,
        historical_data,  # (n_stations, seq_len, n_features): wind_speed, dir_cos, dir_sin, pressure, temp, humidity
        learned_representations,  # (n_stations, hidden_dim): from TemporalEncoder
        horizon='6h',
        positions=None  # (n_stations, 2): lat, lon for spatial alignment
    ):
        """
        Learn the adjacency matrix for a specific forecast horizon.
        
        Args:
            historical_data: Full historical sequences for all stations
            learned_representations: Temporal features from TemporalEncoder
            horizon: '3h', '6h', '12h', or '24h'
            positions: Geographic positions of stations
        
        Returns:
            adjacency: (n_stations, n_stations) sparse adjacency matrix
        """
        
        n_stations = historical_data.shape[0]
        device = historical_data.device
        
        # ========== STEP 1: Extract local temporal dynamics ==========
        # For each station, what is the pattern in the historical data?
        
        # Take wind speed for temporal feature extraction
        wind_speeds = historical_data[:, :, 0]  # (n_stations, seq_len)
        
        temporal_dynamics = self.temporal_feature_extractor(wind_speeds)
        # Shape: (n_stations, embedding_dim)
        # This captures things like: "is this station volatile or stable?"
        
        
        # ========== STEP 2: Combine node information ==========
        # Each station is characterized by:
        # - Its fixed geographic properties (node embeddings)
        # - Its current temporal dynamics (from historical data)
        # - Its learned spatio-temporal patterns (from TemporalEncoder)
        
        node_repr_combined = (
            self.node_embedding_1 +  # Geographic/fixed properties
            temporal_dynamics +      # What's happening now
            learned_representations  # Full spatio-temporal context
        )
        # Shape: (n_stations, embedding_dim)
        
        
        # ========== STEP 3: Compute base adjacency from node similarity ==========
        # Two stations are similar if they have similar combined representations
        # This is DATA-DRIVEN: similarity emerges from what the model learns
        
        M1 = torch.tanh(self.node_embedding_1)
        M2 = torch.tanh(self.node_embedding_2)
        
        # Create asymmetric adjacency (directional influence)
        # A_ij ≠ A_ji because wind at i might influence j differently than vice versa
        
        base_adjacency = torch.relu(
            torch.tanh(
                M1 @ M2.T - M2 @ M1.T  # Asymmetric operation
            )
        )
        # Shape: (n_stations, n_stations)
        # Properties:
        # - Values in [0, 1] after ReLU
        # - Diagonal is zero (no self-loops yet)
        # - Asymmetric (directional)
        
        
        # ========== STEP 4: Modulate by directional information ==========
        # Inductive bias: wind direction affects how influences propagate
        # NOT forcing physics, just saying "direction matters"
        
        if self.use_directional_bias and positions is not None:
            direction_data = historical_data[:, -1, 1:3]  # [cos(dir), sin(dir)] at last timestep
            # Shape: (n_stations, 2)
            
            directional_modulation = torch.ones(
                n_stations, n_stations, device=device
            )
            
            for i in range(n_stations):
                for j in range(n_stations):
                    if i == j:
                        continue
                    
                    # Direction at station i
                    wind_dir_i = direction_data[i]  # [cos, sin]
                    
                    # Vector from station i to j (spatial offset)
                    spatial_offset = positions[j] - positions[i]
                    spatial_offset_norm = torch.norm(spatial_offset)
                    
                    if spatial_offset_norm > 1e-6:
                        spatial_offset_normalized = spatial_offset / spatial_offset_norm
                        
                        # Alignment: how much does wind at i point toward j?
                        alignment = torch.dot(wind_dir_i, spatial_offset_normalized)
                        # Range: [-1, 1]
                    else:
                        alignment = 0.0
                    
                    # Temporal alignment: how synchronized are patterns?
                    temporal_alignment = torch.nn.functional.cosine_similarity(
                        learned_representations[i:i+1],
                        learned_representations[j:j+1]
                    ).item()
                    
                    # Learn how these influence adjacency
                    influence_input = torch.tensor(
                        [wind_dir_i[0].item(), wind_dir_i[1].item(), 
                         alignment, temporal_alignment],
                        device=device
                    )
                    
                    mod_factor = self.directional_influence(influence_input)
                    directional_modulation[i, j] = mod_factor
            
            adjacency = base_adjacency * directional_modulation
        else:
            adjacency = base_adjacency
        
        
        # ========== STEP 5: Incorporate pressure gradient influence ==========
        # Inductive bias: pressure differences may drive correlations
        # Let the model learn IF and HOW MUCH pressure matters
        
        pressure_data = historical_data[:, -1, 3]  # Last timestep pressure
        # Shape: (n_stations,)
        
        pressure_influence = torch.ones(n_stations, n_stations, device=device)
        
        for i in range(n_stations):
            for j in range(n_stations):
                pressure_diff = torch.tensor(
                    [pressure_data[i].item(), pressure_data[j].item()],
                    device=device
                )
                pressure_mod = self.pressure_influence(pressure_diff)
                pressure_influence[i, j] = pressure_mod
        
        # Combine: element-wise multiplication
        adjacency = adjacency * torch.sigmoid(pressure_influence)
        
        
        # ========== STEP 6: Horizon-specific refinement ==========
        # Different forecast horizons need different spatial structures
        # 3h: Focus on fast pressure-driven propagation
        # 24h: Focus on slower thermal/large-scale patterns
        
        horizon_projection = self.horizon_specific_projection[horizon]
        
        # Use the projection to weight the adjacency
        horizon_weights = torch.ones(n_stations, n_stations, device=device)
        
        for i in range(n_stations):
            for j in range(n_stations):
                combined_repr = torch.cat([
                    node_repr_combined[i],
                    node_repr_combined[j]
                ])
                projected = horizon_projection(combined_repr)
                weight = torch.sigmoid(projected.sum())
                horizon_weights[i, j] = weight
        
        adjacency = adjacency * horizon_weights
        
        
        # ========== STEP 7: Sparsification (keep only strong connections) ==========
        # Make the graph sparse using temperature-scaled softmax per row
        
        # Row-wise softmax with temperature scaling
        adjacency_sparse = torch.zeros_like(adjacency)
        
        for i in range(n_stations):
            row = adjacency[i, :]
            
            # Temperature-scaled softmax
            # Lower temperature → sharper selection (sparser)
            row_sparse = torch.softmax(
                row / self.temperature_scale.clamp(min=0.01),
                dim=0
            )
            
            # Keep only top-k connections or threshold-based
            threshold = torch.quantile(row_sparse, 1 - self.sparsity_threshold)
            row_sparse = row_sparse * (row_sparse >= threshold).float()
            
            # Renormalize
            row_sparse = row_sparse / (row_sparse.sum() + 1e-8)
            
            adjacency_sparse[i, :] = row_sparse
        
        
        # ========== STEP 8: Add self-loops (optional but recommended) ==========
        # Allow stations to "use" their own history
        # This helps preserve local information
        
        self_loop_strength = 0.1
        adjacency_sparse = adjacency_sparse + self_loop_strength * torch.eye(
            n_stations, device=device
        )
        
        # Normalize again
        row_sums = adjacency_sparse.sum(dim=1, keepdim=True)
        adjacency_sparse = adjacency_sparse / (row_sums + 1e-8)
        
        
        return adjacency_sparse