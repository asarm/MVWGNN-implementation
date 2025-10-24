import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAdjacencyLearner(nn.Module):
    """
    Learn adaptive spatial adjacency matrix for wind speed forecasting.
    
    Philosophy: Discover relationships from data, don't assume from geography.
    
    Process:
    1. Extract node embeddings (learnable properties of each station)
    2. Create base asymmetric similarity matrix
    3. Modulate by wind direction (inductive bias)
    4. Sparsify (keep only important connections)
    5. Normalize to create valid adjacency matrix
    
    The resulting adjacency matrix A[i,j] represents:
    "How much does station i's state influence station j?"
    
    This is different for each forecast horizon because:
    - 3h: Local pressure-driven propagation dominates
    - 6h: Regional transport becomes important
    - 12h: Diurnal boundary layer mixing kicks in
    - 24h: Large-scale synoptic patterns dominate
    """
    
    def __init__(self, 
                 hidden_dim=64, 
                 n_stations=50, 
                 embedding_dim=32,
                 sparsity_threshold=0.1,
                 temperature_scale=0.1):
        """
        Args:
            hidden_dim: Dimension of input features (64 from TemporalEncoder)
            n_stations: Number of weather stations (50 in your case)
            embedding_dim: Dimension of station embeddings (32)
            sparsity_threshold: Keep top-k% of connections (0.1 = keep top 10%)
            temperature_scale: Temperature for softmax sharpening (learnable)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        self.sparsity_threshold = sparsity_threshold
        
        # =====================================================================
        # COMPONENT 1: Node-level Learnable Embeddings
        # =====================================================================
        # Each station has unique learned properties:
        # - E1: Primary characteristics (altitude, exposure, etc.)
        # - E2: Secondary characteristics (coupling strength, etc.)
        
        self.node_embedding_1 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        self.node_embedding_2 = nn.Parameter(
            torch.randn(n_stations, embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embedding_1)
        nn.init.xavier_uniform_(self.node_embedding_2)
        
        # =====================================================================
        # COMPONENT 2: Temporal Feature Projection
        # =====================================================================
        # Project temporal features to embedding space for similarity computation
        # Input: (50, 64) from TemporalEncoder
        # Output: (50, embedding_dim)
        
        self.temporal_to_embedding = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
        
        # =====================================================================
        # COMPONENT 3: Horizon-Specific Modulation Factors
        # =====================================================================
        # Different forecast horizons need different emphasis on temporal dynamics
        # 3h:  Emphasize local dynamics
        # 6h:  Balanced approach
        # 12h: Regional patterns
        # 24h: Large-scale patterns
        
        self.horizon_modulation = nn.ModuleDict({
            '3h': nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(embedding_dim // 2, 1),
                nn.Sigmoid()  # Output in [0, 1]
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
        
        # =====================================================================
        # COMPONENT 4: Directional Influence Modulation
        # =====================================================================
        # Learn how wind direction affects graph structure
        # Input: [wind_cos, wind_sin, spatial_alignment]
        # Output: modulation factor in [0, 1]
        
        self.directional_influence = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        # =====================================================================
        # COMPONENT 5: Learnable Temperature for Softmax
        # =====================================================================
        # Controls sparsity: higher temperature = sharper peaks = sparser graph
        # Initialized to small value, will be learned during training
        
        self.temperature_scale = nn.Parameter(
            torch.tensor(temperature_scale)
        )
        
        # =====================================================================
        # COMPONENT 6: Learnable Station Bias
        # =====================================================================
        # Each station has a learnable bias affecting outgoing influence
        # Captures properties like "does this station dominate neighbors?"
        
        self.station_bias_out = nn.Parameter(
            torch.randn(n_stations, 1) * 0.1
        )
        self.station_bias_in = nn.Parameter(
            torch.randn(n_stations, 1) * 0.1
        )
        
    
    def _compute_base_adjacency(self, temporal_embedding):
        """
        Compute base adjacency matrix from learnable embeddings.
        
        This creates an ASYMMETRIC graph: A[i,j] ≠ A[j,i]
        This is crucial for wind speed forecasting because:
        - Wind at station i carries influence downwind to station j
        - But station j's wind doesn't carry influence back to i
        
        Args:
            temporal_embedding: (n_stations, embedding_dim)
                                From temporal feature projection
        
        Returns:
            base_adjacency: (n_stations, n_stations)
                           Asymmetric, unbounded
        
        Mathematical intuition:
        - M1 @ M2^T: Forward coupling strength
        - M2 @ M1^T: Backward coupling strength
        - Difference: Directional bias
        """
        
        # Combine learnable embeddings with temporal patterns
        # Each station's state is influenced by:
        # 1. Its fixed properties (learnable embeddings)
        # 2. Its recent dynamics (temporal embedding)
        
        M1 = torch.tanh(self.node_embedding_1 + temporal_embedding)
        # (n_stations, embedding_dim)
        
        M2 = torch.tanh(self.node_embedding_2)
        # (n_stations, embedding_dim)
        
        # Forward coupling: how much does i influence j?
        forward = M1 @ M2.T
        # (n_stations, n_stations)
        
        # Backward coupling: symmetry baseline
        backward = M2 @ M1.T
        # (n_stations, n_stations)
        
        # Asymmetric difference
        # Positive = station i pushes influence forward to j
        # This is what we want for wind propagation!
        base_adjacency = torch.relu(
            torch.tanh(forward - backward)
        )
        # (n_stations, n_stations)
        
        # Add station-specific biases
        # Some stations naturally have stronger outgoing/incoming influence
        base_adjacency = (
            base_adjacency + 
            self.station_bias_out +           # (n_stations, 1)
            self.station_bias_in.T            # (1, n_stations)
        )
        
        return base_adjacency
    
    
    def _apply_directional_modulation(self, adjacency, wind_directions, positions):
        """
        Modulate adjacency by wind direction.
        
        Key insight: Wind carries influence from upwind to downwind.
        This provides an INDUCTIVE BIAS, not a hard constraint.
        The model learns how important this bias actually is.
        
        Args:
            adjacency: (n_stations, n_stations) base similarity
            wind_directions: (n_stations,) wind direction in degrees
            positions: (n_stations, 2) lat/lon normalized
        
        Returns:
            modulated_adjacency: (n_stations, n_stations)
                                 Weighted by directional alignment
        """
        
        device = adjacency.device
        n_stations = adjacency.shape[0]
        
        # Convert wind directions to unit vectors
        wind_dir_rad = torch.deg2rad(wind_directions)
        wind_vectors = torch.stack([
            torch.cos(wind_dir_rad),
            torch.sin(wind_dir_rad)
        ], dim=1)
        # (n_stations, 2)
        
        # Compute directional modulation matrix
        modulation = torch.ones(n_stations, n_stations, device=device)
        
        for i in range(n_stations):
            for j in range(n_stations):
                if i == j:
                    # Self-loops unaffected
                    continue
                
                # Vector from i to j (spatial direction)
                spatial_offset = positions[j] - positions[i]
                spatial_dist = torch.norm(spatial_offset)
                
                if spatial_dist < 1e-6:
                    # Same location (shouldn't happen)
                    mod_factor = torch.tensor(0.5, device=device)
                else:
                    # Normalize spatial direction
                    spatial_dir = spatial_offset / spatial_dist
                    
                    # Compute alignment between wind direction at i 
                    # and spatial direction from i to j
                    alignment = torch.dot(wind_vectors[i], spatial_dir)
                    # alignment in [-1, 1]
                    # +1: j is downwind from i (high influence)
                    # 0: j is perpendicular (neutral)
                    # -1: j is upwind from i (low influence)
                    
                    # Learn how much to emphasize this alignment
                    dir_input = torch.tensor([
                        wind_vectors[i, 0].item(),
                        wind_vectors[i, 1].item(),
                        alignment.item()
                    ], device=device)
                    
                    mod_factor = self.directional_influence(dir_input)
                    # Output in [0, 1]
                
                modulation[i, j] = mod_factor
        
        # Apply modulation
        modulated_adjacency = adjacency * modulation
        
        return modulated_adjacency
    
    
    def _sparsify_adjacency(self, adjacency):
        """
        Sparsify adjacency matrix by keeping only top connections per row.
        
        Why sparsify?
        1. Computational efficiency (sparse matrix operations)
        2. Interpretability (clear which neighbors matter)
        3. Avoids over-smoothing (GNN issue with dense graphs)
        4. Physical realism (distant stations shouldn't directly influence each other)
        
        Process:
        1. Apply temperature-scaled softmax (sharpens distribution)
        2. Quantile thresholding (keep top-k connections)
        3. Renormalize (stochastic matrix property)
        
        Args:
            adjacency: (n_stations, n_stations) dense matrix
        
        Returns:
            sparse_adjacency: (n_stations, n_stations) sparse/sparse-ish
        """
        
        device = adjacency.device
        n_stations = adjacency.shape[0]
        
        sparse_adjacency = torch.zeros_like(adjacency)
        
        for i in range(n_stations):
            row = adjacency[i, :]  # (n_stations,)
            
            # Apply temperature-scaled softmax for sharpening
            # Higher temperature → more uniform
            # Lower temperature → sharper peaks
            temp = self.temperature_scale.clamp(min=0.01)
            row_softmax = torch.softmax(row / temp, dim=0)
            # (n_stations,)
            
            # Quantile-based thresholding
            # Keep only connections above certain percentile
            threshold = torch.quantile(
                row_softmax, 
                1 - self.sparsity_threshold
            )
            
            # Create sparse version
            row_sparse = row_softmax * (row_softmax >= threshold).float()
            # (n_stations,)
            
            # Renormalize so row sums to ~1
            row_sum = row_sparse.sum()
            if row_sum > 1e-8:
                row_sparse = row_sparse / row_sum
            
            sparse_adjacency[i, :] = row_sparse
        
        return sparse_adjacency
    
    
    def _add_self_loops(self, adjacency, self_loop_weight=0.1):
        """
        Add self-loops to adjacency matrix.
        
        Why self-loops?
        - Each station should consider its own history (weight = 0.1)
        - Prevents vanishing signals in deep GNNs
        - Helps gradient flow during training
        
        Args:
            adjacency: (n_stations, n_stations) sparse matrix
            self_loop_weight: Weight for A[i,i]
        
        Returns:
            adjacency_with_loops: (n_stations, n_stations)
        """
        
        n_stations = adjacency.shape[0]
        device = adjacency.device
        
        # Add identity matrix
        identity = torch.eye(n_stations, device=device)
        adjacency_with_loops = adjacency + self_loop_weight * identity
        
        return adjacency_with_loops
    
    
    def _normalize_adjacency(self, adjacency):
        """
        Normalize adjacency matrix (row-wise stochastic).
        
        After all operations, ensure rows sum to 1 (approximately).
        This makes it a proper stochastic matrix and improves numerical stability.
        
        Args:
            adjacency: (n_stations, n_stations)
        
        Returns:
            normalized: (n_stations, n_stations) row-stochastic
        """
        
        # Row-wise normalization
        row_sums = adjacency.sum(dim=1, keepdim=True)
        
        # Avoid division by zero
        row_sums = torch.clamp(row_sums, min=1e-8)
        
        normalized = adjacency / row_sums
        
        return normalized
    
    
    def forward(self, 
                temporal_features,      # (50, 64) from TemporalEncoder
                learned_representations,# (50, 64) combined repr
                horizon='6h',            # '3h', '6h', '12h', '24h'
                positions=None,          # (50, 2) lat/lon
                wind_directions=None):   # (50,) degrees
        """
        Learn adaptive adjacency matrix.
        
        Args:
            temporal_features: (n_stations, hidden_dim)
                Temporal patterns from TemporalEncoder
                Contains: wind speed, pressure, temperature dynamics
                
            learned_representations: (n_stations, hidden_dim)
                Combined spatial + temporal + cyclic representations
                (Not used directly but available for future extensions)
                
            horizon: str, one of ['3h', '6h', '12h', '24h']
                Different horizons get different adjacency matrices
                
            positions: (n_stations, 2) optional
                Normalized latitude/longitude for directional modulation
                
            wind_directions: (n_stations,) optional
                Prevailing wind direction at each station in degrees
                If None, uses default (e.g., 270° for westerly)
        
        Returns:
            adjacency: (n_stations, n_stations)
                Sparse, normalized adjacency matrix
                A[i,j] = influence strength from station i to station j
        """
        
        device = temporal_features.device
        n_stations = temporal_features.shape[0]
        
        # =====================================================================
        # STEP 1: Project temporal features to embedding space
        # =====================================================================
        # Convert temporal patterns to embedding dimension for similarity
        temporal_embedding = self.temporal_to_embedding(temporal_features)
        # (n_stations, embedding_dim)
        
        # =====================================================================
        # STEP 2: Compute base asymmetric adjacency
        # =====================================================================
        # From learnable embeddings + temporal patterns
        base_adjacency = self._compute_base_adjacency(temporal_embedding)
        # (n_stations, n_stations)
        # Shape: unbounded, asymmetric, dense
        
        # =====================================================================
        # STEP 3: Apply horizon-specific modulation
        # =====================================================================
        # Different forecast horizons emphasize different dynamics
        horizon_mod = self.horizon_modulation[horizon](temporal_embedding)
        # (n_stations, 1)
        
        adjacency = base_adjacency * horizon_mod
        # (n_stations, n_stations)
        
        # =====================================================================
        # STEP 4: Apply directional modulation (if available)
        # =====================================================================
        # Incorporate wind direction as inductive bias
        if wind_directions is None:
            # Default: UK typical westerly winds (270°)
            wind_directions = torch.ones(n_stations, device=device) * 270.0
        
        if positions is None:
            # Default: grid positions if not provided
            # Create simple grid layout
            grid_size = int(math.sqrt(n_stations)) + 1
            x_pos = torch.arange(n_stations, device=device) % grid_size
            y_pos = torch.arange(n_stations, device=device) // grid_size
            positions = torch.stack([x_pos, y_pos], dim=1).float()
            positions = positions / positions.max()  # Normalize to [0, 1]
        
        adjacency = self._apply_directional_modulation(
            adjacency, wind_directions, positions
        )
        # (n_stations, n_stations)
        
        # =====================================================================
        # STEP 5: Sparsify adjacency matrix
        # =====================================================================
        # Keep only most important connections
        adjacency = self._sparsify_adjacency(adjacency)
        # (n_stations, n_stations)
        # Shape: mostly zero, only top-k% per row non-zero
        
        # =====================================================================
        # STEP 6: Add self-loops
        # =====================================================================
        # Each station should consider its own recent history
        adjacency = self._add_self_loops(adjacency, self_loop_weight=0.1)
        # (n_stations, n_stations)
        
        # =====================================================================
        # STEP 7: Normalize to stochastic matrix
        # =====================================================================
        # Ensure rows sum to ~1 for numerical stability
        adjacency = self._normalize_adjacency(adjacency)
        # (n_stations, n_stations)
        
        return adjacency