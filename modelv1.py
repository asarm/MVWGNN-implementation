import torch
from torch import nn
import torch.nn.functional as F
from temporalEncoder import TemporalEncoder
from adjLearner import TemporalAdjacencyLearner
from positionalEncoding import SpatialPositionalEncoding
from directionalGAT import DirectionalGAT
from cyclicEncoder import CyclicTemporalEncoding

class DDGNNWind(nn.Module):
    """
    Directional Dynamic Graph Neural Network for Wind Speed Forecasting.
    
    SIMPLIFIED VERSION FOR 1-HOUR FORECASTING:
    - Single horizon prediction (1 hour ahead)
    - Removed multi-horizon architecture and cyclic encoders
    - Focuses on immediate temporal dynamics
    
    Architecture:
        Input: Historical data + spatial info
            ↓
        [Temporal + Spatial] Encoders (parallel)
            ↓
        [Fusion]
            ↓
        [Learn Adjacency from temporal_features]
            ↓
        [GNN Propagation]
            ↓
        [Decode Prediction]
            ↓
        Output: Wind speed prediction (1h ahead)
    """
    
    def __init__(self, n_stations=50, hidden_dim=64, n_heads=4, 
                 seq_len=168, n_gnn_layers=3, input_dim=5, temporal_debug: bool = False):
        super().__init__()
        
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # ========== ENCODERS ==========
        self.cyclic_encoder = CyclicTemporalEncoding(
            hidden_dim=hidden_dim,
            n_harmonics=3
        )

        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim, 
            seq_len=seq_len,
            debug=temporal_debug
        )
        
        self.spatial_encoder = SpatialPositionalEncoding(
            hidden_dim=hidden_dim,
            n_stations=n_stations
        )
        
        # ========== GRAPH LEARNING AND GNN ==========
        
        self.adjacency_learner = TemporalAdjacencyLearner(
            hidden_dim=hidden_dim,
            n_stations=n_stations,
            embedding_dim=32,
            sparsify_mode='top_p',      # variable edges per node based on mass
            nucleus_p=0.9,              # keep minimum set covering 70% prob mass
            temperature_scale=0.2       # slightly smoother distribution for stability
        )
        
        self.gnn_layers = nn.ModuleList([
            DirectionalGAT(hidden_dim, hidden_dim, n_heads=n_heads, dropout=0.3)
            for _ in range(n_gnn_layers)
        ])
        
        # ========== SINGLE DECODER HEAD ==========
        
        # Multi-horizon decoder: outputs 1 prediction [1h ahead]
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)  # 1 horizon: [1]
        )
        
        # Learnable weight for blending spatial and temporal features in adjacency learning
        self.temporal_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, historical_data, lat, lon, current_hour=None, day_of_year=None, 
                wind_direction_deg=None, positions=None):
        """
        Forward pass for multi-horizon wind speed forecasting.
        
        Args:
            historical_data: (n_stations, seq_len, 5) or (B, n_stations, seq_len, 5)
                - wind_speed, wind_dir_cos, wind_dir_sin, pressure, temp
            lat: (n_stations,) or (B, n_stations,) normalized latitude
            lon: (n_stations,) or (B, n_stations,) normalized longitude
            current_hour: scalar or (B,) tensor
            day_of_year: scalar or (B,) tensor
            wind_direction_deg: (n_stations,) or (B, n_stations,) optional prevailing wind direction
            positions: (n_stations, 2) or (B, n_stations, 2) optional for directional modulation
        
        Returns:
            predictions: (n_stations, 5) or (B, n_stations, 5) wind speed predictions for [1, 3, 6, 12, 24] hours ahead
        """
        
        device = historical_data.device

        is_batched = historical_data.dim() == 4

        # =========== STAGE 1: Temporal encoding ===========
        temporal_features = self.temporal_encoder(historical_data)
        # temporal_features: (N, H) or (B, N, H)

        # Ensure cyclic inputs exist
        if current_hour is None:
            current_hour = torch.tensor(0, device=device)
        if day_of_year is None:
            day_of_year = torch.tensor(1, device=device)

        # cyclic encoder expects (batch,) inputs; if not batched, make batch of size 1
        if is_batched:
            # current_hour/day_of_year should be (B,)
            cyclic_embed = self.cyclic_encoder(hour=current_hour, day_of_year=day_of_year)
            # (B, H)
        else:
            cyclic_embed = self.cyclic_encoder(hour=current_hour, day_of_year=day_of_year)
            # (1, H) or (H,) depending on cyclic encoder behavior

        # =========== STAGE 2: Spatial encoding ==========
        spatial_embed = self.spatial_encoder(lat, lon, wind_direction_deg)
        # spatial_embed: (N, H) or (B, N, H)

        # =========== STAGE 3: Combine representations ==========
        if not is_batched:
            # Everything is station-wise: (N, H)
            # Ensure cyclic_embed is (H,) -> expand to (N, H)
            if cyclic_embed.dim() == 2 and cyclic_embed.shape[0] == 1:
                cyc = cyclic_embed.squeeze(0)
            else:
                cyc = cyclic_embed

            node_repr = temporal_features + spatial_embed + cyc
            # =========== STAGE 4: Learn adaptive graph structure ==========
            # CRITICAL FIX: Learn adjacency from SPATIAL features only, not temporal.
            # This prevents the model from creating a graph based on feature similarity,
            # which leads to the "lagging" prediction issue.
            # UPDATED: Blend spatial and temporal for richer representations
            combined_repr = spatial_embed + self.temporal_weight * temporal_features  # Learnable weight
            adjacency = self.adjacency_learner(
                station_representations=combined_repr, 
                positions=positions, 
                wind_directions=wind_direction_deg
            )

            # =========== STAGE 5: Apply graph neural network ==========
            gnn_output = node_repr
            for i, gnn_layer in enumerate(self.gnn_layers):
                residual = gnn_output  # Save residual
                gnn_output = gnn_layer(gnn_output, adjacency)
                gnn_output = F.relu(gnn_output)
                gnn_output = gnn_output + residual  # Add residual connection

            # =========== STAGE 6: Decode prediction ==========
            prediction = self.decoder(gnn_output)
            return prediction  # (n_stations, 5)
        else:
            # Batched case: temporal_features (B,N,H), spatial_embed (B,N,H)
            B, N, H = temporal_features.shape

            # cyclic_embed: (B,H) or (H,) -> make (B,1,H)
            if cyclic_embed.dim() == 1:
                cyc = cyclic_embed.unsqueeze(0).expand(B, -1)
            else:
                cyc = cyclic_embed
            cyc = cyc.unsqueeze(1)  # (B,1,H)

            node_repr = temporal_features + spatial_embed + cyc

            # adjacency: (B,N,N)
            
            combined_repr = spatial_embed + self.temporal_weight * temporal_features
            adjacency = self.adjacency_learner(
                station_representations=combined_repr, positions=positions, wind_directions=wind_direction_deg
            )

            # Apply GNN layers (batched)
            gnn_output = node_repr
            for i, gnn_layer in enumerate(self.gnn_layers):
                residual = gnn_output  # Save residual
                gnn_output = gnn_layer(gnn_output, adjacency)
                gnn_output = F.relu(gnn_output)
                gnn_output = gnn_output + residual  # Add residual connection

            # Decode: (B,N,H) -> (B,N,1)
            prediction = self.decoder(gnn_output)
            # Return (B, N, 1)
            return prediction