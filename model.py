import torch
from torch import nn
import torch.nn.functional as F
from temporalEncoder import TemporalEncoder
from adjLearner import TemporalAdjacencyLearner
from positionalEncoding import SpatialPositionalEncoding
from directionalGAT import DirectionalGAT
from cyclicEncoder import CyclicTemporalEncoder

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
                 seq_len=168, n_gnn_layers=3):
        super().__init__()
        
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim
        
        # ========== ENCODERS ==========
        self.cyclic_encoder = CyclicTemporalEncoder(
            hidden_dim=hidden_dim,
            n_harmonics=3
        )

        self.temporal_encoder = TemporalEncoder(
            input_dim=5, 
            hidden_dim=hidden_dim, 
            seq_len=seq_len
        )
        
        self.spatial_encoder = SpatialPositionalEncoding(
            hidden_dim=hidden_dim,
            n_stations=n_stations
        )
        
        # ========== GRAPH LEARNING AND GNN ==========
        
        self.adjacency_learner = TemporalAdjacencyLearner(
            hidden_dim=hidden_dim,
            n_stations=n_stations,
            embedding_dim=32
        )
        
        self.gnn_layers = nn.ModuleList([
            DirectionalGAT(hidden_dim, hidden_dim, n_heads=n_heads, dropout=0.1)
            for _ in range(n_gnn_layers)
        ])
        
        # ========== SINGLE DECODER HEAD ==========
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, historical_data, lat, lon, current_hour=None, day_of_year=None, 
                wind_direction_deg=None, positions=None):
        """
        Forward pass for 1-hour ahead wind speed forecasting.
        
        Args:
            historical_data: (n_stations, seq_len, 5)
                - wind_speed, wind_dir_cos, wind_dir_sin, pressure, temp
            lat: (n_stations,) normalized latitude
            lon: (n_stations,) normalized longitude
            current_hour: scalar or None (unused for 1h prediction)
            day_of_year: scalar or None (unused for 1h prediction)
            wind_direction_deg: (n_stations,) optional prevailing wind direction
            positions: (n_stations, 2) optional for directional modulation
        
        Returns:
            predictions: (n_stations,) wind speed prediction for 1 hour ahead
        """
        
        device = historical_data.device
        n_stations = historical_data.shape[0]
        
        # =========== STAGE 1: Temporal encoding ===========
        # Extract temporal patterns from historical data
        temporal_features = self.temporal_encoder(historical_data)
        # Shape: (n_stations, hidden_dim)

        cyclic_embed = self.cyclic_encoder(
            hour=current_hour, day_of_year=day_of_year
        )
        
        # =========== STAGE 2: Spatial encoding ===========
        spatial_embed = self.spatial_encoder(lat, lon, wind_direction_deg)
        # Shape: (n_stations, hidden_dim)
        
        # =========== STAGE 3: Combine representations ==========
        node_repr = temporal_features + spatial_embed + cyclic_embed
        # Shape: (n_stations, hidden_dim)
        # Combines: What happened (temporal) + Where is it (spatial)
        
        # =========== STAGE 4: Learn adaptive graph structure ===========
        adjacency = self.adjacency_learner(
            temporal_features,
            spatial_embed,
            horizon='1h',
            positions=positions,
            wind_directions=wind_direction_deg
        )
        # Shape: (n_stations, n_stations)
        
        # =========== STAGE 5: Apply graph neural network ===========
        gnn_output = node_repr
        for gnn_layer in self.gnn_layers:
            gnn_output = gnn_layer(gnn_output, adjacency)
            gnn_output = F.relu(gnn_output)
        # Shape: (n_stations, hidden_dim)
        
        # =========== STAGE 6: Decode prediction ===========
        prediction = self.decoder(gnn_output)
        # Shape: (n_stations, 1)
        
        return prediction.squeeze(-1)  # (n_stations,)