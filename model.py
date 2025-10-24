import torch
from torch import nn
import torch.nn.functional as F
from temporalEncoder import TemporalEncoder
from adjLearner import TemporalAdjacencyLearner
from positionalEncoding import SpatialPositionalEncoding
from cyclicEncoder import CyclicTemporalEncoding
from directionalGAT import DirectionalGAT

class DDGNNWind(nn.Module):
    """
    Directional Dynamic Graph Neural Network for Wind Speed Forecasting.
    
    UPDATED VERSION:
    - TemporalAdjacencyLearner receives temporal_features (not raw data)
    - No redundant wind speed extraction
    - Cleaner, more efficient design
    
    Architecture:
        Input: Historical data + spatial + temporal info
            ↓
        [Temporal + Spatial + Cyclic] Encoders (parallel)
            ↓
        [Fusion]
            ↓
        FOR EACH HORIZON:
            [Learn Adjacency from temporal_features]
            [GNN Propagation]
            [Decode Prediction]
            ↓
        Output: Wind speed predictions (3h, 6h, 12h, 24h)
    """
    
    def __init__(self, n_stations=50, hidden_dim=64, n_heads=4, 
                 seq_len=168, n_gnn_layers=3):
        super().__init__()
        
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim
        
        # ========== ENCODERS ==========
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=5, 
            hidden_dim=hidden_dim, 
            seq_len=seq_len
        )
        
        self.spatial_encoder = SpatialPositionalEncoding(
            hidden_dim=hidden_dim,
            n_stations=n_stations
        )
        
        self.temporal_cyclic = nn.ModuleDict({
            '3h': CyclicTemporalEncoding(hidden_dim, n_harmonics=3),
            '6h': CyclicTemporalEncoding(hidden_dim, n_harmonics=3),
            '12h': CyclicTemporalEncoding(hidden_dim, n_harmonics=3),
            '24h': CyclicTemporalEncoding(hidden_dim, n_harmonics=3)
        })
        
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
        
        # ========== MULTI-HORIZON DECODERS ==========
        
        self.decoder_heads = nn.ModuleDict({
            '3h': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            '6h': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            '12h': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ),
            '24h': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        })
        
    def forward(self, historical_data, lat, lon, current_hour, day_of_year, 
                wind_direction_deg=None, positions=None):
        """
        Complete forward pass of DDGNNWind (UPDATED).
        
        Args:
            historical_data: (n_stations, seq_len, 5)
                - wind_speed, wind_dir_cos, wind_dir_sin, pressure, temp
            lat: (n_stations,) normalized latitude
            lon: (n_stations,) normalized longitude
            current_hour: scalar (0-23)
            day_of_year: scalar (0-365)
            wind_direction_deg: (n_stations,) optional prevailing wind direction
            positions: (n_stations, 2) optional for directional modulation
        
        Returns:
            predictions: dict with keys '3h', '6h', '12h', '24h'
                Each value is (n_stations,) with predicted wind speed
        """
        
        device = historical_data.device
        n_stations = historical_data.shape[0]
        
        # =========== STAGE 1: Temporal encoding ===========
        # Extract ALL temporal patterns (wind, pressure, temp, etc.)
        temporal_features = self.temporal_encoder(historical_data)
        # Shape: (n_stations, hidden_dim=64)
        # This ALREADY contains wind speed dynamics!
        
        
        # =========== STAGE 2: Spatial encoding ===========
        spatial_embed = self.spatial_encoder(lat, lon, wind_direction_deg)
        # Shape: (n_stations, hidden_dim=64)
        
        
        # =========== STAGE 3: Loop over forecast horizons ===========
        predictions = {}
        
        for horizon in ['3h', '6h', '12h', '24h']:
            
            # --------- STAGE 3a: Horizon-specific temporal context ---------
            if isinstance(current_hour, (int, float)):
                hour_tensor = torch.tensor(current_hour, device=device).unsqueeze(0)
            else:
                hour_tensor = current_hour
            
            if isinstance(day_of_year, (int, float)):
                day_tensor = torch.tensor(day_of_year, device=device).unsqueeze(0)
            else:
                day_tensor = day_of_year
            
            horizon_context = self.temporal_cyclic[horizon](hour_tensor, day_tensor)
            # Shape: (1, hidden_dim=64)
            
            # Broadcast to all stations
            horizon_context = horizon_context.expand(n_stations, -1)
            # Shape: (n_stations, hidden_dim=64)
            
            
            # --------- STAGE 3b: Combine all representations ---------
            node_repr = temporal_features + spatial_embed + horizon_context
            # Shape: (n_stations, hidden_dim=64)
            # Combines: What happened? Where is it? What time is it?
            
            
            # --------- STAGE 3c: Learn adaptive graph structure ---------
            # ⭐ KEY FIX: Pass temporal_features (no redundancy!)
            adjacency = self.adjacency_learner(
                temporal_features,        # (50, 64) ← Already extracted!
                node_repr,
                horizon=horizon,
                positions=positions,
                wind_directions=wind_direction_deg
            )
            # Shape: (n_stations, n_stations)
            
            
            # --------- STAGE 3d: Apply graph neural network ---------
            gnn_output = node_repr
            for gnn_layer in self.gnn_layers:
                gnn_output = gnn_layer(gnn_output, adjacency)
                gnn_output = F.relu(gnn_output)
            # Shape: (n_stations, hidden_dim=64)
            
            
            # --------- STAGE 3e: Decode for this horizon ---------
            horizon_pred = self.decoder_heads[horizon](gnn_output)
            # Shape: (n_stations, 1)
            
            predictions[horizon] = horizon_pred.squeeze(-1)  # (n_stations,)
        
        
        return predictions