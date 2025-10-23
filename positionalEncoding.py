import torch
from torch import nn

class SpatialPositionalEncoding(nn.Module):
    """
    Simplified spatial encoding using only:
    - Latitude, Longitude (station positions)
    - Wind Direction (inductive bias)
    
    Minimal learnable parameters with maximum inductive bias.
    """
    
    def __init__(self, hidden_dim=64, n_stations=50):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # ========== ONLY LEARNABLE COMPONENTS ==========
        
        # 1. Learnable station-specific embeddings (captures unique characteristics)
        # One embedding per station: accounts for local effects we can't measure
        self.station_embeddings = nn.Parameter(
            torch.randn(n_stations, hidden_dim)
        )
        nn.init.xavier_uniform_(self.station_embeddings)
        
        # 2. Project spatial features (lat/lon/direction) to hidden space
        # Input: [lat, lon, dir_cos, dir_sin] (4 values)
        # This is the ONLY learnable transformation of spatial features
        self.spatial_projection = nn.Linear(4, hidden_dim)
        
        
    def forward(self, lat, lon, wind_direction_deg):
        """
        Args:
            lat: (n_stations,) latitude normalized to [0, 1]
            lon: (n_stations,) longitude normalized to [0, 1]
            wind_direction_deg: (n_stations,) wind direction in degrees [0, 360)
        
        Returns:
            spatial_embed: (n_stations, hidden_dim)
        """
        
        device = lat.device
        n_stations = lat.shape[0]
        
        # Convert direction to circular representation
        wind_dir_rad = torch.deg2rad(wind_direction_deg)
        wind_cos = torch.cos(wind_dir_rad)
        wind_sin = torch.sin(wind_dir_rad)
        
        # Stack all spatial features
        spatial_features = torch.stack([lat, lon, wind_cos, wind_sin], dim=1)
        # Shape: (n_stations, 4)
        
        # Project to hidden space
        spatial_proj = self.spatial_projection(spatial_features)
        # Shape: (n_stations, hidden_dim)
        
        # Add station-specific embeddings (learnable offsets)
        spatial_embed = spatial_proj + self.station_embeddings
        
        return spatial_embed