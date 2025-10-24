import torch
from torch import nn
import math

class CyclicTemporalEncoding(nn.Module):
    """
    Simplified cyclic temporal encoding.
    
    Only encodes:
    - Hour of day (diurnal cycle)
    - Day of year (seasonal cycle)
    - With multiple harmonics
    """
    
    def __init__(self, hidden_dim=64, n_harmonics=3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_harmonics = n_harmonics
        
        # Compute cyclic feature dimension
        # For each cycle (hourly, annual), we have 2 * n_harmonics features
        # 2 (sin/cos) × n_harmonics (different frequencies)
        cyclic_feature_dim = 2 * n_harmonics * 2  # 2 cycles: hourly + annual
        
        # ONLY ONE learnable layer: project cyclic features to hidden space
        self.cyclic_projection = nn.Linear(cyclic_feature_dim, hidden_dim)
        
        
    def _create_cyclic_features(self, value, period, device):
        """
        Create sine/cosine features at multiple frequencies.
        These are fixed
        """
        value = value.float().to(device)
        features = []
        for harmonic in range(1, self.n_harmonics + 1):
            freq = 2 * math.pi * harmonic / period # 2*p for converting radians, 
            features.append(torch.sin(value * freq))
            features.append(torch.cos(value * freq))
        
        return torch.stack(features, dim=1)
    
    
    def forward(self, hour, day_of_year):
        """
        Args:
            hour: (batch_size,) hour of day 0-23 (can be fractional)
            day_of_year: (batch_size,) day of year 0-365
        
        Returns:
            temporal_embed: (batch_size, hidden_dim)
        """
        
        device = hour.device
        hour = hour.float().to(device)
        day_of_year = day_of_year.float().to(device)
        
        # Create cyclic features (fixed, no parameters)
        hourly_cyclic = self._create_cyclic_features(hour, period=24, device=device)
        # Shape: (batch_size, 2 * n_harmonics)
        
        annual_cyclic = self._create_cyclic_features(day_of_year, period=365.25, device=device)
        # Shape: (batch_size, 2 * n_harmonics)
        
        # Concatenate all cyclic features
        all_cyclic = torch.cat([hourly_cyclic, annual_cyclic], dim=1)
        # Shape: (batch_size, 2 * n_harmonics * 2)
        
        # Project to hidden space (ONLY learnable part)
        temporal_embed = self.cyclic_projection(all_cyclic)
        # Shape: (batch_size, hidden_dim)
        
        return temporal_embed
