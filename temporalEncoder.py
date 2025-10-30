import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    """
    Extract temporal patterns from historical sequences using multiple scales
    
    Key insight: Different temporal patterns matter at different scales
    - High frequency: hourly fluctuations (local turbulence)
    - Medium frequency: diurnal cycle (heating/cooling)
    - Low frequency: synoptic patterns (pressure systems)
    """
    
    def __init__(self, input_dim=7, hidden_dim=64, seq_len=168, debug: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # Debug only active when both the flag is True and TEMPORAL_DEBUG=1
        self.debug = bool(debug) and (os.getenv("TEMPORAL_DEBUG", "0") == "1")
        
        # ========== Multi-scale Temporal Convolutions ==========
        # Dilated convolutions capture patterns at different timescales
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=8, padding=8),
        ])
        
        # ========== LSTM removed ==========
        # We now rely purely on multi-scale temporal convolutions
        
        # ========== Attention over Time ==========
        # Removed attention mechanism to simplify encoder and reduce compute
        
        # ========== Output projection ==========
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # project pooled conv features
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, historical_data):
        """
        Args:
            historical_data: (n_stations, seq_len, input_dim) or
                             (batch, n_stations, seq_len, input_dim)

        Returns:
            temporal_features: (n_stations, hidden_dim) or
                (batch, n_stations, hidden_dim) when batched
                Aggregated representation of the full history
        """
        is_batched = historical_data.dim() == 4

        if is_batched:
            B, N, S, D = historical_data.shape
            # merge batch and station dims for temporal processing
            x = historical_data.reshape(B * N, S, D).transpose(1, 2)
        else:
            # Reshape for Conv1d: (n_stations, input_dim, seq_len)
            x = historical_data.transpose(1, 2)
        
        # ========== Multi-scale convolution ==========
        multi_scale_features = []
        for conv in self.dilated_convs:
            features = F.relu(conv(x))
            # Shape: (batch*n_stations, hidden_dim, seq_len) or (n_stations, hidden_dim, seq_len)
            multi_scale_features.append(features)
        
        # Concatenate all scales
        x_conv = torch.cat(multi_scale_features, dim=1)
        # Shape: (n_stations, hidden_dim*4, seq_len)

        # ========== Aggregate over time (no LSTM/attention) ==========
        # CRITICAL: Use MEAN pooling to prevent last-timestep bias
        # Pool directly over time dimension of concatenated multi-scale convs
        temporal_features = torch.mean(x_conv, dim=2)
        if self.debug:
            print(f"[ENCODER] Final output - mean: {temporal_features.mean():.4f}, "
                  f"std: {temporal_features.std():.4f}, min: {temporal_features.min():.4f}, "
                  f"max: {temporal_features.max():.4f}")
        # Shape: (batch*n_stations, hidden_dim)
        
        # ========== Final projection ==========
        temporal_features = self.output_proj(temporal_features)
        # Shape: (batch*n_stations, hidden_dim) or (n_stations, hidden_dim)

        if is_batched:
            temporal_features = temporal_features.view(B, N, self.hidden_dim)

        return temporal_features
    

class DirectionalTemporalEncoder(nn.Module):
    """
    Encode temporal patterns CONDITIONAL on wind direction
    
    Key idea: Different wind regimes evolve differently over time
    - Northerly wind: cold, stable
    - Southerly wind: warm, unstable
    - Different temporal patterns for each regime
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Separate encoders for different wind regimes
        self.northerly_encoder = TemporalEncoder(hidden_dim)
        self.easterly_encoder = TemporalEncoder(hidden_dim)
        self.southerly_encoder = TemporalEncoder(hidden_dim)
        self.westerly_encoder = TemporalEncoder(hidden_dim)
        
        # Learn which regime at each timestep
        self.regime_classifier = nn.Sequential(
            nn.Linear(2, 32),  # wind direction vector
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 regimes
            nn.Softmax(dim=-1)
        )
        
    def forward(self, historical_data, wind_directions):
        """
        Args:
            historical_data: (n_stations, seq_len, features)
            wind_directions: (n_stations, seq_len, 2) - [cos(dir), sin(dir)]
        """
        
        # Classify dominant wind regime at each station
        regime_probs = self.regime_classifier(wind_directions)
        # Shape: (n_stations, seq_len, 4)
        
        # Get temporal encoding from each regime-specific encoder
        encodings = [
            self.northerly_encoder(historical_data),
            self.easterly_encoder(historical_data),
            self.southerly_encoder(historical_data),
            self.westerly_encoder(historical_data)
        ]
        
        # Weighted combination based on regime
        final_repr = torch.sum(torch.stack(encodings) * regime_probs.T, dim=0)
        
        return final_repr