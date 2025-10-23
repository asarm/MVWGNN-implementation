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
    
    def __init__(self, input_dim=7, hidden_dim=64, seq_len=168):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # ========== Multi-scale Temporal Convolutions ==========
        # Dilated convolutions capture patterns at different timescales
        
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=8, padding=8),
        ])
        
        # ========== LSTM for Long-Range Dependencies ==========
        # RNNs excel at capturing complex temporal dynamics
        # But we use bidirectional to see full context
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 4,  # concatenate all dilated conv outputs
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )
        
        # ========== Attention over Time ==========
        # Learn which timesteps matter most
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # from bi-LSTM
            num_heads=4,
            batch_first=True
        )
        
        # ========== Output projection ==========
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, historical_data):
        """
        Args:
            historical_data: (n_stations, seq_len, input_dim)
        
        Returns:
            temporal_features: (n_stations, hidden_dim)
                Aggregated representation of the full history
        """
        
        # Reshape for Conv1d: (n_stations, input_dim, seq_len)
        x = historical_data.transpose(1, 2)
        
        # ========== Multi-scale convolution ==========
        multi_scale_features = []
        for conv in self.dilated_convs:
            features = F.relu(conv(x))
            # Shape: (n_stations, hidden_dim, seq_len)
            multi_scale_features.append(features)
        
        # Concatenate all scales
        x_conv = torch.cat(multi_scale_features, dim=1)
        # Shape: (n_stations, hidden_dim*4, seq_len)
        
        # Reshape back: (n_stations, seq_len, hidden_dim*4)
        x_conv = x_conv.transpose(1, 2)
        
        
        # ========== LSTM for temporal aggregation ==========
        lstm_output, (h_n, c_n) = self.lstm(x_conv)
        # lstm_output: (n_stations, seq_len, hidden_dim*2)
        # h_n: (2*num_layers, n_stations, hidden_dim) - final hidden state
        
        
        # ========== Attention: which timesteps matter? ==========
        attended, _ = self.temporal_attention(
            lstm_output, lstm_output, lstm_output
        )
        # Shape: (n_stations, seq_len, hidden_dim*2)
        
        
        # ========== Aggregate over time ==========
        # Use the last hidden state + attention-weighted average
        
        last_state = h_n[-1]  # Take last layer's hidden state
        # Shape: (n_stations, hidden_dim)
        
        # Attention-weighted aggregation
        attention_weights = torch.softmax(
            torch.sum(attended, dim=-1, keepdim=True), dim=1
        )
        # Shape: (n_stations, seq_len, 1)
        
        attended_aggregated = torch.sum(
            attended * attention_weights, dim=1
        )
        # Shape: (n_stations, hidden_dim*2)
        
        # Combine final state + attended aggregation
        temporal_repr = torch.cat([last_state, attended_aggregated], dim=-1)
        # Shape: (n_stations, hidden_dim*3)
        
        
        # ========== Final projection ==========
        temporal_features = self.output_proj(temporal_repr[:, :2*self.hidden_dim])
        # Shape: (n_stations, hidden_dim)
        
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