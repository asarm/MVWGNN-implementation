import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def date2cyclic_features(dayofyear, hourofday):
    """
    Create cyclic features for daily and yearly patterns.
    
    Args:
        dayofyear: 1-365 (tensor or scalar)
        hourofday: 0-23 (tensor or scalar)
    
    Returns:
        day_sin, day_cos, hour_sin, hour_cos
    """
    # Convert to tensor or numpy array
    if isinstance(dayofyear, (int, float)):
        dayofyear = np.array([dayofyear])
    if isinstance(hourofday, (int, float)):
        hourofday = np.array([hourofday])
    
    if torch.is_tensor(dayofyear):
        dayofyear = dayofyear.cpu().numpy()
    if torch.is_tensor(hourofday):
        hourofday = hourofday.cpu().numpy()
    
    day_rad = 2 * np.pi * (dayofyear - 1) / 365.0
    hour_rad = 2 * np.pi * hourofday / 24.0
    
    day_sin = np.sin(day_rad)
    day_cos = np.cos(day_rad)
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)
    
    return day_sin, day_cos, hour_sin, hour_cos


class DilatedTemporalEncoder(nn.Module):
    """
    Temporal Convolutional Network (TCN) with dilated convolutions.
    Captures multi-scale temporal patterns efficiently.
    
    Architecture:
    - 4 dilated conv layers with dilation rates [1, 2, 4, 8]
    - Residual connections for gradient flow
    - BatchNorm + Dropout for regularization
    - Cyclic temporal features integration
    """
    def __init__(self, n_features=4, embedding_dim=64, dropout=0.3, horizons=6):
        super().__init__()
        
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.horizons = horizons
        
        # Dilated convolutional layers with increasing dilation rates
        # This captures patterns at multiple temporal scales
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(n_features, embedding_dim//4, kernel_size=3, padding=1, dilation=1),
            nn.Conv1d(n_features, embedding_dim//4, kernel_size=3, padding=2, dilation=2),
            nn.Conv1d(n_features, embedding_dim//4, kernel_size=3, padding=4, dilation=4),
            nn.Conv1d(n_features, embedding_dim//4, kernel_size=3, padding=8, dilation=8),
        ])
        
        # Batch normalization for each dilated conv
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(embedding_dim//4) for _ in range(4)
        ])
        
        # Residual connection projection
        self.residual_proj = nn.Conv1d(n_features, embedding_dim, kernel_size=1)
        
        # Global pooling to aggregate temporal information
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Cyclic temporal feature embedder (day/hour)
        self.cyclic_embedder = nn.Linear(4, embedding_dim//4)
        
        # Fusion layer to combine temporal and cyclic features
        self.fusion = nn.Linear(embedding_dim + embedding_dim//4, embedding_dim)
        
        # Projection layers for multi-horizon prediction
        self.projections = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in range(horizons)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, dayofyear=None, hourofday=None):
        """
        Args:
            x: [batch_size, lookback, n_features] or [batch_size, n_features, lookback]
            dayofyear: Day of year for temporal encoding (optional)
            hourofday: Hour of day for temporal encoding (optional)
        
        Returns:
            temporal_embedding: [batch_size, embedding_dim]
            predictions: [batch_size, horizons] if training, else None
        """
        # Handle input shape: ensure [batch, n_features, lookback]
        if x.shape[1] != self.n_features:
            x = x.transpose(1, 2)
        
        batch_size = x.shape[0]
        
        # Apply dilated convolutions with different dilation rates
        dilated_outputs = []
        for conv, bn in zip(self.dilated_convs, self.batch_norms):
            out = conv(x)  # [batch, embedding_dim//4, lookback]
            out = bn(out)
            out = self.activation(out)
            out = self.dropout(out)
            dilated_outputs.append(out)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(dilated_outputs, dim=1)  # [batch, embedding_dim, lookback]
        
        # Residual connection
        residual = self.residual_proj(x)  # [batch, embedding_dim, lookback]
        combined = multi_scale + residual  # [batch, embedding_dim, lookback]
        
        # Global temporal pooling
        pooled = self.pool(combined).squeeze(-1)  # [batch, embedding_dim]
        
        # Add cyclic temporal features if provided
        if dayofyear is not None and hourofday is not None:
            # Convert to cyclic features
            day_sin, day_cos, hour_sin, hour_cos = date2cyclic_features(dayofyear, hourofday)
            
            # Create cyclic feature tensor
            cyclic_features = torch.tensor(
                np.stack([day_sin, day_cos, hour_sin, hour_cos], axis=-1),
                dtype=torch.float32,
                device=x.device
            )  # [batch, 4]
            
            # Embed cyclic features
            cyclic_embed = self.cyclic_embedder(cyclic_features)  # [batch, embedding_dim//4]
            
            # Concatenate and fuse
            fused = torch.cat([pooled, cyclic_embed], dim=-1)  # [batch, embedding_dim + embedding_dim//4]
            temporal_embedding = self.fusion(fused)  # [batch, embedding_dim]
        else:
            temporal_embedding = pooled
        
        temporal_embedding = self.dropout(temporal_embedding)
        
        # Multi-horizon predictions
        predictions = None
        if self.training:
            predictions = torch.stack([
                proj(temporal_embedding).squeeze(-1) for proj in self.projections
            ], dim=-1)  # [batch, horizons]
        
        return temporal_embedding, predictions


class StackedDilatedEncoder(nn.Module):
    """
    Deeper TCN with stacked dilated conv blocks.
    Better for capturing long-range temporal dependencies.
    
    Architecture:
    - Multiple residual blocks with dilated convolutions
    - Each block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm
    - Skip connections across blocks
    """
    def __init__(self, n_features=4, embedding_dim=64, dropout=0.3, 
                 horizons=6, num_blocks=3):
        super().__init__()
        
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.horizons = horizons
        self.num_blocks = num_blocks
        
        # Initial projection
        self.input_proj = nn.Conv1d(n_features, embedding_dim, kernel_size=1)
        
        # Stacked residual blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # Exponentially increasing dilation: 1, 2, 4, 8, ...
            self.blocks.append(self._make_residual_block(embedding_dim, dilation, dropout))
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Cyclic temporal feature embedder
        self.cyclic_embedder = nn.Linear(4, embedding_dim//4)
        
        # Fusion layer
        self.fusion = nn.Linear(embedding_dim + embedding_dim//4, embedding_dim)
        
        # Multi-horizon projections
        self.projections = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in range(horizons)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def _make_residual_block(self, channels, dilation, dropout):
        """Create a residual block with dilated convolutions."""
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, 
                     padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, 
                     padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
    
    def forward(self, x, dayofyear=None, hourofday=None):
        """
        Args:
            x: [batch_size, lookback, n_features] or [batch_size, n_features, lookback]
            dayofyear: Day of year for temporal encoding (optional)
            hourofday: Hour of day for temporal encoding (optional)
        
        Returns:
            temporal_embedding: [batch_size, embedding_dim]
            predictions: [batch_size, horizons] if training, else None
        """
        # Handle input shape
        if x.shape[1] != self.n_features:
            x = x.transpose(1, 2)
        
        # Initial projection
        out = self.input_proj(x)  # [batch, embedding_dim, lookback]
        
        # Apply residual blocks with skip connections
        for block in self.blocks:
            identity = out
            out = block(out) + identity  # Residual connection
            out = F.relu(out)
        
        # Global pooling
        pooled = self.pool(out).squeeze(-1)  # [batch, embedding_dim]
        
        # Add cyclic features if provided
        if dayofyear is not None and hourofday is not None:
            day_sin, day_cos, hour_sin, hour_cos = date2cyclic_features(dayofyear, hourofday)
            cyclic_features = torch.tensor(
                np.stack([day_sin, day_cos, hour_sin, hour_cos], axis=-1),
                dtype=torch.float32,
                device=x.device
            )
            cyclic_embed = self.cyclic_embedder(cyclic_features)
            fused = torch.cat([pooled, cyclic_embed], dim=-1)
            temporal_embedding = self.fusion(fused)
        else:
            temporal_embedding = pooled
        
        temporal_embedding = self.dropout(temporal_embedding)
        
        # Multi-horizon predictions
        predictions = None
        if self.training:
            predictions = torch.stack([
                proj(temporal_embedding).squeeze(-1) for proj in self.projections
            ], dim=-1)
        
        return temporal_embedding, predictions


class WaveNetEncoder(nn.Module):
    """
    WaveNet-style temporal encoder with gated activations.
    Uses causal dilated convolutions with gated linear units.
    
    Architecture:
    - Causal dilated convolutions (no future information leakage)
    - Gated activation: tanh(W_f * x) ⊙ sigmoid(W_g * x)
    - Residual and skip connections
    """
    def __init__(self, n_features=4, embedding_dim=64, dropout=0.3, 
                 horizons=6, num_layers=4):
        super().__init__()
        
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.horizons = horizons
        self.num_layers = num_layers
        
        # Initial causal convolution
        self.input_conv = nn.Conv1d(n_features, embedding_dim, kernel_size=1)
        
        # Dilated causal conv layers with gated activations
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            
            # Filter and gate convolutions (gated activation)
            self.filter_convs.append(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2,
                         padding=dilation, dilation=dilation)
            )
            self.gate_convs.append(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=2,
                         padding=dilation, dilation=dilation)
            )
            
            # Residual connection
            self.residual_convs.append(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
            )
            
            # Skip connection
            self.skip_convs.append(
                nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
            )
        
        # Final layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cyclic_embedder = nn.Linear(4, embedding_dim//4)
        self.fusion = nn.Linear(embedding_dim + embedding_dim//4, embedding_dim)
        
        self.projections = nn.ModuleList([
            nn.Linear(embedding_dim, 1) for _ in range(horizons)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, dayofyear=None, hourofday=None):
        """
        Args:
            x: [batch_size, lookback, n_features] or [batch_size, n_features, lookback]
            dayofyear: Day of year for temporal encoding (optional)
            hourofday: Hour of day for temporal encoding (optional)
        
        Returns:
            temporal_embedding: [batch_size, embedding_dim]
            predictions: [batch_size, horizons] if training, else None
        """
        # Handle input shape
        if x.shape[1] != self.n_features:
            x = x.transpose(1, 2)
        
        # Initial convolution
        out = self.input_conv(x)  # [batch, embedding_dim, lookback]
        
        # Accumulate skip connections
        skip_connections = []
        
        # Apply gated dilated convolutions
        for filter_conv, gate_conv, residual_conv, skip_conv in zip(
            self.filter_convs, self.gate_convs, self.residual_convs, self.skip_convs
        ):
            # Apply convolutions
            filter_out = filter_conv(out)
            gate_out = gate_conv(out)
            
            # Match temporal dimension by slicing to the smallest size
            min_len = min(filter_out.size(2), out.size(2))
            filter_out = filter_out[:, :, :min_len]
            gate_out = gate_out[:, :, :min_len]
            
            # Gated activation: tanh(filter) ⊙ sigmoid(gate)
            gated = torch.tanh(filter_out) * torch.sigmoid(gate_out)
            
            # Residual connection
            residual = residual_conv(gated)
            # Ensure residual matches out's temporal dimension
            out_sliced = out[:, :, :min_len]
            out = out_sliced + residual
            
            # Skip connection
            skip = skip_conv(gated)
            skip_connections.append(skip)
        
        # Combine skip connections
        skip_sum = sum(skip_connections)  # [batch, embedding_dim, lookback]
        
        # Global pooling
        pooled = self.pool(skip_sum).squeeze(-1)  # [batch, embedding_dim]
        
        # Add cyclic features
        if dayofyear is not None and hourofday is not None:
            day_sin, day_cos, hour_sin, hour_cos = date2cyclic_features(dayofyear, hourofday)
            cyclic_features = torch.tensor(
                np.stack([day_sin, day_cos, hour_sin, hour_cos], axis=-1),
                dtype=torch.float32,
                device=x.device
            )
            cyclic_embed = self.cyclic_embedder(cyclic_features)
            fused = torch.cat([pooled, cyclic_embed], dim=-1)
            temporal_embedding = self.fusion(fused)
        else:
            temporal_embedding = pooled
        
        temporal_embedding = self.dropout(temporal_embedding)
        
        # Multi-horizon predictions
        predictions = None
        if self.training:
            predictions = torch.stack([
                proj(temporal_embedding).squeeze(-1) for proj in self.projections
            ], dim=-1)
        
        return temporal_embedding, predictions

# Convenience function to get encoder by name
def get_temporal_encoder(encoder_type='dilated', **kwargs):
    """
    Factory function to create temporal encoder by name.
    
    Args:
        encoder_type: One of ['dilated', 'stacked', 'wavenet']
        **kwargs: Arguments for encoder initialization
    
    Returns:
        Temporal encoder instance
    """
    encoders = {
        'dilated': DilatedTemporalEncoder,
        'stacked': StackedDilatedEncoder,
        'wavenet': WaveNetEncoder,
    }
    
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available: {list(encoders.keys())}")
    
    return encoders[encoder_type](**kwargs)
