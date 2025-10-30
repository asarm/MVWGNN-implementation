import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    """
    V2: Attention-based temporal encoding
    
    Key improvements:
    - Learns which timesteps are important (not just last + mean)
    - Captures temporal dependencies
    - Better gradient flow
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, seq_len=24, debug: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.debug = bool(debug) and (os.getenv("TEMPORAL_DEBUG", "0") == "1")
        
        # ========== Multi-scale Temporal Convolutions ==========
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=1, padding=1),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=2, padding=2),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=4, padding=4),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, dilation=8, padding=8),
        ])
        
        # ========== TEMPORAL ATTENTION ==========
        # Learn importance weights for each timestep
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # ========== Recency Bias (inductive bias for time series) ==========
        # Give exponentially more weight to recent timesteps
        self.register_buffer('recency_bias', self._create_recency_bias(seq_len))
        
        # ========== Output projection ==========
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def _create_recency_bias(self, seq_len, decay=0.95):
        """Create exponential recency bias: more weight on recent timesteps"""
        # [0.95^23, 0.95^22, ..., 0.95^1, 0.95^0=1.0]
        bias = torch.tensor([decay ** (seq_len - i - 1) for i in range(seq_len)])
        return bias.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
        
    def forward(self, historical_data):
        """
        Args:
            historical_data: (n_stations, seq_len, input_dim) or
                             (batch, n_stations, seq_len, input_dim)

        Returns:
            temporal_features: (n_stations, hidden_dim) or
                (batch, n_stations, hidden_dim)
        """
        is_batched = historical_data.dim() == 4

        if is_batched:
            B, N, S, D = historical_data.shape
            x = historical_data.reshape(B * N, S, D).transpose(1, 2)
        else:
            x = historical_data.transpose(1, 2)
        
        # ========== Multi-scale convolution ==========
        multi_scale_features = []
        for conv in self.dilated_convs:
            features = F.relu(conv(x))
            multi_scale_features.append(features)
        
        x_conv = torch.cat(multi_scale_features, dim=1)
        # Shape: (B*N, hidden_dim*4, seq_len)

        # ========== ATTENTION-BASED TEMPORAL AGGREGATION ==========
        # Transpose for attention: (B*N, seq_len, hidden_dim*4)
        x_conv_t = x_conv.transpose(1, 2)
        
        # Compute attention scores for each timestep
        attention_logits = self.attention(x_conv_t)  # (B*N, seq_len, 1)
        attention_logits = attention_logits.squeeze(-1)  # (B*N, seq_len)
        
        # Apply recency bias (learnable through backprop via multiplicative effect)
        if is_batched:
            recency = self.recency_bias.expand(B * N, -1, -1).squeeze(1)
        else:
            N_unbatched = x_conv.shape[0]
            recency = self.recency_bias.expand(N_unbatched, -1, -1).squeeze(1)
        
        attention_logits = attention_logits + torch.log(recency + 1e-8)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=-1)  # (B*N, seq_len)
        
        if self.debug:
            # Print which timesteps get high attention
            avg_weights = attention_weights.mean(dim=0)
            print(f"[ENCODER] Attention weights (last 5 timesteps): {avg_weights[-5:].tolist()}")
        
        # Weighted sum over time
        # attention_weights: (B*N, seq_len) -> (B*N, 1, seq_len)
        # x_conv: (B*N, hidden_dim*4, seq_len)
        attention_weights = attention_weights.unsqueeze(1)
        temporal_features = torch.bmm(attention_weights, x_conv.transpose(1, 2))
        # (B*N, 1, hidden_dim*4)
        temporal_features = temporal_features.squeeze(1)  # (B*N, hidden_dim*4)
        
        if self.debug:
            print(f"[ENCODER] Output - mean: {temporal_features.mean():.4f}, "
                  f"std: {temporal_features.std():.4f}")
        
        # ========== Final projection ==========
        temporal_features = self.output_proj(temporal_features)

        if is_batched:
            temporal_features = temporal_features.view(B, N, self.hidden_dim)

        return temporal_features