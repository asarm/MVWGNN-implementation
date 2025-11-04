import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionTemporalEncoder(nn.Module):
    """
    Simplified attention-based temporal encoder with regularization.
    - Multi-head attention for temporal dependencies.
    - Dropout and layer norm for stability.
    - Lower capacity to prevent overfitting.
    """
    
    def __init__(self, n_features=4, lookback=24, embedding_dim=32, n_heads=4, dropout=0.3):
        super().__init__()
        self.lookback = lookback
        self.embedding_dim = embedding_dim
        
        # Input projection
        self.input_proj = nn.Linear(n_features, embedding_dim)
        
        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, lookback, embedding_dim))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, 6)  # 6 saatlik tahmin
    
    def forward(self, x):
        # x: (batch, lookback, n_features)
        batch_size = x.shape[0]
        
        # Project input
        x = self.input_proj(x)  # (batch, lookback, embedding_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # Aggregate (mean pooling over time)
        x = x.mean(dim=1)  # (batch, embedding_dim)
        
        # Output
        out = self.output_proj(x)  # (batch, 1)
        return out  # (batch, 6)