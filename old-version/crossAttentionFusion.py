import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based feature fusion module.
    
    Instead of simple addition, this module learns to:
    - Let temporal features attend to spatial patterns
    - Let spatial features attend to temporal patterns
    - Combine cyclic features with context-aware weights
    
    This creates richer, more adaptive feature representations.
    """
    
    def __init__(self, hidden_dim=64, n_heads=4, dropout=0.1):
        """
        Args:
            hidden_dim: Dimension of input features
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        # ========== Temporal-to-Spatial Cross-Attention ==========
        # Temporal features query spatial patterns
        self.temporal_query = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_key = nn.Linear(hidden_dim, hidden_dim)
        self.spatial_value = nn.Linear(hidden_dim, hidden_dim)
        
        # ========== Spatial-to-Temporal Cross-Attention ==========
        # Spatial features query temporal patterns
        self.spatial_query = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_key = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_value = nn.Linear(hidden_dim, hidden_dim)
        
        # ========== Cyclic Feature Integration ==========
        # Learn context-dependent weights for cyclic features
        self.cyclic_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # ========== Output Projection ==========
        # Combine all attended features
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _split_heads(self, x):
        """
        Split features into multiple heads.
        
        Args:
            x: (B, N, H) or (N, H)
        Returns:
            (B, n_heads, N, head_dim) or (n_heads, N, head_dim)
        """
        is_batched = x.dim() == 3
        
        if is_batched:
            B, N, H = x.shape
            # (B, N, H) -> (B, N, n_heads, head_dim) -> (B, n_heads, N, head_dim)
            x = x.view(B, N, self.n_heads, self.head_dim)
            x = x.transpose(1, 2)
        else:
            N, H = x.shape
            # (N, H) -> (N, n_heads, head_dim) -> (n_heads, N, head_dim)
            x = x.view(N, self.n_heads, self.head_dim)
            x = x.transpose(0, 1)
        
        return x
    
    def _combine_heads(self, x):
        """
        Combine multiple heads back.
        
        Args:
            x: (B, n_heads, N, head_dim) or (n_heads, N, head_dim)
        Returns:
            (B, N, H) or (N, H)
        """
        is_batched = x.dim() == 4
        
        if is_batched:
            B, n_heads, N, head_dim = x.shape
            # (B, n_heads, N, head_dim) -> (B, N, n_heads, head_dim) -> (B, N, H)
            x = x.transpose(1, 2).contiguous()
            x = x.view(B, N, self.hidden_dim)
        else:
            n_heads, N, head_dim = x.shape
            # (n_heads, N, head_dim) -> (N, n_heads, head_dim) -> (N, H)
            x = x.transpose(0, 1).contiguous()
            x = x.view(N, self.hidden_dim)
        
        return x
    
    def _multi_head_attention(self, query, key, value):
        """
        Compute multi-head attention.
        
        Args:
            query: (B, N, H) or (N, H)
            key: (B, N, H) or (N, H)
            value: (B, N, H) or (N, H)
        Returns:
            output: (B, N, H) or (N, H)
            attention_weights: (B, n_heads, N, N) or (n_heads, N, N)
        """
        # Project and split heads
        Q = self._split_heads(query)  # (B, n_heads, N, head_dim)
        K = self._split_heads(key)
        V = self._split_heads(value)
        
        # Scaled dot-product attention
        # (B, n_heads, N, head_dim) @ (B, n_heads, head_dim, N) -> (B, n_heads, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (B, n_heads, N, N) @ (B, n_heads, N, head_dim) -> (B, n_heads, N, head_dim)
        output = torch.matmul(attention_weights, V)
        
        # Combine heads
        output = self._combine_heads(output)  # (B, N, H)
        
        return output, attention_weights
    
    def forward(self, temporal_features, spatial_features, cyclic_features):
        """
        Fuse temporal, spatial, and cyclic features using cross-attention.
        
        Args:
            temporal_features: (N, H) or (B, N, H) - from temporal encoder
            spatial_features: (N, H) or (B, N, H) - from spatial encoder
            cyclic_features: (H,) or (B, H) or (B, 1, H) - from cyclic encoder
        
        Returns:
            fused_features: (N, H) or (B, N, H) - combined representation
        """
        is_batched = temporal_features.dim() == 3
        
        # ========== Handle cyclic features shape ==========
        if cyclic_features.dim() == 1:
            # (H,) -> (1, H) -> broadcast to (N, H) or (B, N, H)
            if is_batched:
                B, N, H = temporal_features.shape
                cyclic_features = cyclic_features.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
            else:
                N, H = temporal_features.shape
                cyclic_features = cyclic_features.unsqueeze(0).expand(N, -1)
        elif cyclic_features.dim() == 2:
            # (B, H) -> (B, N, H)
            if is_batched:
                B, N, H = temporal_features.shape
                cyclic_features = cyclic_features.unsqueeze(1).expand(-1, N, -1)
            else:
                # (1, H) -> (N, H)
                N, H = temporal_features.shape
                cyclic_features = cyclic_features.expand(N, -1)
        elif cyclic_features.dim() == 3:
            # Already (B, N, H) or (B, 1, H)
            if is_batched and cyclic_features.shape[1] == 1:
                B, N, H = temporal_features.shape
                cyclic_features = cyclic_features.expand(-1, N, -1)
        
        # ========== Cross-Attention 1: Temporal queries Spatial ==========
        # Temporal features attend to spatial patterns
        Q_temp = self.temporal_query(temporal_features)
        K_spat = self.spatial_key(spatial_features)
        V_spat = self.spatial_value(spatial_features)
        
        temporal_aware, _ = self._multi_head_attention(Q_temp, K_spat, V_spat)
        # temporal_aware: temporal features enhanced with spatial context
        
        # ========== Cross-Attention 2: Spatial queries Temporal ==========
        # Spatial features attend to temporal patterns
        Q_spat = self.spatial_query(spatial_features)
        K_temp = self.temporal_key(temporal_features)
        V_temp = self.temporal_value(temporal_features)
        
        spatial_aware, _ = self._multi_head_attention(Q_spat, K_temp, V_temp)
        # spatial_aware: spatial features enhanced with temporal context
        
        # ========== Gated Cyclic Integration ==========
        # Learn context-dependent weights for cyclic features
        # Concatenate temporal and spatial context
        context = torch.cat([temporal_aware, spatial_aware], dim=-1)
        cyclic_gate = self.cyclic_gate(context)  # (B, N, H) or (N, H)
        
        # Apply gating to cyclic features
        cyclic_aware = cyclic_gate * cyclic_features
        
        # ========== Combine All Features ==========
        # Concatenate and project
        combined = torch.cat([temporal_aware, spatial_aware, cyclic_aware], dim=-1)
        # (B, N, H*3) or (N, H*3)
        
        fused = self.output_proj(combined)
        # (B, N, H) or (N, H)
        
        # ========== Residual Connection ==========
        # Add original temporal features (main information carrier)
        fused = fused + temporal_features
        fused = self.layer_norm(fused)
        
        return fused
    
    def get_attention_maps(self, temporal_features, spatial_features, cyclic_features):
        """
        Get attention maps for visualization.
        
        Returns:
            dict with attention weights from both cross-attention operations
        """
        is_batched = temporal_features.dim() == 3
        
        # Handle cyclic features
        if cyclic_features.dim() == 1:
            if is_batched:
                B, N, H = temporal_features.shape
                cyclic_features = cyclic_features.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
            else:
                N, H = temporal_features.shape
                cyclic_features = cyclic_features.unsqueeze(0).expand(N, -1)
        elif cyclic_features.dim() == 2 and is_batched:
            B, N, H = temporal_features.shape
            cyclic_features = cyclic_features.unsqueeze(1).expand(-1, N, -1)
        
        # Get attention maps
        Q_temp = self.temporal_query(temporal_features)
        K_spat = self.spatial_key(spatial_features)
        V_spat = self.spatial_value(spatial_features)
        _, temp_to_spat_attn = self._multi_head_attention(Q_temp, K_spat, V_spat)
        
        Q_spat = self.spatial_query(spatial_features)
        K_temp = self.temporal_key(temporal_features)
        V_temp = self.temporal_value(temporal_features)
        _, spat_to_temp_attn = self._multi_head_attention(Q_spat, K_temp, V_temp)
        
        return {
            'temporal_to_spatial': temp_to_spat_attn,
            'spatial_to_temporal': spat_to_temp_attn
        }
