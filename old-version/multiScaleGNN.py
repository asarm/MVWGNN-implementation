"""
Multi-Scale Graph Neural Network for capturing spatial patterns at multiple scales.

This module implements a multi-hop GNN that aggregates information from:
- 1-hop neighbors (local patterns, 50-100km)
- 2-hop neighbors (regional patterns, 100-300km)
- 3-hop neighbors (synoptic patterns, 300-500km)

Approach: Shared weights with scale-specific modulation (Option 2)
- Uses a single GNN for all scales (parameter efficient)
- Applies scale-specific transformations
- Fuses multi-scale features adaptively
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleGNN(nn.Module):
    """Multi-Scale GNN with shared weights and scale-specific modulation.
    
    Architecture:
    1. Compute multi-hop adjacency matrices (A, A², A³)
    2. Apply same GNN to each scale
    3. Apply scale-specific modulation
    4. Fuse features with learnable weights
    
    Args:
        gnn_layer: Base GNN module (DirectionalGCN or DirectionalGAT)
        hidden_dim: Hidden dimension size
        num_scales: Number of scales to use (default: 3 for 1-hop, 2-hop, 3-hop)
        dropout: Dropout rate
        fusion_type: How to fuse scales ('concat', 'weighted_sum', 'attention')
    """
    
    def __init__(self, gnn_layer, hidden_dim, num_scales=3, dropout=0.3, fusion_type='concat'):
        super(MultiScaleGNN, self).__init__()
        
        self.gnn_layer = gnn_layer
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.fusion_type = fusion_type
        
        # Scale-specific modulation (lightweight transformation for each scale)
        self.scale_modulators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_scales)
        ])
        
        # Learnable scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # Fusion layer
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_scales, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.ELU()
            )
        elif fusion_type == 'attention':
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
        else:  # weighted_sum
            self.fusion = nn.Identity()
    
    def compute_multihop_adjacency(self, adj, max_hops=3):
        """Compute multi-hop adjacency matrices.
        
        Args:
            adj: Base adjacency matrix [batch, nodes, nodes]
            max_hops: Maximum number of hops
            
        Returns:
            List of adjacency matrices [A, A², A³, ...]
        """
        adj_list = [adj]
        current_adj = adj
        
        for hop in range(1, max_hops):
            # A^(k+1) = A^k × A
            current_adj = torch.bmm(current_adj, adj)
            
            # Optional: Normalize to prevent values from exploding
            # Binary adjacency: set non-zero to 1
            current_adj = (current_adj > 0).float()
            
            adj_list.append(current_adj)
        
        return adj_list
    
    def forward(self, x, adj):
        """Forward pass through multi-scale GNN.
        
        Args:
            x: Node features [batch_size, num_nodes, hidden_dim]
            adj: Base adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Multi-scale aggregated features [batch_size, num_nodes, hidden_dim]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Compute multi-hop adjacency matrices
        adj_scales = self.compute_multihop_adjacency(adj, max_hops=self.num_scales)
        
        # Apply GNN at each scale
        scale_features = []
        for i, adj_scale in enumerate(adj_scales):
            # Same GNN, different adjacency
            h_scale = self.gnn_layer(x, adj_scale)
            
            # Scale-specific modulation
            h_scale = self.scale_modulators[i](h_scale)
            
            scale_features.append(h_scale)
        
        # Fuse multi-scale features
        if self.fusion_type == 'concat':
            # Concatenate and project
            h_fused = torch.cat(scale_features, dim=-1)  # [B, N, hidden_dim * num_scales]
            output = self.fusion(h_fused)  # [B, N, hidden_dim]
            
        elif self.fusion_type == 'attention':
            # Stack scales and apply cross-attention
            h_stacked = torch.stack(scale_features, dim=1)  # [B, num_scales, N, hidden_dim]
            
            # Reshape for attention: [B*N, num_scales, hidden_dim]
            h_stacked = h_stacked.transpose(1, 2).reshape(batch_size * num_nodes, self.num_scales, self.hidden_dim)
            
            # Self-attention across scales
            h_attended, _ = self.scale_attention(h_stacked, h_stacked, h_stacked)
            
            # Average across scales and reshape back
            h_fused = h_attended.mean(dim=1).reshape(batch_size, num_nodes, self.hidden_dim)
            output = self.fusion(h_fused)
            
        else:  # weighted_sum
            # Learnable weighted combination
            weights = F.softmax(self.scale_weights, dim=0)
            output = sum(w * h for w, h in zip(weights, scale_features))
        
        return output


class MultiScaleGNNStack(nn.Module):
    """Stack of multiple Multi-Scale GNN layers with residual connections.
    
    This is a drop-in replacement for the ModuleList of GNN layers in the main model.
    """
    
    def __init__(self, gnn_layer_fn, hidden_dim, num_layers=2, num_scales=3, 
                 dropout=0.3, fusion_type='concat'):
        super(MultiScaleGNNStack, self).__init__()
        
        self.num_layers = num_layers
        
        # Create multi-scale GNN layers
        self.ms_gnn_layers = nn.ModuleList([
            MultiScaleGNN(
                gnn_layer=gnn_layer_fn(),
                hidden_dim=hidden_dim,
                num_scales=num_scales,
                dropout=dropout,
                fusion_type=fusion_type
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, adj):
        """Forward pass through stacked multi-scale GNN layers.
        
        Args:
            x: Node features [batch_size, num_nodes, hidden_dim]
            adj: Base adjacency matrix [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, hidden_dim]
        """
        h = x
        
        for i, (ms_gnn, norm) in enumerate(zip(self.ms_gnn_layers, self.layer_norms)):
            # Multi-scale GNN
            h_new = ms_gnn(h, adj)
            
            # Residual connection
            h = h + h_new
            
            # Layer normalization
            h = norm(h)
        
        return h


if __name__ == "__main__":
    # Quick test
    from graphConv import DirectionalGCN
    
    batch_size = 4
    num_nodes = 10
    hidden_dim = 128
    
    x = torch.randn(batch_size, num_nodes, hidden_dim)
    adj = torch.rand(batch_size, num_nodes, num_nodes)
    adj = (adj > 0.7).float()  # Sparse adjacency
    adj = (adj + adj.transpose(1, 2)) / 2  # Symmetric
    
    # Create base GNN layer
    def create_gcn():
        return DirectionalGCN(
            in_features=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            dropout=0.3
        )
    
    # Test Multi-Scale GNN
    print("Testing MultiScaleGNN...")
    ms_gnn = MultiScaleGNN(
        gnn_layer=create_gcn(),
        hidden_dim=hidden_dim,
        num_scales=3,
        fusion_type='concat'
    )
    
    out = ms_gnn(x, adj)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in ms_gnn.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test MultiScaleGNNStack
    print("\nTesting MultiScaleGNNStack...")
    ms_stack = MultiScaleGNNStack(
        gnn_layer_fn=create_gcn,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_scales=3,
        fusion_type='concat'
    )
    
    out = ms_stack(x, adj)
    print(f"Input: {x.shape}, Output: {out.shape}")
    
    stack_params = sum(p.numel() for p in ms_stack.parameters())
    print(f"Stack parameters: {stack_params:,}")
    
    print("\n✓ All tests passed!")
