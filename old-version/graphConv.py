"""
Graph Convolutional Network (GCN) layer for spatial modeling.

GCN is simpler than GAT and often generalizes better because:
- No learnable attention weights (fewer parameters)
- Uses normalized adjacency for aggregation
- More stable gradients
- Less prone to overfitting

For weather forecasting, spatial relationships are relatively stable,
so GCN's simpler aggregation may work better than GAT's dynamic attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer.
    
    H' = σ(D^(-1/2) A D^(-1/2) H W)
    
    Where:
    - A: Adjacency matrix (with self-loops)
    - D: Degree matrix
    - H: Node features
    - W: Learnable weight matrix
    - σ: Activation function
    """
    
    def __init__(self, in_features, out_features, dropout=0.3, use_bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable transformation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """Forward pass.
        
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        # Apply dropout to input features
        x = self.dropout(x)
        
        # Linear transformation: H * W
        support = torch.matmul(x, self.weight)  # [B, N, out_features]
        
        # Normalize adjacency matrix (symmetric normalization)
        adj_normalized = self.normalize_adjacency(adj)
        
        # Graph convolution: A_norm * H * W
        output = torch.bmm(adj_normalized, support)  # [B, N, out_features]
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def normalize_adjacency(self, adj):
        """Symmetric normalization of adjacency matrix.
        
        A_norm = D^(-1/2) * A * D^(-1/2)
        
        Args:
            adj: [batch_size, num_nodes, num_nodes]
        
        Returns:
            Normalized adjacency [batch_size, num_nodes, num_nodes]
        """
        # Add self-loops
        batch_size, num_nodes, _ = adj.shape
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self_loops = adj + identity
        
        # Compute degree matrix
        degree = adj_with_self_loops.sum(dim=-1)  # [B, N]
        
        # D^(-1/2)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # Create diagonal matrix D^(-1/2)
        degree_matrix_inv_sqrt = torch.diag_embed(degree_inv_sqrt)  # [B, N, N]
        
        # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
        adj_normalized = torch.bmm(
            torch.bmm(degree_matrix_inv_sqrt, adj_with_self_loops),
            degree_matrix_inv_sqrt
        )
        
        return adj_normalized


class DirectionalGCN(nn.Module):
    """Multi-layer GCN for directional wind modeling.
    
    Simpler alternative to DirectionalGAT with fewer parameters.
    Better for avoiding overfitting on spatial patterns.
    """
    
    def __init__(self, in_features, hidden_dim, num_layers=2, dropout=0.3):
        super(DirectionalGCN, self).__init__()
        self.num_layers = num_layers
        
        self.gcn_layers = nn.ModuleList()
        
        # First layer
        self.gcn_layers.append(GCNLayer(in_features, hidden_dim, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim, dropout=dropout))
        
        self.activation = nn.ELU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, adj):
        """Forward pass through multiple GCN layers.
        
        Args:
            x: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
            Updated node features [batch_size, num_nodes, hidden_dim]
        """
        h = x
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            h_new = gcn_layer(h, adj)
            h_new = self.activation(h_new)
            
            # Residual connection (except first layer)
            if i > 0 and h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new
            
            # Layer normalization
            h = self.layer_norm(h)
        
        return h


if __name__ == "__main__":
    # Quick test
    batch_size = 4
    num_nodes = 10
    in_features = 128
    hidden_dim = 128
    
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.rand(batch_size, num_nodes, num_nodes)
    adj = (adj + adj.transpose(1, 2)) / 2  # Make symmetric
    
    # Test single GCN layer
    gcn = GCNLayer(in_features, hidden_dim, dropout=0.3)
    out = gcn(x, adj)
    print(f"GCN Layer - Input: {x.shape}, Output: {out.shape}")
    
    # Test multi-layer DirectionalGCN
    model = DirectionalGCN(in_features, hidden_dim, num_layers=2, dropout=0.3)
    out = model(x, adj)
    print(f"DirectionalGCN - Input: {x.shape}, Output: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
