import math
import torch
from torch import nn

class DirectionalGAT(nn.Module):
    """
    Graph Attention Network layer with directional/asymmetric properties.
    """
    
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        
        # Linear transformations for multi-head attention
        self.query = nn.Linear(in_dim, out_dim)
        self.key = nn.Linear(in_dim, out_dim)
        self.value = nn.Linear(in_dim, out_dim)
        
        # Attention coefficients
        self.attention_weights = nn.Parameter(torch.randn(1, n_heads, 1, self.head_dim))
        nn.init.xavier_uniform_(self.attention_weights)
        
        # Output projection
        self.output_linear = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, node_features, adjacency):
        """
        Args:
            node_features: (n_stations, in_dim)
            adjacency: (n_stations, n_stations) learned adjacency matrix
        
        Returns:
            output: (n_stations, out_dim)
        """
        
        n_nodes = node_features.shape[0]
        device = node_features.device
        
        # Linear transformations
        Q = self.query(node_features)  # (n_stations, out_dim)
        K = self.key(node_features)
        V = self.value(node_features)
        
        # Reshape for multi-head attention
        Q = Q.view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
        # Each: (n_heads, n_stations, head_dim)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (n_heads, n_stations, n_stations)
        
        # Mask with learned adjacency
        adjacency_mask = adjacency.unsqueeze(0)  # (1, n_stations, n_stations)
        scores = scores + torch.log(adjacency_mask + 1e-8)  # -inf where no edge
        
        # Attention weights
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        output = torch.matmul(attention, V)
        # (n_heads, n_stations, head_dim)
        
        # Reshape back
        output = output.transpose(0, 1).contiguous()
        output = output.view(n_nodes, self.out_dim)
        
        # Output projection
        output = self.output_linear(output)
        
        return output