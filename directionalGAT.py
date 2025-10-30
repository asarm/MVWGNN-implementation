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
            node_features: (n_stations, in_dim) or (batch, n_stations, in_dim)
            adjacency: (n_stations, n_stations) or (batch, n_stations, n_stations) learned adjacency matrix

        Returns:
            output: (n_stations, out_dim) or (batch, n_stations, out_dim)
        """
        device = node_features.device

        # Determine if batched
        is_batched = node_features.dim() == 3

        if not is_batched:
            # (N, in_dim) -> (N, out_dim)
            n_nodes = node_features.shape[0]

            Q = self.query(node_features)  # (N, out_dim)
            K = self.key(node_features)
            V = self.value(node_features)

            Q = Q.view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
            K = K.view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
            V = V.view(n_nodes, self.n_heads, self.head_dim).transpose(0, 1)
            # (n_heads, N, head_dim)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # (n_heads, N, N)

            adjacency_mask = adjacency.unsqueeze(0)  # (1, N, N)
            scores = scores + torch.log(adjacency_mask + 1e-8)

            attention = torch.softmax(scores, dim=-1)
            attention = self.dropout(attention)

            output = torch.matmul(attention, V)
            # (n_heads, N, head_dim)

            output = output.transpose(0, 1).contiguous()
            output = output.view(n_nodes, self.out_dim)

            output = self.output_linear(output)
            return output
        else:
            # Batched: (B, N, in_dim) and adjacency (B, N, N) or (N, N)
            B, N, _ = node_features.shape

            Q = self.query(node_features)  # (B, N, out_dim)
            K = self.key(node_features)
            V = self.value(node_features)

            # reshape to (B, n_heads, N, head_dim)
            Q = Q.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            V = V.view(B, N, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            # (B, n_heads, N, head_dim)

            # Compute scores: (B, n_heads, N, N)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if adjacency.dim() == 2:
                adjacency_mask = adjacency.unsqueeze(0)
            else:
                adjacency_mask = adjacency
            # adjacency_mask: (B, N, N)
            scores = scores + torch.log(adjacency_mask.unsqueeze(1) + 1e-8)

            attention = torch.softmax(scores, dim=-1)
            attention = self.dropout(attention)

            # Apply attention to V -> (B, n_heads, N, head_dim)
            output = torch.matmul(attention, V)

            # reshape back to (B, N, out_dim)
            output = output.permute(0, 2, 1, 3).contiguous()
            output = output.view(B, N, self.out_dim)

            output = self.output_linear(output)
            return output