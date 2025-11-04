import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from adjLearner import DynamicAdjacencyLearner

class TaskAwareAdjacencyLearner(DynamicAdjacencyLearner):
    """
    Task-aware adjacency learner that refines graph structure based on actual wind patterns.
    
    Key improvements over base DynamicAdjacencyLearner:
    - Takes current wind speeds as input
    - Learns wind similarity patterns to guide connectivity
    - Adaptively blends structure-based and task-based adjacencies
    - Better optimization for wind speed prediction task
    """
    
    def __init__(self, hidden_dim=64, n_stations=50, embedding_dim=32,
                 sparsify_mode: str = "top_p", nucleus_p: float = 0.5,
                 temperature_scale: float = 0.2):
        super().__init__(
            hidden_dim=hidden_dim,
            n_stations=n_stations,
            embedding_dim=embedding_dim,
            sparsify_mode=sparsify_mode,
            nucleus_p=nucleus_p,
            temperature_scale=temperature_scale
        )
        
        # ========== Wind Pattern Analyzer ==========
        # Learns to extract task-relevant features from wind data
        self.wind_pattern_encoder = nn.Sequential(
            nn.Linear(1, 32),  # Input: current wind speed
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, embedding_dim),
            nn.Tanh()
        )
        
        # ========== Adaptive Refinement Weight ==========
        # Learns when to trust structure vs. wind patterns
        self.refinement_weight = nn.Parameter(
            torch.tensor(0.3)  # Start with 30% wind-based, 70% structure-based
        )
        
        # ========== Wind Similarity Network ==========
        # Predicts edge weights based on wind pattern similarity
        self.wind_similarity_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spatial_features, temporal_features, 
                positions=None, wind_directions=None, current_wind_speeds=None):
        """
        Learn task-aware dynamic adjacency.
        
        Args:
            spatial_features: (N, H) or (B, N, H) - spatial embeddings
            temporal_features: (N, H) or (B, N, H) - temporal features
            positions: (N, 2) or (B, N, 2) - lat/lon
            wind_directions: (N,) or (B, N) - wind direction in degrees
            current_wind_speeds: (N, 1) or (B, N, 1) - current wind speed values
        
        Returns:
            adjacency: (N, N) or (B, N, N) - task-aware dynamic adjacency matrix
        """
        device = spatial_features.device
        is_batched = spatial_features.dim() == 3
        
        # ========== STEP 1: Get base adjacency from parent class ==========
        base_adj = super().forward(
            spatial_features=spatial_features,
            temporal_features=temporal_features,
            positions=positions,
            wind_directions=wind_directions
        )
        
        # If no wind speed data provided, return base adjacency
        if current_wind_speeds is None:
            return base_adj
        
        # ========== STEP 2: Encode wind patterns ==========
        # Extract task-relevant features from current wind speeds
        if current_wind_speeds.dim() == 2 and not is_batched:
            # (N, 1) -> (N, embedding_dim)
            wind_features = self.wind_pattern_encoder(current_wind_speeds)
        elif current_wind_speeds.dim() == 2 and is_batched:
            # (B, N) -> (B, N, 1)
            current_wind_speeds = current_wind_speeds.unsqueeze(-1)
            wind_features = self.wind_pattern_encoder(current_wind_speeds)
        else:
            # (B, N, 1) -> (B, N, embedding_dim)
            wind_features = self.wind_pattern_encoder(current_wind_speeds)
        
        # ========== STEP 3: Compute wind-based similarity ==========
        # Stations with similar wind patterns should be connected
        if is_batched:
            B, N, D = wind_features.shape
            
            # Pairwise similarity using dot product
            # (B, N, D) @ (B, D, N) -> (B, N, N)
            wind_similarity = torch.bmm(wind_features, wind_features.transpose(1, 2))
            wind_similarity = wind_similarity / math.sqrt(D)
            
            # Apply softmax with temperature for smooth gradients
            wind_adj = torch.softmax(wind_similarity / 0.1, dim=-1)
            
        else:
            N, D = wind_features.shape
            
            # (N, D) @ (D, N) -> (N, N)
            wind_similarity = torch.mm(wind_features, wind_features.T)
            wind_similarity = wind_similarity / math.sqrt(D)
            
            # Apply softmax
            wind_adj = torch.softmax(wind_similarity / 0.1, dim=-1)
        
        # ========== STEP 4: Refine with edge-wise predictions ==========
        # For each potential edge, predict its importance based on wind patterns
        if is_batched:
            B, N, _ = wind_features.shape
            
            # Create pairwise feature combinations
            # (B, N, 1, D) and (B, 1, N, D) -> (B, N, N, D*2)
            wind_i = wind_features.unsqueeze(2).expand(B, N, N, D)
            wind_j = wind_features.unsqueeze(1).expand(B, N, N, D)
            edge_features = torch.cat([wind_i, wind_j], dim=-1)
            
            # Predict edge weights
            edge_weights = self.wind_similarity_net(edge_features).squeeze(-1)
            # (B, N, N)
            
            # Combine with similarity-based adjacency
            wind_adj = wind_adj * edge_weights
            
        else:
            N, D = wind_features.shape
            
            # (N, 1, D) and (1, N, D) -> (N, N, D*2)
            wind_i = wind_features.unsqueeze(1).expand(N, N, D)
            wind_j = wind_features.unsqueeze(0).expand(N, N, D)
            edge_features = torch.cat([wind_i, wind_j], dim=-1)
            
            # Predict edge weights
            edge_weights = self.wind_similarity_net(edge_features).squeeze(-1)
            # (N, N)
            
            wind_adj = wind_adj * edge_weights
        
        # ========== STEP 5: Adaptive blending ==========
        # Learn optimal combination of structure-based and task-based adjacencies
        alpha = self.refinement_weight.sigmoid()  # Constrain to [0, 1]
        
        # Blend: (1-alpha) * base + alpha * wind_based
        # This allows the model to learn which adjacency is more useful
        refined_adj = (1 - alpha) * base_adj + alpha * wind_adj
        
        # ========== STEP 6: Normalize ==========
        # Ensure adjacency is properly normalized
        refined_adj = self._normalize_adjacency(refined_adj)
        
        return refined_adj
    
    def get_blend_weight(self):
        """Return current blend weight for monitoring."""
        return self.refinement_weight.sigmoid().item()
