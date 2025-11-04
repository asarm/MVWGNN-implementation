import torch
from torch import nn
import torch.nn.functional as F
from temporalEncoder import TemporalEncoder
from adjLearner import DynamicAdjacencyLearner
from taskAwareAdjLearner import TaskAwareAdjacencyLearner
from positionalEncoding import SpatialPositionalEncoding
from directionalGAT import DirectionalGAT
from graphConv import DirectionalGCN
from multiScaleGNN import MultiScaleGNN, MultiScaleGNNStack
from cyclicEncoder import CyclicTemporalEncoding
from crossAttentionFusion import CrossAttentionFusion


class DDGNNWind(nn.Module):
    """
    V5: Enhanced DDGNN with flexible GNN architecture:
    - Attention-based temporal encoding
    - Task-aware dynamic adjacency learning (wind pattern based)
    - Cross-attention feature fusion (instead of simple addition)
    - Flexible GNN selection: 'gat', 'gcn', or 'multiscale'
    - Multi-scale GNN for capturing spatial patterns at different scales
    - Better gradient flow
    """
    
    def __init__(self, n_stations=50, hidden_dim=64, n_heads=4, 
                 seq_len=24, n_gnn_layers=2, input_dim=5, temporal_debug: bool = False,
                 use_task_aware_adj: bool = True, use_cross_attention: bool = True,
                 gnn_type: str = 'gcn'):  # 'gat', 'gcn', or 'multiscale'
        super().__init__()
        
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type.lower()
        self.input_dim = input_dim
        self.use_task_aware_adj = use_task_aware_adj
        self.use_cross_attention = use_cross_attention
        
        # Validate GNN type
        assert gnn_type in ['gat', 'gcn', 'multiscale'], \
            f"gnn_type must be 'gat', 'gcn', or 'multiscale', got '{gnn_type}'"
        
        # ========== ENCODERS ==========
        self.cyclic_encoder = CyclicTemporalEncoding(
            hidden_dim=hidden_dim,
            n_harmonics=3
        )

        # V2: Attention-based temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim, 
            seq_len=seq_len,
            debug=temporal_debug
        )
        
        self.spatial_encoder = SpatialPositionalEncoding(
            hidden_dim=hidden_dim,
            n_stations=n_stations
        )
        
        # ========== CROSS-ATTENTION FUSION ==========
        # V4: Cross-attention instead of simple addition
        if self.use_cross_attention:
            self.feature_fusion = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=0.1
            )
        
        # ========== DYNAMIC GRAPH LEARNING ==========
        # V4: Task-aware adjacency that uses wind patterns
        if self.use_task_aware_adj:
            self.adjacency_learner = TaskAwareAdjacencyLearner(
                hidden_dim=hidden_dim,
                n_stations=n_stations,
                embedding_dim=64,
                sparsify_mode='top_p',
                nucleus_p=0.6,  # 0.5 -> 0.9180
                temperature_scale=0.2
            )
        else:
            # Fallback to V3 adjacency learner
            self.adjacency_learner = DynamicAdjacencyLearner(
                hidden_dim=hidden_dim,
                n_stations=n_stations,
                embedding_dim=64,
                sparsify_mode='top_p',
                nucleus_p=0.6,  # 0.5 -> 0.9180
                temperature_scale=0.2
            )
        
        # ========== GNN LAYERS ==========
        # Flexible GNN architecture selection
        if self.gnn_type == 'multiscale':
            # Multi-Scale GNN: Captures patterns at different spatial scales
            def create_base_gnn():
                return DirectionalGCN(
                    in_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=0.3
                )
            
            self.gnn_layers = MultiScaleGNNStack(
                gnn_layer_fn=create_base_gnn,
                hidden_dim=hidden_dim,
                num_layers=n_gnn_layers,
                num_scales=2,  # 1-hop, 2-hop only (3-hop causes over-smoothing)
                dropout=0.3,
                fusion_type='weighted_sum'  # Fewer params than 'concat'
            )
            self.use_multiscale = True
            
        elif self.gnn_type == 'gcn':
            # GCN: Simpler, fewer parameters, better generalization
            self.gnn_layers = nn.ModuleList([
                DirectionalGCN(
                    in_features=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=0.3
                )
                for i in range(n_gnn_layers)
            ])
            self.use_multiscale = False
            
        elif self.gnn_type == 'gat':
            # GAT: More complex, attention-based (original)
            self.gnn_layers = nn.ModuleList([
                DirectionalGAT(hidden_dim, hidden_dim, n_heads=n_heads, dropout=0.3)
                for _ in range(n_gnn_layers)
            ])
            self.use_multiscale = False
        
        # Layer normalization (not needed for multiscale, it has internal norms)
        if not self.use_multiscale:
            self.gnn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(n_gnn_layers)
            ])
        
        self.gnn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)  # LayerNorm for better stability
            for _ in range(n_gnn_layers)
        ])
        
        # ========== DECODER ==========
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, historical_data, lat, lon, current_hour=None, day_of_year=None, 
                wind_direction_deg=None, positions=None):
        """
        Forward pass.
        
        Args:
            historical_data: (n_stations, seq_len, 5) or (B, n_stations, seq_len, 5)
            lat: (n_stations,) or (B, n_stations,)
            lon: (n_stations,) or (B, n_stations,)
            current_hour: scalar or (B,)
            day_of_year: scalar or (B,)
            wind_direction_deg: (n_stations,) or (B, n_stations,)
            positions: (n_stations, 2) or (B, n_stations, 2)
        
        Returns:
            predictions: (n_stations, 1) or (B, n_stations, 1)
        """
        
        device = historical_data.device
        is_batched = historical_data.dim() == 4

        # =========== STAGE 1: Temporal encoding (with attention) ===========
        temporal_features = self.temporal_encoder(historical_data)

        # Cyclic encoding
        if current_hour is None:
            current_hour = torch.tensor(0, device=device)
        if day_of_year is None:
            day_of_year = torch.tensor(1, device=device)

        if is_batched:
            cyclic_embed = self.cyclic_encoder(hour=current_hour, day_of_year=day_of_year)
        else:
            cyclic_embed = self.cyclic_encoder(hour=current_hour, day_of_year=day_of_year)

        # =========== STAGE 2: Spatial encoding ===========
        spatial_embed = self.spatial_encoder(lat, lon, wind_direction_deg)

        # =========== STAGE 3: Combine representations ===========
        if not is_batched:
            if cyclic_embed.dim() == 2 and cyclic_embed.shape[0] == 1:
                cyc = cyclic_embed.squeeze(0)
            else:
                cyc = cyclic_embed

            # V4: Use cross-attention fusion or fallback to addition
            if self.use_cross_attention:
                node_repr = self.feature_fusion(temporal_features, spatial_embed, cyc)
            else:
                node_repr = temporal_features + spatial_embed + cyc
            
            # =========== STAGE 4: Learn DYNAMIC adjacency ===========
            # V4: Task-aware adjacency with current wind speeds
            # Extract current wind speed (last timestep)
            current_wind = historical_data[:, -1, 0].unsqueeze(-1)  # (N, 1)
            
            if self.use_task_aware_adj:
                adjacency = self.adjacency_learner(
                    spatial_features=spatial_embed,
                    temporal_features=temporal_features,
                    positions=positions, 
                    wind_directions=wind_direction_deg,
                    current_wind_speeds=current_wind
                )
            else:
                adjacency = self.adjacency_learner(
                    spatial_features=spatial_embed,
                    temporal_features=temporal_features,
                    positions=positions, 
                    wind_directions=wind_direction_deg
                )

            # =========== STAGE 5: Apply GNN ===========
            if self.use_multiscale:
                # Multi-scale GNN: single forward pass handles everything
                gnn_output = self.gnn_layers(node_repr, adjacency)
            else:
                # Regular GNN: iterate through layers with residual connections
                gnn_output = node_repr
                for i, gnn_layer in enumerate(self.gnn_layers):
                    residual = gnn_output
                    gnn_output = self.gnn_norms[i](gnn_output)
                    gnn_output = gnn_layer(gnn_output, adjacency)
                    gnn_output = F.relu(gnn_output)
                    gnn_output = gnn_output + residual

            # =========== STAGE 6: Decode ===========
            prediction = self.decoder(gnn_output)
            return prediction
        else:
            # Batched case
            B, N, H = temporal_features.shape

            if cyclic_embed.dim() == 1:
                cyc = cyclic_embed.unsqueeze(0).expand(B, -1)
            else:
                cyc = cyclic_embed
            cyc = cyc.unsqueeze(1)

            # V4: Use cross-attention fusion or fallback to addition
            if self.use_cross_attention:
                node_repr = self.feature_fusion(temporal_features, spatial_embed, cyc)
            else:
                node_repr = temporal_features + spatial_embed + cyc

            # Dynamic adjacency (batched)
            # V4: Extract current wind speeds
            current_wind = historical_data[:, :, -1, 0].unsqueeze(-1)  # (B, N, 1)
            
            if self.use_task_aware_adj:
                adjacency = self.adjacency_learner(
                    spatial_features=spatial_embed,
                    temporal_features=temporal_features,
                    positions=positions, 
                    wind_directions=wind_direction_deg,
                    current_wind_speeds=current_wind
                )
            else:
                adjacency = self.adjacency_learner(
                    spatial_features=spatial_embed,
                    temporal_features=temporal_features,
                    positions=positions, 
                    wind_directions=wind_direction_deg
                )

            # GNN (batched)
            if self.use_multiscale:
                # Multi-scale GNN: single forward pass handles everything
                gnn_output = self.gnn_layers(node_repr, adjacency)
            else:
                # Regular GNN: iterate through layers with residual connections
                gnn_output = node_repr
                for i, gnn_layer in enumerate(self.gnn_layers):
                    residual = gnn_output
                    gnn_output = self.gnn_norms[i](gnn_output)
                    gnn_output = gnn_layer(gnn_output, adjacency)
                    gnn_output = F.relu(gnn_output)
                    gnn_output = gnn_output + residual

            # Decode (batched)
            B, N, H = gnn_output.shape
            gnn_flat = gnn_output.view(B * N, H)
            prediction_flat = self.decoder(gnn_flat)
            prediction = prediction_flat.view(B, N, 1)
            
            return prediction