import torch
from torch import nn
import torch.nn.functional as F
import math

# Import existing modules
from adjLearner import DynamicAdjacencyLearner
from taskAwareAdjLearner import TaskAwareAdjacencyLearner
from positionalEncoding import SpatialPositionalEncoding
from cyclicEncoder import CyclicTemporalEncoding
from crossAttentionFusion import CrossAttentionFusion


# ============================================================================
# PATCHTST TEMPORAL ENCODER (2023 - SOTA)
# ============================================================================

class PatchTSTEncoder(nn.Module):
    """
    PatchTST: Patching + Transformer for time series.
    
    State-of-the-art temporal encoder that:
    - Divides sequence into patches (reduces length)
    - Applies transformer to patches
    - Processes each channel independently
    
    For seq_len=24, patch_len=4 → 6 patches (4x reduction)
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, seq_len=24,
                 patch_len=4, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        
        # Channel-independent patching (process each feature separately)
        self.patch_embedding = nn.Linear(patch_len, hidden_dim)
        
        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, hidden_dim))
        
        # Transformer for each channel
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Aggregate across patches and channels
        self.channel_agg = nn.Linear(input_dim * hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: (N, L, D) or (B, N, L, D)
        Returns:
            (N, H) or (B, N, H)
        """
        is_batched = x.dim() == 4
        
        if is_batched:
            B, N, L, D = x.shape
            x = x.reshape(B * N, L, D)
        else:
            N, L, D = x.shape
            B = 1
        
        # Truncate to patch-aligned length
        L_patched = self.n_patches * self.patch_len
        x = x[:, :L_patched, :]
        
        # Reshape to patches: (B*N, D, n_patches, patch_len)
        x = x.reshape(B * N, L_patched, D)
        x = x.reshape(B * N, self.n_patches, self.patch_len, D)
        x = x.permute(0, 3, 1, 2)  # (B*N, D, n_patches, patch_len)
        
        # Process each channel independently
        outputs = []
        for d in range(D):
            x_d = x[:, d, :, :]  # (B*N, n_patches, patch_len)
            
            # Embed patches
            x_d = self.patch_embedding(x_d)  # (B*N, n_patches, H)
            
            # Add positional encoding
            x_d = x_d + self.pos_encoding
            
            # Apply transformer
            x_d = self.transformer(x_d)  # (B*N, n_patches, H)
            
            # Mean pool over patches
            x_d = x_d.mean(dim=1)  # (B*N, H)
            outputs.append(x_d)
        
        # Concatenate channels and aggregate
        x = torch.cat(outputs, dim=-1)  # (B*N, D*H)
        x = self.channel_agg(x)  # (B*N, H)
        x = self.norm(x)
        
        # Normalize output to prevent scale mismatch
        x = x / (x.std(dim=-1, keepdim=True).clamp(min=0.1))
        
        if is_batched:
            x = x.reshape(B, N, self.hidden_dim)
        
        return x


# ============================================================================
# GATV2 LAYER (2021 - Improved Graph Attention)
# ============================================================================

class GATv2Layer(nn.Module):
    """
    GATv2: Improved graph attention networks.
    
    Key improvement over GAT v1:
    - v1: a^T [Wh_i || Wh_j]  (static ranking)
    - v2: a^T LeakyReLU(W [h_i || h_j])  (dynamic)
    
    This fixes the static attention problem and makes attention more expressive.
    """
    
    def __init__(self, in_dim, out_dim, n_heads=4, dropout=0.1, concat=True):
        super().__init__()
        
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // n_heads
        self.concat = concat
        
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        
        # Linear transformations
        self.lin_src = nn.Linear(in_dim, out_dim)
        self.lin_dst = nn.Linear(in_dim, out_dim)
        
        # Attention mechanism (GATv2 style)
        self.attn_weight = nn.Parameter(torch.randn(1, n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_weight)
        
        # Output projection
        self.output_linear = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x, adj):
        """
        Args:
            x: (N, D) or (B, N, D) node features
            adj: (N, N) or (B, N, N) adjacency matrix
        Returns:
            (N, out_dim) or (B, N, out_dim)
        """
        is_batched = x.dim() == 3
        
        if is_batched:
            B, N, D = x.shape
        else:
            N, D = x.shape
        
        # Linear transformations
        h_src = self.lin_src(x)
        h_dst = self.lin_dst(x)
        
        # Reshape for multi-head attention
        if is_batched:
            h_src = h_src.view(B, N, self.n_heads, self.head_dim)
            h_dst = h_dst.view(B, N, self.n_heads, self.head_dim)
        else:
            h_src = h_src.view(N, self.n_heads, self.head_dim)
            h_dst = h_dst.view(N, self.n_heads, self.head_dim)
        
        # Compute attention scores (GATv2 way: dynamic)
        if is_batched:
            # (B, N, 1, H, D) + (B, 1, N, H, D) = (B, N, N, H, D)
            h_src_bc = h_src.unsqueeze(2)
            h_dst_bc = h_dst.unsqueeze(1)
            attn_for_edges = h_src_bc + h_dst_bc
            
            # Apply LeakyReLU then attention weight (KEY DIFFERENCE FROM GAT v1)
            attn_for_edges = self.leaky_relu(attn_for_edges)
            attn_scores = (attn_for_edges * self.attn_weight.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
            # (B, N, N, H)
            
        else:
            h_src_bc = h_src.unsqueeze(1)  # (N, 1, H, D)
            h_dst_bc = h_dst.unsqueeze(0)  # (1, N, H, D)
            attn_for_edges = h_src_bc + h_dst_bc
            
            attn_for_edges = self.leaky_relu(attn_for_edges)
            attn_scores = (attn_for_edges * self.attn_weight).sum(dim=-1)
            # (N, N, H)
        
        # Mask with adjacency
        if is_batched:
            adj_mask = adj.unsqueeze(-1)  # (B, N, N, 1)
        else:
            adj_mask = adj.unsqueeze(-1)  # (N, N, 1)
        
        attn_scores = attn_scores.masked_fill(adj_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-2 if is_batched else 0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        if is_batched:
            # (B, N, N, H) @ (B, N, H, D) -> (B, N, H, D)
            attn_weights_T = attn_weights.permute(0, 3, 1, 2)  # (B, H, N, N)
            h_dst_T = h_dst.permute(0, 2, 1, 3)  # (B, H, N, D)
            out = torch.matmul(attn_weights_T, h_dst_T)  # (B, H, N, D)
            out = out.permute(0, 2, 1, 3).contiguous()  # (B, N, H, D)
        else:
            attn_weights_T = attn_weights.permute(2, 0, 1)  # (H, N, N)
            h_dst_T = h_dst.permute(1, 0, 2)  # (H, N, D)
            out = torch.matmul(attn_weights_T, h_dst_T)  # (H, N, D)
            out = out.permute(1, 0, 2).contiguous()  # (N, H, D)
        
        # Concatenate or average heads
        if self.concat:
            if is_batched:
                out = out.view(B, N, self.out_dim)
            else:
                out = out.view(N, self.out_dim)
        else:
            out = out.mean(dim=-2 if is_batched else 1)
        
        # Output projection
        out = self.output_linear(out)
        
        return out


# ============================================================================
# MAIN MODEL WITH PATCHTST + GATV2
# ============================================================================

class DDGNNWind(nn.Module):
    """
    IMPROVED MODEL V5: Modern Architecture
    
    Key upgrades:
    1. PatchTST temporal encoder (2023 SOTA)
    2. GATv2 spatial GNN (2021 improved attention)
    3. Task-aware dynamic adjacency (kept from V4)
    4. Cross-attention fusion (kept from V4)
    5. Better regularization (BatchNorm, higher dropout)
    
    Expected improvement: -0.04 to -0.06 MAE
    """
    
    def __init__(self, n_stations=50, hidden_dim=64, n_heads=4, 
                 seq_len=24, n_gnn_layers=2, input_dim=5, 
                 temporal_debug: bool = False,
                 use_task_aware_adj: bool = True, 
                 use_cross_attention: bool = True,
                 dropout: float = 0.3,
                 patch_len: int = 4):  # NEW: patch length for PatchTST
        super().__init__()
        
        self.n_stations = n_stations
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.use_task_aware_adj = use_task_aware_adj
        self.use_cross_attention = use_cross_attention
        self.dropout = dropout
        
        print(f"[MODEL V5] Initializing with PatchTST + GATv2")
        print(f"  - Temporal: PatchTST (patch_len={patch_len})")
        print(f"  - Spatial: GATv2 (improved attention)")
        print(f"  - Hidden dim: {hidden_dim}")
        print(f"  - Dropout: {dropout}")
        
        # ========== CYCLIC ENCODER ==========
        self.cyclic_encoder = CyclicTemporalEncoding(
            hidden_dim=hidden_dim,
            n_harmonics=3
        )

        # ========== TEMPORAL ENCODER: PATCHTST (NEW!) ==========
        self.temporal_encoder = PatchTSTEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim, 
            seq_len=seq_len,
            patch_len=patch_len,
            n_heads=n_heads,
            n_layers=2,
            dropout=dropout
        )
        print(f"  ✓ PatchTST initialized: {seq_len} steps → {seq_len//patch_len} patches")
        
        # ========== SPATIAL ENCODER ==========
        self.spatial_encoder = SpatialPositionalEncoding(
            hidden_dim=hidden_dim,
            n_stations=n_stations
        )
        
        # ========== CROSS-ATTENTION FUSION ==========
        if self.use_cross_attention:
            self.feature_fusion = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                dropout=dropout
            )
        
        # ========== DYNAMIC GRAPH LEARNING ==========
        if self.use_task_aware_adj:
            self.adjacency_learner = TaskAwareAdjacencyLearner(
                hidden_dim=hidden_dim,
                n_stations=n_stations,
                embedding_dim=64,
                sparsify_mode='top_p',
                nucleus_p=0.85,  # Less aggressive than before (was 0.9)
                temperature_scale=0.3  # Softer softmax (was 0.2)
            )
            print(f"  ✓ Task-aware adjacency with nucleus_p=0.85")
        else:
            self.adjacency_learner = DynamicAdjacencyLearner(
                hidden_dim=hidden_dim,
                n_stations=n_stations,
                embedding_dim=64,
                sparsify_mode='top_p',
                nucleus_p=0.85,
                temperature_scale=0.3
            )
        
        # ========== GNN LAYERS: GATV2 (NEW!) ==========
        self.gnn_layers = nn.ModuleList([
            GATv2Layer(
                in_dim=hidden_dim, 
                out_dim=hidden_dim, 
                n_heads=n_heads, 
                dropout=dropout
            )
            for _ in range(n_gnn_layers)
        ])
        print(f"  ✓ {n_gnn_layers} GATv2 layers initialized")
        
        # BatchNorm for GNN layers (better regularization)
        self.gnn_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim)
            for _ in range(n_gnn_layers)
        ])
        
        # Additional dropout between GNN layers
        self.gnn_dropout = nn.Dropout(dropout)
        
        # ========== DECODER ==========
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        print(f"[MODEL V5] Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def forward(self, historical_data, lat, lon, current_hour=None, day_of_year=None, 
                wind_direction_deg=None, positions=None):
        """
        Forward pass with PatchTST + GATv2.
        
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

        # =========== STAGE 1: Temporal encoding (PatchTST) ===========
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

            if self.use_cross_attention:
                node_repr = self.feature_fusion(temporal_features, spatial_embed, cyc)
            else:
                node_repr = temporal_features + spatial_embed + cyc
            
            # =========== STAGE 4: Learn DYNAMIC adjacency ===========
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

            # =========== STAGE 5: Apply GATv2 with BatchNorm ===========
            gnn_output = node_repr
            for i, gnn_layer in enumerate(self.gnn_layers):
                residual = gnn_output
                
                # BatchNorm requires (B, C, *) format
                # For non-batched: (N, H) → (1, H, N) → BatchNorm → (N, H)
                gnn_output_bn = gnn_output.unsqueeze(0).transpose(1, 2)  # (1, H, N)
                gnn_output_bn = self.gnn_norms[i](gnn_output_bn)
                gnn_output = gnn_output_bn.transpose(1, 2).squeeze(0)  # (N, H)
                
                gnn_output = gnn_layer(gnn_output, adjacency)
                gnn_output = F.relu(gnn_output)
                gnn_output = self.gnn_dropout(gnn_output)
                gnn_output = gnn_output + residual

            # =========== STAGE 6: Decode ===========
            # BatchNorm in decoder requires proper batching
            gnn_output_unsq = gnn_output.unsqueeze(0)  # (1, N, H)
            prediction = self._decode_with_batchnorm(gnn_output_unsq)
            prediction = prediction.squeeze(0)  # (N, 1)
            
            return prediction
            
        else:
            # Batched case
            B, N, H = temporal_features.shape

            if cyclic_embed.dim() == 1:
                cyc = cyclic_embed.unsqueeze(0).expand(B, -1)
            else:
                cyc = cyclic_embed
            cyc = cyc.unsqueeze(1)

            if self.use_cross_attention:
                node_repr = self.feature_fusion(temporal_features, spatial_embed, cyc)
            else:
                node_repr = temporal_features + spatial_embed + cyc

            # Dynamic adjacency (batched)
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

            # GATv2 (batched) with BatchNorm
            gnn_output = node_repr
            for i, gnn_layer in enumerate(self.gnn_layers):
                residual = gnn_output
                
                # BatchNorm for batched input: (B, N, H) → (B, H, N) → BatchNorm → (B, N, H)
                gnn_output_bn = gnn_output.transpose(1, 2)  # (B, H, N)
                gnn_output_bn = self.gnn_norms[i](gnn_output_bn)
                gnn_output = gnn_output_bn.transpose(1, 2)  # (B, N, H)
                
                gnn_output = gnn_layer(gnn_output, adjacency)
                gnn_output = F.relu(gnn_output)
                gnn_output = self.gnn_dropout(gnn_output)
                gnn_output = gnn_output + residual

            # Decode (batched)
            prediction = self._decode_with_batchnorm(gnn_output)
            
            return prediction
    
    def _decode_with_batchnorm(self, gnn_output):
        """
        Helper to apply decoder with BatchNorm properly.
        
        Args:
            gnn_output: (B, N, H)
        Returns:
            prediction: (B, N, 1)
        """
        B, N, H = gnn_output.shape
        
        # Flatten: (B, N, H) → (B*N, H)
        gnn_flat = gnn_output.reshape(B * N, H)
        
        # Apply decoder layers sequentially
        x = self.decoder[0](gnn_flat)       # Linear: (B*N, H) → (B*N, H//2)
        x = self.decoder[1](x)              # BatchNorm: (B*N, H//2)
        x = self.decoder[2](x)              # Dropout
        x = self.decoder[3](x)              # ReLU
        x = self.decoder[4](x)              # Dropout
        prediction_flat = self.decoder[5](x)  # Linear: (B*N, H//2) → (B*N, 1)
        
        # Reshape: (B*N, 1) → (B, N, 1)
        prediction = prediction_flat.view(B, N, 1)
        
        return prediction