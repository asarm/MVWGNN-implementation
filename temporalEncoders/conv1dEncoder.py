import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def date2cyclic_features(dayofyear, hourofday):
    """
    Günlük ve yıllık döngüsel özellikler oluşturur.
    dayofyear: 1-365 (tensor veya scalar)
    hourofday: 0-23 (tensor veya scalar)
    """
    # Tensor veya numpy array'e çevir
    if isinstance(dayofyear, (int, float)):
        dayofyear = np.array([dayofyear])
    if isinstance(hourofday, (int, float)):
        hourofday = np.array([hourofday])
    
    if torch.is_tensor(dayofyear):
        dayofyear = dayofyear.cpu().numpy()
    if torch.is_tensor(hourofday):
        hourofday = hourofday.cpu().numpy()
    
    day_rad = 2 * np.pi * (dayofyear - 1) / 365.0
    hour_rad = 2 * np.pi * hourofday / 24.0
    
    day_sin = np.sin(day_rad)
    day_cos = np.cos(day_rad)
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)
    
    return day_sin, day_cos, hour_sin, hour_cos

class Conv1DTemporalEncoder(nn.Module):
    def __init__(self, n_features=4, embedding_dim=64, dropout=0.3, horizons=6):
        super().__init__()
        
        # Multi-scale 1D convolutions (TCN-inspired ama çok basit)
        self.conv1 = nn.Conv1d(n_features, embedding_dim//2, 
                              kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_features, embedding_dim//2, 
                              kernel_size=7, padding=3)
                              
        # Global pooling + projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        self.cyclic_embedder = nn.Linear(4, embedding_dim//2)  # 4 döngüsel özellik için
        self.linear = nn.Linear(embedding_dim+embedding_dim//2, embedding_dim//2)
        self.projections = nn.ModuleList([nn.Linear(embedding_dim//2, 1) for _ in range(horizons)])  # Her horizon için ayrı linear katman
        
    def forward(self, x, dayofyear=None, hourofday=None):
        # x: [batch, lookback, n_features]
        x = x.transpose(1, 2)  # [batch, n_features, lookback]
        
        cyclical_embed = None
        if dayofyear is not None and hourofday is not None:
            day_sin, day_cos, hour_sin, hour_cos = date2cyclic_features(dayofyear, hourofday)
            
            # Stack ve tensor'e çevir: [batch, 4]
            cyclical_features = np.stack([day_sin, day_cos, hour_sin, hour_cos], axis=-1)
            cyclical_features = torch.tensor(cyclical_features, dtype=x.dtype, device=x.device)
            
            # Embedding katmanından geçir
            cyclical_embed = F.relu(self.cyclic_embedder(cyclical_features))  # [batch, embedding_dim//2]
            cyclical_embed = self.dropout(cyclical_embed)

        # Multi-scale feature extraction
        feat1 = F.relu(self.conv1(x))  # short-term patterns
        feat1 = self.dropout(feat1)
        feat2 = F.relu(self.conv2(x))  # longer-term patterns
        feat2 = self.dropout(feat2)
        
        # Combine
        combined = torch.cat([feat1, feat2], dim=1)  # [batch, embed_dim, lookback]
        
        # Global pooling
        pooled = self.pool(combined).squeeze(-1)  # [batch, embed_dim]
        pooled = self.dropout(pooled)
        
        # Döngüsel özellikleri ekle
        pooled = torch.cat([pooled, cyclical_embed], dim=-1)
            
        embed = F.relu(self.linear(pooled))  # [batch, embed_dim/2]
        embed = self.dropout(embed)
        
        # Projections for each horizon
        outputs = []
        for proj in self.projections:
            outputs.append(proj(embed))

        return torch.cat(outputs, dim=-1)  # [batch, horizons]