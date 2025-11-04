import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUConv1DEncoder(nn.Module):
    def __init__(self, n_features=4, lookback=96, gru_hidden_dim=32, embedding_dim=64):
        super().__init__()
        
        self.n_features = n_features
        self.gru_hidden_dim = gru_hidden_dim
        
        # Her feature için ayrı GRU encoder
        self.gru_encoders = nn.ModuleList([
            nn.GRU(input_size=1, hidden_size=gru_hidden_dim, batch_first=True)
            for _ in range(n_features)
        ])
        
        # Linear layer: GRU output'larından doğrudan embedding'e
        self.linear = nn.Linear(n_features * gru_hidden_dim * lookback, embedding_dim)
        
        # Final projection
        self.projection = nn.Linear(embedding_dim, 6)  # Output: 6 saatlik tahmin (rüzgar hızı)
    
    def forward(self, x):
        # x: [batch, lookback, n_features]
        batch_size, lookback, n_features = x.shape
        
        # Her feature'ı ayrı ayrı GRU'dan geçir
        gru_outputs = []
        for i in range(n_features):
            # Feature i'yi al: [batch, lookback, 1]
            feature_i = x[:, :, i:i+1]
            
            # GRU'dan geçir: output shape [batch, lookback, gru_hidden_dim]
            gru_out, _ = self.gru_encoders[i](feature_i)
            gru_outputs.append(gru_out)
        
        # Tüm GRU output'ları birleştir: [batch, lookback, n_features * gru_hidden_dim]
        combined_gru = torch.cat(gru_outputs, dim=2)
        
        # Flatten: [batch, lookback * n_features * gru_hidden_dim]
        flattened = combined_gru.reshape(batch_size, -1)
        
        # Linear layer ile embedding'e dönüştür
        embedding = F.relu(self.linear(flattened))  # [batch, embedding_dim]
        
        # Final projection
        return self.projection(embedding)  # [batch, 6]
