import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedGRUEncoder(nn.Module):
    def __init__(self, n_features=4, lookback=96, gru_hidden_dim=32, embedding_dim=64):
        super().__init__()
        
        self.n_features = n_features
        self.gru_hidden_dim = gru_hidden_dim
        
        # Tüm feature'lar için paylaşılan tek bir GRU encoder
        self.shared_gru = nn.GRU(input_size=n_features, hidden_size=gru_hidden_dim, batch_first=True)
        
        # Linear layer: GRU output'larından doğrudan embedding'e
        self.linear = nn.Linear(gru_hidden_dim * lookback, embedding_dim)
        
        # Final projection
        self.projection = nn.Linear(embedding_dim, 6)  # Output: 6 saatlik tahmin (rüzgar hızı)
    
    def forward(self, x):
        # x: [batch, lookback, n_features]
        batch_size, lookback, n_features = x.shape
        
        # Paylaşılan GRU'dan geçir: output shape [batch, lookback, gru_hidden_dim]
        gru_out, _ = self.shared_gru(x)
        
        # Flatten: [batch, lookback * gru_hidden_dim]
        flattened = gru_out.reshape(batch_size, -1)
        
        # Linear layer ile embedding'e dönüştür
        embedding = F.relu(self.linear(flattened))  # [batch, embedding_dim]
        
        # Final projection
        return self.projection(embedding)  # [batch, 6]
