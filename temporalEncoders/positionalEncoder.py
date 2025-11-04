import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def coords2cyclic_features(latitude, longitude):
    """
    Converts latitude and longitude to cyclic features using sin and cos.
    
    Args:
        latitude: Latitude values in degrees, range [-90, 90] (tensor, array, or scalar)
        longitude: Longitude values in degrees, range [-180, 180] (tensor, array, or scalar)
    
    Returns:
        lat_sin, lat_cos, lon_sin, lon_cos: Cyclic features for latitude and longitude
    """
    # Convert to numpy array if needed
    if isinstance(latitude, (int, float)):
        latitude = np.array([latitude])
    if isinstance(longitude, (int, float)):
        longitude = np.array([longitude])
    
    if torch.is_tensor(latitude):
        latitude = latitude.cpu().numpy()
    if torch.is_tensor(longitude):
        longitude = longitude.cpu().numpy()
    
    # Normalize latitude from [-90, 90] to [0, 2π]
    lat_rad = np.pi * (latitude + 90) / 180.0
    
    # Normalize longitude from [-180, 180] to [0, 2π]
    lon_rad = np.pi * (longitude + 180) / 180.0
    
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)
    lon_sin = np.sin(lon_rad)
    lon_cos = np.cos(lon_rad)
    
    return lat_sin, lat_cos, lon_sin, lon_cos


class PositionalEncoder(nn.Module):
    """
    Positional encoder for station embeddings based on geographic coordinates.
    
    This module learns a single embedding for each station to create undirected graphs.
    The adjacency matrix is computed as: Adjacency = E · E^T (symmetric)
    
    This module:
    1. Converts lat/lon to cyclic sin/cos features
    2. Projects these features through a linear layer
    3. Combines with learnable station embeddings
    4. Returns final station embeddings through another linear layer
    """
    
    def __init__(self, n_stations, embedding_dim=64, dropout=0.3):
        """
        Args:
            n_stations: Number of stations in the dataset
            embedding_dim: Dimension of the output embedding
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.n_stations = n_stations
        self.embedding_dim = embedding_dim
        
        # Learnable station embeddings
        self.station_embedding = nn.Embedding(n_stations, embedding_dim)
        
        # Linear layer for cyclic geographic features (4 features: lat_sin, lat_cos, lon_sin, lon_cos)
        # Coordinate embeddings are half the size of station embeddings
        self.coord_projector = nn.Linear(4, embedding_dim // 2)
        
        # Final projection layer that combines coordinate features and station embeddings
        self.final_projector = nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, station_ids, latitude=None, longitude=None):
        """
        Forward pass of the positional encoder.
        
        Args:
            station_ids: Tensor of station indices [batch_size] or [batch_size, n_stations]
            latitude: Latitude values for each station (optional, can be precomputed)
            longitude: Longitude values for each station (optional, can be precomputed)
        
        Returns:
            Station embeddings: [batch_size, n_stations, embedding_dim]
        """
        # Get learnable station embeddings
        station_embed = self.station_embedding(station_ids)  # [batch_size, embedding_dim] or [batch_size, n_stations, embedding_dim]
        station_embed = self.dropout(station_embed)
        
        if latitude is not None and longitude is not None:
            # Convert coordinates to cyclic features
            lat_sin, lat_cos, lon_sin, lon_cos = coords2cyclic_features(latitude, longitude)
            
            # Stack and convert to tensor: [batch_size, 4] or [n_stations, 4]
            coord_features = np.stack([lat_sin, lat_cos, lon_sin, lon_cos], axis=-1)
            coord_features = torch.tensor(coord_features, dtype=station_embed.dtype, device=station_embed.device)
            
            # Project coordinate features
            coord_embed = F.relu(self.coord_projector(coord_features))  # [batch_size, embedding_dim//2] or [n_stations, embedding_dim//2]
            coord_embed = self.dropout(coord_embed)
            
            # Ensure dimensions match for concatenation
            if station_embed.dim() == 2 and coord_embed.dim() == 2:
                # Both are [batch_size, embedding_dim]
                combined = torch.cat([station_embed, coord_embed], dim=-1)
            elif station_embed.dim() == 3:
                # station_embed is [batch_size, n_stations, embedding_dim]
                # Expand coord_embed if needed
                if coord_embed.dim() == 2 and coord_embed.shape[0] == station_embed.shape[1]:
                    # coord_embed is [n_stations, embedding_dim//2], expand to [batch_size, n_stations, embedding_dim//2]
                    coord_embed = coord_embed.unsqueeze(0).expand(station_embed.shape[0], -1, -1)
                combined = torch.cat([station_embed, coord_embed], dim=-1)
            else:
                combined = torch.cat([station_embed, coord_embed], dim=-1)
        else:
            # If no coordinates provided, duplicate station embeddings for concatenation
            combined = torch.cat([station_embed, station_embed], dim=-1)
        
        # Final projection
        output = F.relu(self.final_projector(combined))
        output = self.dropout(output)
        
        return output
