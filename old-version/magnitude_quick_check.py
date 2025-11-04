"""
Quick Magnitude Check
=====================

Lightweight version that prints key metrics every N batches.
"""

import torch

def quick_magnitude_check(model, temporal_features=None, spatial_features=None,
                          adjacency=None, combined_adj_before_wind=None,
                          combined_adj_after_wind=None):
    """
    Quick console check for component magnitudes.
    
    Args:
        model: DDGNNWind model
        temporal_features: (N, H) temporal embeddings
        spatial_features: (N, H) spatial embeddings
        adjacency: Final adjacency matrix
        combined_adj_before_wind: Adjacency before wind modulation
        combined_adj_after_wind: Adjacency after wind modulation
    
    Prints concise summary to console.
    """
    
    with torch.no_grad():
        print("\n[MAGNITUDE CHECK]")
        
        # Temporal diversity
        if temporal_features is not None:
            if temporal_features.dim() == 3:
                temporal_features = temporal_features[0]
            
            temp_norm = temporal_features / (temporal_features.norm(dim=-1, keepdim=True) + 1e-8)
            temp_sim = torch.mm(temp_norm, temp_norm.T)
            N = temp_sim.shape[0]
            mask = ~torch.eye(N, dtype=torch.bool, device=temp_sim.device)
            
            temp_sim_mean = temp_sim[mask].mean().item()
            temp_sim_std = temp_sim[mask].std().item()
            
            if temp_sim_mean > 0.9:
                status = "❌ TOO SIMILAR"
            elif temp_sim_std < 0.05:
                status = "⚠️  LOW VARIANCE"
            else:
                status = "✓ OK"
            
            print(f"  Temporal: similarity={temp_sim_mean:.3f} ± {temp_sim_std:.3f} {status}")
        
        # Spatial diversity
        if spatial_features is not None:
            if spatial_features.dim() == 3:
                spatial_features = spatial_features[0]
            
            spat_norm = spatial_features / (spatial_features.norm(dim=-1, keepdim=True) + 1e-8)
            spat_sim = torch.mm(spat_norm, spat_norm.T)
            mask = ~torch.eye(spat_sim.shape[0], dtype=torch.bool, device=spat_sim.device)
            
            spat_sim_mean = spat_sim[mask].mean().item()
            
            print(f"  Spatial:  similarity={spat_sim_mean:.3f}")
        
        # Wind effect
        if combined_adj_before_wind is not None and combined_adj_after_wind is not None:
            if combined_adj_before_wind.dim() == 3:
                combined_adj_before_wind = combined_adj_before_wind[0]
                combined_adj_after_wind = combined_adj_after_wind[0]
            
            diff = (combined_adj_after_wind - combined_adj_before_wind).abs().mean()
            relative = diff / (combined_adj_before_wind.abs().mean() + 1e-8) * 100
            
            if relative < 2:
                status = "❌ MINIMAL"
            elif relative < 5:
                status = "⚠️  WEAK"
            else:
                status = "✓ GOOD"
            
            print(f"  Wind:     effect={relative:.1f}% {status}")


def add_magnitude_hooks(model):
    """
    Add forward hooks to model to automatically capture intermediate values.
    
    Usage:
        add_magnitude_hooks(model)
        # Now model will store intermediate values in model._mag_check_cache
    """
    
    model._mag_check_cache = {}
    
    def temporal_hook(module, input, output):
        model._mag_check_cache['temporal_features'] = output.detach()
    
    def spatial_hook(module, input, output):
        model._mag_check_cache['spatial_features'] = output.detach()
    
    model.temporal_encoder.register_forward_hook(temporal_hook)
    model.spatial_encoder.register_forward_hook(spatial_hook)
    
    print("[INFO] Magnitude check hooks registered")
