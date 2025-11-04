"""
Magnitude Analysis for Adjacency Components
============================================

Check if each component (spatial, temporal, wind) actually has enough
variance to make a difference.

Key Questions:
1. Does temporal_adj vary across nodes? Or all ~same?
2. Does wind modulation vary? Or always ~1.0?
3. What's the magnitude of each component's contribution?
"""

import torch
import numpy as np

def analyze_adjacency_magnitudes(model, historical_data, lat, lon, 
                                 wind_direction_deg, positions, 
                                 current_hour=None, day_of_year=None):
    """
    Comprehensive magnitude analysis of adjacency components.
    
    This function intercepts intermediate values to see if they
    actually vary enough to matter.
    
    Args:
        model: Your DDGNNWind model
        historical_data, lat, lon, etc: Model inputs
    
    Returns:
        dict with detailed statistics
    """
    
    print("\n" + "="*70)
    print("MAGNITUDE ANALYSIS: Adjacency Components")
    print("="*70)
    
    device = historical_data.device
    model.eval()
    
    with torch.no_grad():
        # ========== Get intermediate representations ==========
        
        # 1. Temporal features
        temporal_features = model.temporal_encoder(historical_data)
        # (N, H) or (B, N, H)
        
        # 2. Spatial features
        spatial_embed = model.spatial_encoder(lat, lon, wind_direction_deg)
        # (N, H) or (B, N, H)
        
        # Handle batching
        if temporal_features.dim() == 3:
            B, N, H = temporal_features.shape
            is_batched = True
            # Take first sample for analysis
            temporal_features = temporal_features[0]
            spatial_embed = spatial_embed[0]
            positions = positions[0] if positions.dim() == 3 else positions
            wind_direction_deg = wind_direction_deg[0] if wind_direction_deg.dim() == 2 else wind_direction_deg
        else:
            N, H = temporal_features.shape
            is_batched = False
        
        print(f"\nNodes: {N}, Hidden dim: {H}")
        
        # ========== TEMPORAL ANALYSIS ==========
        print("\n" + "-"*70)
        print("1. TEMPORAL FEATURES")
        print("-"*70)
        
        temp_mean = temporal_features.mean(dim=0)  # Mean across nodes
        temp_std = temporal_features.std(dim=0)    # Std across nodes
        
        # Average variance across dimensions
        avg_node_variance = temp_std.mean().item()
        max_node_variance = temp_std.max().item()
        min_node_variance = temp_std.min().item()
        
        print(f"Node-wise variance (across {N} nodes):")
        print(f"  Mean std: {avg_node_variance:.6f}")
        print(f"  Max std:  {max_node_variance:.6f}")
        print(f"  Min std:  {min_node_variance:.6f}")
        
        # Pairwise similarity (cosine)
        temporal_norm = temporal_features / (temporal_features.norm(dim=-1, keepdim=True) + 1e-8)
        temporal_sim = torch.mm(temporal_norm, temporal_norm.T)  # (N, N)
        
        # Remove diagonal
        mask = ~torch.eye(N, dtype=torch.bool, device=device)
        temporal_sim_offdiag = temporal_sim[mask]
        
        print(f"\nTemporal pairwise similarity (cosine):")
        print(f"  Mean: {temporal_sim_offdiag.mean():.6f}")
        print(f"  Std:  {temporal_sim_offdiag.std():.6f}")
        print(f"  Min:  {temporal_sim_offdiag.min():.6f}")
        print(f"  Max:  {temporal_sim_offdiag.max():.6f}")
        
        if temporal_sim_offdiag.mean() > 0.9:
            print("  ⚠️  WARNING: Temporal features TOO SIMILAR (mean > 0.9)")
            print("     → Temporal encoder may not be discriminative!")
        elif temporal_sim_offdiag.std() < 0.05:
            print("  ⚠️  WARNING: Temporal similarity has LOW VARIANCE")
            print("     → All nodes have similar temporal patterns!")
        else:
            print("  ✓ Temporal features show good diversity")
        
        # ========== SPATIAL ANALYSIS ==========
        print("\n" + "-"*70)
        print("2. SPATIAL FEATURES")
        print("-"*70)
        
        spatial_std = spatial_embed.std(dim=0).mean().item()
        print(f"Spatial feature variance: {spatial_std:.6f}")
        
        spatial_norm = spatial_embed / (spatial_embed.norm(dim=-1, keepdim=True) + 1e-8)
        spatial_sim = torch.mm(spatial_norm, spatial_norm.T)
        spatial_sim_offdiag = spatial_sim[mask]
        
        print(f"\nSpatial pairwise similarity:")
        print(f"  Mean: {spatial_sim_offdiag.mean():.6f}")
        print(f"  Std:  {spatial_sim_offdiag.std():.6f}")
        
        if spatial_sim_offdiag.mean() > 0.9:
            print("  ⚠️  WARNING: Spatial features TOO SIMILAR")
        else:
            print("  ✓ Spatial features show good diversity")
        
        # ========== ADJACENCY COMPONENTS ==========
        print("\n" + "-"*70)
        print("3. ADJACENCY COMPONENTS (Before Combination)")
        print("-"*70)
        
        # Call adjacency learner with debugging
        adj_learner = model.adjacency_learner
        
        # Compute spatial adjacency
        if is_batched:
            spatial_emb = adj_learner.spatial_node_embedding.unsqueeze(0)
        else:
            spatial_emb = adj_learner.spatial_node_embedding
        
        spatial_adj = torch.mm(spatial_emb.squeeze(), spatial_emb.squeeze().T)
        spatial_adj = spatial_adj / np.sqrt(adj_learner.embedding_dim)
        
        print(f"\nSpatial Adjacency (before softmax):")
        print(f"  Mean:  {spatial_adj.mean():.6f}")
        print(f"  Std:   {spatial_adj.std():.6f}")
        print(f"  Range: [{spatial_adj.min():.6f}, {spatial_adj.max():.6f}]")
        
        # Compute temporal modulation
        temporal_mod = adj_learner.temporal_modulator(temporal_features)
        temporal_adj = torch.mm(temporal_mod, temporal_mod.T)
        temporal_adj = temporal_adj / np.sqrt(adj_learner.embedding_dim)
        
        print(f"\nTemporal Adjacency (before softmax):")
        print(f"  Mean:  {temporal_adj.mean():.6f}")
        print(f"  Std:   {temporal_adj.std():.6f}")
        print(f"  Range: [{temporal_adj.min():.6f}, {temporal_adj.max():.6f}]")
        
        # Check relative magnitudes
        spatial_magnitude = spatial_adj.std().item()
        temporal_magnitude = temporal_adj.std().item()
        
        ratio = temporal_magnitude / (spatial_magnitude + 1e-8)
        
        print(f"\n**Magnitude Ratio (temporal/spatial): {ratio:.4f}**")
        
        if ratio < 0.1:
            print("  ❌ CRITICAL: Temporal variance << Spatial variance")
            print("     → Temporal component is DROWNED OUT!")
            print(f"     → 0.3 * {temporal_magnitude:.4f} << 0.7 * {spatial_magnitude:.4f}")
        elif ratio < 0.5:
            print("  ⚠️  WARNING: Temporal weaker than spatial")
            print("     → Consider increasing temporal weight")
        elif ratio > 2.0:
            print("  ⚠️  WARNING: Temporal much stronger than spatial")
            print("     → Consider decreasing temporal weight")
        else:
            print("  ✓ Magnitudes are reasonably balanced")
        
        # ========== COMBINED ADJACENCY ==========
        print("\n" + "-"*70)
        print("4. COMBINED ADJACENCY")
        print("-"*70)
        
        combined_adj = 0.7 * spatial_adj + 0.3 * temporal_adj
        
        print(f"\nCombined (0.7*spatial + 0.3*temporal):")
        print(f"  Mean:  {combined_adj.mean():.6f}")
        print(f"  Std:   {combined_adj.std():.6f}")
        print(f"  Range: [{combined_adj.min():.6f}, {combined_adj.max():.6f}]")
        
        # Contribution analysis
        spatial_contribution = (0.7 * spatial_adj).std().item()
        temporal_contribution = (0.3 * temporal_adj).std().item()
        
        total_std = combined_adj.std().item()
        
        spatial_pct = (spatial_contribution / total_std) * 100
        temporal_pct = (temporal_contribution / total_std) * 100
        
        print(f"\n**Component Contributions (to variance):**")
        print(f"  Spatial:  {spatial_pct:.1f}%")
        print(f"  Temporal: {temporal_pct:.1f}%")
        
        if temporal_pct < 10:
            print("  ❌ CRITICAL: Temporal contributes < 10% of variance!")
            print("     → Temporal is essentially IGNORED")
        elif temporal_pct < 20:
            print("  ⚠️  WARNING: Temporal contribution < 20%")
        else:
            print("  ✓ Both components contribute meaningfully")
        
        # ========== WIND MODULATION ==========
        print("\n" + "-"*70)
        print("5. WIND DIRECTION MODULATION")
        print("-"*70)
        
        if wind_direction_deg is not None and positions is not None:
            # Compute wind modulation
            wind_rad = torch.deg2rad(wind_direction_deg)
            wind_vectors = torch.stack([torch.cos(wind_rad), torch.sin(wind_rad)], dim=-1)
            
            offsets = positions[:, None, :] - positions[None, :, :]
            dists = torch.norm(offsets, dim=-1)
            spatial_dir = offsets / (dists[..., None] + 1e-6)
            
            wind_exp = wind_vectors[:, None, :]
            alignment = (wind_exp * spatial_dir).sum(dim=-1)
            
            # Compute modulation weights
            wind_x_mat = wind_vectors[:, 0].unsqueeze(1).expand(-1, N)
            wind_y_mat = wind_vectors[:, 1].unsqueeze(1).expand(-1, N)
            
            dir_input = torch.stack([wind_x_mat, wind_y_mat, alignment], dim=-1)
            
            flat_input = dir_input.reshape(-1, 3)
            mod_flat = adj_learner.directional_influence(flat_input)
            modulation = mod_flat.view(N, N)
            
            print(f"\nWind Modulation Weights:")
            print(f"  Mean:  {modulation.mean():.6f}")
            print(f"  Std:   {modulation.std():.6f}")
            print(f"  Range: [{modulation.min():.6f}, {modulation.max():.6f}]")
            
            # Effect on adjacency
            modulated_adj = combined_adj * modulation
            
            effect = (modulated_adj - combined_adj).abs().mean().item()
            relative_effect = effect / (combined_adj.abs().mean().item() + 1e-8)
            
            print(f"\n**Wind Effect on Adjacency:**")
            print(f"  Absolute change: {effect:.6f}")
            print(f"  Relative change: {relative_effect*100:.2f}%")
            
            if modulation.std() < 0.05:
                print("  ❌ CRITICAL: Modulation has LOW VARIANCE")
                print("     → Wind modulation is basically UNIFORM (~1.0)")
                print("     → Not actually changing the graph!")
            elif relative_effect < 5:
                print("  ⚠️  WARNING: Wind changes adjacency < 5%")
                print("     → Effect is minimal")
            else:
                print("  ✓ Wind modulation has meaningful effect")
        else:
            print("No wind direction provided - skipping analysis")
        
        # ========== SUMMARY ==========
        print("\n" + "="*70)
        print("SUMMARY & RECOMMENDATIONS")
        print("="*70)
        
        issues = []
        recommendations = []
        
        # Check temporal features
        if temporal_sim_offdiag.mean() > 0.9:
            issues.append("❌ Temporal features too similar across nodes")
            recommendations.append("→ Increase temporal encoder capacity (hidden_dim)")
            recommendations.append("→ Check if attention is actually learning (debug=True)")
        
        # Check magnitude balance
        if ratio < 0.1:
            issues.append("❌ Temporal component drowned out by spatial")
            recommendations.append(f"→ Increase temporal weight: 0.3 → {min(0.5, 0.3 * (1/ratio)):.2f}")
            recommendations.append("→ Or normalize components before combining")
        
        if temporal_pct < 10:
            issues.append("❌ Temporal contributes < 10% of variance")
            recommendations.append("→ Rebalance weights or normalize inputs")
        
        # Check wind modulation
        if wind_direction_deg is not None and modulation.std() < 0.05:
            issues.append("❌ Wind modulation has no variance (always ~1.0)")
            recommendations.append("→ Check directional_influence network weights")
            recommendations.append("→ May need stronger initialization or different architecture")
        
        if issues:
            print("\n**Issues Found:**")
            for issue in issues:
                print(f"  {issue}")
            print("\n**Recommendations:**")
            for rec in recommendations:
                print(f"  {rec}")
        else:
            print("\n✅ All components show good variance and balance!")
        
        print("="*70 + "\n")
        
        return {
            'temporal_similarity': temporal_sim_offdiag.mean().item(),
            'temporal_std': temporal_sim_offdiag.std().item(),
            'spatial_similarity': spatial_sim_offdiag.mean().item(),
            'magnitude_ratio': ratio,
            'temporal_contribution_pct': temporal_pct,
            'wind_modulation_std': modulation.std().item() if wind_direction_deg is not None else None,
            'wind_effect_pct': relative_effect * 100 if wind_direction_deg is not None else None,
        }


# ==================== USAGE ====================
"""
# In train.py, epoch 1, first batch:

if epoch == 1 and batch_idx == 0:
    from magnitude_analysis import analyze_adjacency_magnitudes
    
    results = analyze_adjacency_magnitudes(
        model=model,
        historical_data=historical_data,
        lat=lat,
        lon=lon,
        wind_direction_deg=wind_dir_deg_last,
        positions=positions,
        current_hour=current_hour,
        day_of_year=day_of_year
    )
    
    # results contains all statistics for logging
"""