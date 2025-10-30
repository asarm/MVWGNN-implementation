"""
Test script for new architectural components:
1. TaskAwareAdjacencyLearner
2. CrossAttentionFusion
"""

import torch
from taskAwareAdjLearner import TaskAwareAdjacencyLearner
from crossAttentionFusion import CrossAttentionFusion
from model import DDGNNWind

def test_task_aware_adjacency():
    print("=" * 60)
    print("Testing TaskAwareAdjacencyLearner...")
    print("=" * 60)
    
    # Test parameters
    n_stations = 50
    hidden_dim = 64
    batch_size = 8
    
    # Create learner
    learner = TaskAwareAdjacencyLearner(
        hidden_dim=hidden_dim,
        n_stations=n_stations,
        embedding_dim=32
    )
    
    # Test non-batched
    print("\n1. Testing non-batched input...")
    spatial_features = torch.randn(n_stations, hidden_dim)
    temporal_features = torch.randn(n_stations, hidden_dim)
    positions = torch.randn(n_stations, 2)
    wind_directions = torch.randn(n_stations) * 360
    current_wind_speeds = torch.randn(n_stations, 1).abs()  # Wind speeds should be positive
    
    adj = learner(
        spatial_features=spatial_features,
        temporal_features=temporal_features,
        positions=positions,
        wind_directions=wind_directions,
        current_wind_speeds=current_wind_speeds
    )
    
    print(f"   Input shapes: spatial={spatial_features.shape}, temporal={temporal_features.shape}")
    print(f"   Wind speeds shape: {current_wind_speeds.shape}")
    print(f"   Output adjacency shape: {adj.shape}")
    print(f"   Adjacency range: [{adj.min().item():.4f}, {adj.max().item():.4f}]")
    print(f"   Mean edges per node: {(adj > 1e-6).sum(dim=-1).float().mean().item():.2f}")
    print(f"   Blend weight (alpha): {learner.get_blend_weight():.3f}")
    assert adj.shape == (n_stations, n_stations), "Adjacency shape mismatch!"
    
    # Test batched
    print("\n2. Testing batched input...")
    spatial_features_b = torch.randn(batch_size, n_stations, hidden_dim)
    temporal_features_b = torch.randn(batch_size, n_stations, hidden_dim)
    positions_b = torch.randn(batch_size, n_stations, 2)
    wind_directions_b = torch.randn(batch_size, n_stations) * 360
    current_wind_speeds_b = torch.randn(batch_size, n_stations, 1).abs()
    
    adj_b = learner(
        spatial_features=spatial_features_b,
        temporal_features=temporal_features_b,
        positions=positions_b,
        wind_directions=wind_directions_b,
        current_wind_speeds=current_wind_speeds_b
    )
    
    print(f"   Input shapes: spatial={spatial_features_b.shape}, temporal={temporal_features_b.shape}")
    print(f"   Wind speeds shape: {current_wind_speeds_b.shape}")
    print(f"   Output adjacency shape: {adj_b.shape}")
    print(f"   Adjacency range: [{adj_b.min().item():.4f}, {adj_b.max().item():.4f}]")
    assert adj_b.shape == (batch_size, n_stations, n_stations), "Batched adjacency shape mismatch!"
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = adj_b.sum()
    loss.backward()
    
    has_grad = learner.refinement_weight.grad is not None
    print(f"   Refinement weight has gradient: {has_grad}")
    assert has_grad, "Gradient flow broken!"
    
    print("\n✓ TaskAwareAdjacencyLearner tests passed!")


def test_cross_attention_fusion():
    print("\n" + "=" * 60)
    print("Testing CrossAttentionFusion...")
    print("=" * 60)
    
    # Test parameters
    n_stations = 50
    hidden_dim = 64
    batch_size = 8
    n_heads = 4
    
    # Create fusion module
    fusion = CrossAttentionFusion(
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        dropout=0.1
    )
    
    # Test non-batched
    print("\n1. Testing non-batched input...")
    temporal_features = torch.randn(n_stations, hidden_dim)
    spatial_features = torch.randn(n_stations, hidden_dim)
    cyclic_features = torch.randn(hidden_dim)  # Global cyclic encoding
    
    fused = fusion(temporal_features, spatial_features, cyclic_features)
    
    print(f"   Input shapes: temporal={temporal_features.shape}, spatial={spatial_features.shape}, cyclic={cyclic_features.shape}")
    print(f"   Output shape: {fused.shape}")
    print(f"   Output range: [{fused.min().item():.4f}, {fused.max().item():.4f}]")
    assert fused.shape == temporal_features.shape, "Output shape mismatch!"
    
    # Test batched
    print("\n2. Testing batched input...")
    temporal_features_b = torch.randn(batch_size, n_stations, hidden_dim)
    spatial_features_b = torch.randn(batch_size, n_stations, hidden_dim)
    cyclic_features_b = torch.randn(batch_size, hidden_dim)
    
    fused_b = fusion(temporal_features_b, spatial_features_b, cyclic_features_b)
    
    print(f"   Input shapes: temporal={temporal_features_b.shape}, spatial={spatial_features_b.shape}, cyclic={cyclic_features_b.shape}")
    print(f"   Output shape: {fused_b.shape}")
    print(f"   Output range: [{fused_b.min().item():.4f}, {fused_b.max().item():.4f}]")
    assert fused_b.shape == temporal_features_b.shape, "Batched output shape mismatch!"
    
    # Test gradient flow
    print("\n3. Testing gradient flow...")
    loss = fused_b.sum()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in fusion.parameters())
    print(f"   Fusion module has gradients: {has_grad}")
    assert has_grad, "Gradient flow broken!"
    
    # Test attention maps
    print("\n4. Testing attention map extraction...")
    fusion.eval()
    with torch.no_grad():
        attn_maps = fusion.get_attention_maps(temporal_features_b, spatial_features_b, cyclic_features_b)
    
    print(f"   Temporal-to-Spatial attention shape: {attn_maps['temporal_to_spatial'].shape}")
    print(f"   Spatial-to-Temporal attention shape: {attn_maps['spatial_to_temporal'].shape}")
    
    print("\n✓ CrossAttentionFusion tests passed!")


def test_full_model():
    print("\n" + "=" * 60)
    print("Testing Full Model with New Components...")
    print("=" * 60)
    
    # Test parameters
    batch_size = 8
    n_stations = 50
    seq_len = 24
    input_dim = 5
    hidden_dim = 64
    
    # Create model with new features
    print("\n1. Creating model with task-aware adjacency + cross-attention...")
    model = DDGNNWind(
        n_stations=n_stations,
        hidden_dim=hidden_dim,
        n_heads=4,
        seq_len=seq_len,
        n_gnn_layers=2,
        input_dim=input_dim,
        use_task_aware_adj=True,
        use_cross_attention=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test forward pass (non-batched)
    print("\n2. Testing non-batched forward pass...")
    historical_data = torch.randn(n_stations, seq_len, input_dim)
    lat = torch.randn(n_stations)
    lon = torch.randn(n_stations)
    current_hour = torch.tensor(12.0)
    day_of_year = torch.tensor(180.0)
    wind_direction_deg = torch.randn(n_stations) * 360
    positions = torch.randn(n_stations, 2)
    
    predictions = model(
        historical_data=historical_data,
        lat=lat,
        lon=lon,
        current_hour=current_hour,
        day_of_year=day_of_year,
        wind_direction_deg=wind_direction_deg,
        positions=positions
    )
    
    print(f"   Input shape: {historical_data.shape}")
    print(f"   Output shape: {predictions.shape}")
    print(f"   Prediction range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    assert predictions.shape == (n_stations, 1), "Prediction shape mismatch!"
    
    # Test forward pass (batched)
    print("\n3. Testing batched forward pass...")
    historical_data_b = torch.randn(batch_size, n_stations, seq_len, input_dim)
    lat_b = torch.randn(batch_size, n_stations)
    lon_b = torch.randn(batch_size, n_stations)
    current_hour_b = torch.randint(0, 24, (batch_size,)).float()
    day_of_year_b = torch.randint(1, 366, (batch_size,)).float()
    wind_direction_deg_b = torch.randn(batch_size, n_stations) * 360
    positions_b = torch.randn(batch_size, n_stations, 2)
    
    predictions_b = model(
        historical_data=historical_data_b,
        lat=lat_b,
        lon=lon_b,
        current_hour=current_hour_b,
        day_of_year=day_of_year_b,
        wind_direction_deg=wind_direction_deg_b,
        positions=positions_b
    )
    
    print(f"   Input shape: {historical_data_b.shape}")
    print(f"   Output shape: {predictions_b.shape}")
    print(f"   Prediction range: [{predictions_b.min().item():.4f}, {predictions_b.max().item():.4f}]")
    assert predictions_b.shape == (batch_size, n_stations, 1), "Batched prediction shape mismatch!"
    
    # Test gradient flow
    print("\n4. Testing gradient flow through full model...")
    target = torch.randn(batch_size, n_stations, 1)
    loss = torch.nn.functional.mse_loss(predictions_b, target)
    loss.backward()
    
    has_grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params_count = sum(1 for p in model.parameters())
    print(f"   Parameters with gradients: {has_grad_count}/{total_params_count}")
    assert has_grad_count > 0, "No gradients computed!"
    
    # Test with features disabled
    print("\n5. Testing model with features disabled (baseline)...")
    model_baseline = DDGNNWind(
        n_stations=n_stations,
        hidden_dim=hidden_dim,
        n_heads=4,
        seq_len=seq_len,
        n_gnn_layers=2,
        input_dim=input_dim,
        use_task_aware_adj=False,
        use_cross_attention=False
    )
    
    predictions_baseline = model_baseline(
        historical_data=historical_data_b,
        lat=lat_b,
        lon=lon_b,
        current_hour=current_hour_b,
        day_of_year=day_of_year_b,
        wind_direction_deg=wind_direction_deg_b,
        positions=positions_b
    )
    
    print(f"   Baseline output shape: {predictions_baseline.shape}")
    assert predictions_baseline.shape == (batch_size, n_stations, 1), "Baseline prediction shape mismatch!"
    
    print("\n✓ Full model tests passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING NEW ARCHITECTURAL COMPONENTS")
    print("=" * 60)
    
    try:
        test_task_aware_adjacency()
        test_cross_attention_fusion()
        test_full_model()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nNew components are ready to use:")
        print("  - TaskAwareAdjacencyLearner: Refines graph based on wind patterns")
        print("  - CrossAttentionFusion: Better feature integration")
        print("\nTo use in training:")
        print("  model = DDGNNWind(..., use_task_aware_adj=True, use_cross_attention=True)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
