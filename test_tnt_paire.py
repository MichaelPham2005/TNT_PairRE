"""
Comprehensive unit tests for TNTPairRE model

Tests:
1. Model initialization
2. Score function correctness
3. Entity unit norm constraint
4. Temporal decomposition (TNT-style)
5. Forward pass dimensions
6. Gradient flow
7. Time-conditioned relations
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'tkbc')

from tkbc.models import TNTPairRE


def test_initialization():
    """Test model initialization and parameter shapes"""
    print("\n" + "="*60)
    print("TEST 1: Model Initialization")
    print("="*60)
    
    sizes = (100, 20, 100, 50)  # (n_entities, n_relations, n_entities, n_timestamps)
    rank = 64
    
    model = TNTPairRE(sizes, rank)
    
    # Check entity embeddings
    assert model.entity_embeddings.weight.shape == (100, 64), \
        f"Entity shape mismatch: {model.entity_embeddings.weight.shape}"
    
    # Check time embeddings
    assert model.time_embeddings.weight.shape == (50, 64), \
        f"Time shape mismatch: {model.time_embeddings.weight.shape}"
    
    # Check relation embeddings (4 vectors per relation)
    assert model.rel_H.weight.shape == (20, 64), \
        f"rel_H shape mismatch: {model.rel_H.weight.shape}"
    assert model.rel_T.weight.shape == (20, 64), \
        f"rel_T shape mismatch: {model.rel_T.weight.shape}"
    assert model.rel_H_t.weight.shape == (20, 64), \
        f"rel_H_t shape mismatch: {model.rel_H_t.weight.shape}"
    assert model.rel_T_t.weight.shape == (20, 64), \
        f"rel_T_t shape mismatch: {model.rel_T_t.weight.shape}"
    
    print("✓ All embeddings initialized with correct shapes")
    print(f"  - Entity embeddings: {model.entity_embeddings.weight.shape}")
    print(f"  - Time embeddings: {model.time_embeddings.weight.shape}")
    print(f"  - Relation embeddings (4 types): {model.rel_H.weight.shape}")
    
    return model


def test_entity_norm_constraint():
    """Test that entity embeddings are normalized to unit L2 norm"""
    print("\n" + "="*60)
    print("TEST 2: Entity Unit Norm Constraint")
    print("="*60)
    
    sizes = (100, 20, 100, 50)
    rank = 64
    model = TNTPairRE(sizes, rank)
    
    # Initial norms (before normalization)
    initial_norms = torch.norm(model.entity_embeddings.weight.data, p=2, dim=1)
    print(f"Initial entity norms - mean: {initial_norms.mean():.4f}, std: {initial_norms.std():.4f}")
    
    # Apply normalization
    model._normalize_entities()
    
    # Check norms after normalization
    norms = torch.norm(model.entity_embeddings.weight.data, p=2, dim=1)
    print(f"After normalization - mean: {norms.mean():.6f}, std: {norms.std():.6f}")
    
    # All norms should be 1.0 (within numerical precision)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Entity norms not unit: mean={norms.mean()}, max_dev={torch.abs(norms - 1.0).max()}"
    
    print("✓ All entity embeddings have unit L2 norm")
    print(f"  - Max deviation from 1.0: {torch.abs(norms - 1.0).max():.8f}")
    
    return model


def test_score_function():
    """Test score function computation"""
    print("\n" + "="*60)
    print("TEST 3: Score Function Correctness")
    print("="*60)
    
    sizes = (10, 5, 10, 8)
    rank = 16
    model = TNTPairRE(sizes, rank)
    model._normalize_entities()
    
    # Create a batch of quadruples (h, r, t, l)
    batch = torch.LongTensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
        [2, 0, 1, 5],
    ])
    
    # Compute scores
    scores = model.score(batch)
    
    print(f"Batch shape: {batch.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Scores: {scores.squeeze().tolist()}")
    
    # Check output shape
    assert scores.shape == (3, 1), f"Score shape mismatch: {scores.shape}"
    
    # Manually verify first score
    e_h = model.entity_embeddings.weight[0]
    e_t = model.entity_embeddings.weight[2]
    tau = model.time_embeddings.weight[3]
    r_H = model.rel_H.weight[1]
    r_T = model.rel_T.weight[1]
    r_H_t = model.rel_H_t.weight[1]
    r_T_t = model.rel_T_t.weight[1]
    
    # Time-conditioned relations
    r_H_l = r_H + r_H_t * tau
    r_T_l = r_T + r_T_t * tau
    
    # Score: -||e_h * r^H_l - e_t * r^T_l||_1
    diff = e_h * r_H_l - e_t * r_T_l
    expected_score = -torch.sum(torch.abs(diff))
    
    print(f"\nManual computation for first quadruple:")
    print(f"  Expected score: {expected_score:.6f}")
    print(f"  Model score: {scores[0, 0]:.6f}")
    print(f"  Difference: {torch.abs(expected_score - scores[0, 0]):.8f}")
    
    assert torch.allclose(scores[0, 0], expected_score, atol=1e-5), \
        "Score computation mismatch"
    
    print("✓ Score function computes correctly")
    
    return model


def test_temporal_decomposition():
    """Test TNT-style temporal decomposition"""
    print("\n" + "="*60)
    print("TEST 4: TNT-Style Temporal Decomposition")
    print("="*60)
    
    sizes = (10, 5, 10, 8)
    rank = 16
    model = TNTPairRE(sizes, rank, init_size=0.1)  # Larger init for visible temporal effects
    model._normalize_entities()  # Normalize after init
    
    # Test that different timestamps produce different scores for same (h,r,t)
    h, r, t = 0, 1, 2
    
    scores_by_time = []
    for time_idx in range(sizes[3]):
        batch = torch.LongTensor([[h, r, t, time_idx]])
        score = model.score(batch)
        scores_by_time.append(score.item())
    
    print(f"Scores across different timestamps:")
    for i, s in enumerate(scores_by_time):
        print(f"  t={i}: {s:.6f}")
    
    # Check that scores vary with time
    score_variance = np.var(scores_by_time)
    print(f"\nScore variance across time: {score_variance:.6f}")
    
    # If temporal components are non-zero, scores should vary
    assert score_variance > 1e-8, "Scores don't vary with time (temporal components might be zero)"
    
    print("✓ Temporal decomposition produces time-varying scores")
    
    # Test decomposition structure: r_l = r_static + r_temporal * tau
    r_H = model.rel_H.weight[r]
    r_T = model.rel_T.weight[r]
    r_H_t = model.rel_H_t.weight[r]
    r_T_t = model.rel_T_t.weight[r]
    
    print(f"\nRelation vector norms:")
    print(f"  ||r^H|| = {torch.norm(r_H):.6f}")
    print(f"  ||r^T|| = {torch.norm(r_T):.6f}")
    print(f"  ||r^{{H,t}}|| = {torch.norm(r_H_t):.6f}")
    print(f"  ||r^{{T,t}}|| = {torch.norm(r_T_t):.6f}")
    
    return model


def test_forward_pass():
    """Test forward pass for 1-vs-all training"""
    print("\n" + "="*60)
    print("TEST 5: Forward Pass (1-vs-All)")
    print("="*60)
    
    sizes = (50, 10, 50, 20)
    rank = 32
    model = TNTPairRE(sizes, rank)
    
    # Batch of quadruples
    batch = torch.LongTensor([
        [0, 1, 2, 3],
        [5, 2, 10, 8],
        [10, 3, 15, 12],
    ])
    
    scores, factors, time = model.forward(batch)
    
    print(f"Input batch shape: {batch.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Expected shape: ({batch.shape[0]}, {sizes[0]})")
    
    # Check output dimensions
    assert scores.shape == (3, 50), f"Scores shape mismatch: {scores.shape}"
    assert len(factors) == 3, f"Expected 3 regularization factors, got {len(factors)}"
    assert time is not None, "Time embeddings not returned"
    
    print(f"✓ Forward pass produces correct dimensions")
    print(f"  - Scores for all entities: {scores.shape}")
    print(f"  - Regularization factors: {len(factors)} tuples")
    print(f"  - Time embeddings: {time.shape}")
    
    # Check that scores vary across entities
    score_stds = scores.std(dim=1)
    print(f"\nScore std per sample: {score_stds.tolist()}")
    assert torch.all(score_stds > 1e-6), "Scores don't vary across entities"
    
    return model, scores


def test_forward_over_time():
    """Test forward_over_time method"""
    print("\n" + "="*60)
    print("TEST 6: Forward Over Time")
    print("="*60)
    
    sizes = (30, 8, 30, 15)
    rank = 24
    model = TNTPairRE(sizes, rank)
    
    # Batch of triples (h, r, t) - note: no timestamp
    batch = torch.LongTensor([
        [0, 1, 2, 0],  # dummy timestamp
        [5, 2, 10, 0],
    ])
    
    scores = model.forward_over_time(batch)
    
    print(f"Input batch shape: {batch.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Expected shape: ({batch.shape[0]}, {sizes[3]})")
    
    # Check dimensions
    assert scores.shape == (2, 15), f"Scores shape mismatch: {scores.shape}"
    
    print("✓ forward_over_time produces correct dimensions")
    print(f"  - Scores across all timestamps: {scores.shape}")
    
    # Check that scores vary over time
    score_stds = scores.std(dim=1)
    print(f"\nScore std per triple: {score_stds.tolist()}")
    
    return model


def test_gradient_flow():
    """Test that gradients flow properly through the model"""
    print("\n" + "="*60)
    print("TEST 7: Gradient Flow")
    print("="*60)
    
    sizes = (20, 5, 20, 10)
    rank = 16
    model = TNTPairRE(sizes, rank)
    
    # Create synthetic batch
    batch = torch.LongTensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ])
    
    # Forward pass
    scores, factors, time = model.forward(batch)
    
    # Compute loss
    targets = batch[:, 2]
    loss = -scores[range(len(targets)), targets].mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist for all parameters
    params_with_grad = []
    params_without_grad = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            params_with_grad.append(name)
        else:
            params_without_grad.append(name)
    
    print(f"Parameters WITH gradients: {len(params_with_grad)}")
    for name in params_with_grad:
        print(f"  ✓ {name}")
    
    if params_without_grad:
        print(f"\nParameters WITHOUT gradients: {len(params_without_grad)}")
        for name in params_without_grad:
            print(f"  ✗ {name}")
    
    # At least entity, relation, and time embeddings should have gradients
    assert len(params_with_grad) >= 6, "Not enough parameters have gradients"
    
    print(f"\n✓ Gradients flow through {len(params_with_grad)} parameters")
    
    return model


def test_consistency():
    """Test consistency between score() and forward()"""
    print("\n" + "="*60)
    print("TEST 8: Consistency (score vs forward)")
    print("="*60)
    
    sizes = (30, 8, 30, 12)
    rank = 24
    model = TNTPairRE(sizes, rank)
    model._normalize_entities()
    
    # Test batch
    batch = torch.LongTensor([
        [0, 1, 5, 3],
        [2, 3, 8, 6],
    ])
    
    # Get scores from score() method
    scores_direct = model.score(batch)
    
    # Get scores from forward() method
    scores_all, _, _ = model.forward(batch)
    scores_from_forward = scores_all[range(len(batch)), batch[:, 2]].unsqueeze(1)
    
    print(f"Scores from score():   {scores_direct.squeeze().tolist()}")
    print(f"Scores from forward(): {scores_from_forward.squeeze().tolist()}")
    print(f"Difference: {(scores_direct - scores_from_forward).abs().max():.8f}")
    
    # They should match
    assert torch.allclose(scores_direct, scores_from_forward, atol=1e-4), \
        "Inconsistency between score() and forward()"
    
    print("✓ score() and forward() are consistent")
    
    return model


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "#"*60)
    print("# TNTPairRE Model Unit Tests")
    print("#"*60)
    
    try:
        test_initialization()
        test_entity_norm_constraint()
        test_score_function()
        test_temporal_decomposition()
        test_forward_pass()
        test_forward_over_time()
        test_gradient_flow()
        test_consistency()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)
        print("\nTNTPairRE model implementation is correct!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
