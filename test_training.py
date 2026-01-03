"""
Integration test for TNTPairRE training

Tests:
1. Data loading
2. Complete training loop
3. Loss components (main, regularization, temporal)
4. Entity normalization during training
5. Model convergence on small dataset
"""

import torch
import numpy as np
import sys
sys.path.insert(0, 'tkbc')

from tkbc.models import TNTPairRE
from tkbc.regularizers import N3, TemporalSmoothness
from torch import nn, optim


def create_synthetic_dataset(n_entities=50, n_relations=10, n_timestamps=20, n_samples=500):
    """Create synthetic temporal KG data"""
    print("\n" + "="*60)
    print("Creating synthetic dataset")
    print("="*60)
    
    # Random quadruples (h, r, t, l)
    data = torch.randint(0, n_entities, (n_samples, 4))
    data[:, 1] = torch.randint(0, n_relations, (n_samples,))  # relations
    data[:, 3] = torch.randint(0, n_timestamps, (n_samples,))  # timestamps
    
    print(f"Dataset: {n_samples} quadruples")
    print(f"  Entities: {n_entities}")
    print(f"  Relations: {n_relations}")
    print(f"  Timestamps: {n_timestamps}")
    print(f"  Sample: {data[:3].tolist()}")
    
    return data


def test_forward_backward():
    """Test forward and backward pass"""
    print("\n" + "="*60)
    print("TEST 1: Forward and Backward Pass")
    print("="*60)
    
    sizes = (50, 10, 50, 20)
    rank = 32
    model = TNTPairRE(sizes, rank)
    
    # Create batch
    batch = torch.randint(0, 50, (16, 4))
    batch[:, 1] = torch.randint(0, 10, (16,))
    batch[:, 3] = torch.randint(0, 20, (16,))
    
    # Forward
    predictions, factors, time = model.forward(batch)
    
    # Loss
    truth = batch[:, 2]
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(predictions, truth)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    assert model.entity_embeddings.weight.grad is not None
    assert model.time_embeddings.weight.grad is not None
    assert model.rel_H.weight.grad is not None
    
    print("✓ Forward and backward pass successful")
    print(f"  Entity grad norm: {model.entity_embeddings.weight.grad.norm():.4f}")
    print(f"  Time grad norm: {model.time_embeddings.weight.grad.norm():.4f}")
    
    return model


def test_regularization():
    """Test regularization components"""
    print("\n" + "="*60)
    print("TEST 2: Regularization Components")
    print("="*60)
    
    sizes = (50, 10, 50, 20)
    rank = 32
    model = TNTPairRE(sizes, rank)
    
    # Create batch
    batch = torch.randint(0, 50, (16, 4))
    batch[:, 1] = torch.randint(0, 10, (16,))
    batch[:, 3] = torch.randint(0, 20, (16,))
    
    # Forward
    predictions, factors, time = model.forward(batch)
    
    # Test N3 regularizer
    emb_reg = N3(weight=0.01)
    l_reg = emb_reg.forward(factors)
    print(f"N3 regularization: {l_reg.item():.6f}")
    assert l_reg.item() >= 0, "Regularization should be non-negative"
    
    # Test temporal smoothness regularizer
    time_reg = TemporalSmoothness(weight=0.001)
    l_time = time_reg.forward(time)
    print(f"Temporal smoothness: {l_time.item():.6f}")
    assert l_time.item() >= 0, "Temporal reg should be non-negative"
    
    # Test with uniform time embeddings (should have low smoothness loss)
    model.time_embeddings.weight.data.fill_(1.0)
    l_time_uniform = time_reg.forward(model.time_embeddings.weight)
    print(f"Temporal smoothness (uniform): {l_time_uniform.item():.6f}")
    assert l_time_uniform.item() < 1e-6, "Uniform embeddings should have ~0 smoothness loss"
    
    print("✓ All regularization components work correctly")
    
    return model


def test_entity_normalization():
    """Test entity normalization during training"""
    print("\n" + "="*60)
    print("TEST 3: Entity Normalization During Training")
    print("="*60)
    
    sizes = (30, 5, 30, 10)
    rank = 16
    model = TNTPairRE(sizes, rank)
    
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    # Before training
    norms_before = torch.norm(model.entity_embeddings.weight.data, p=2, dim=1)
    print(f"Before training - norm mean: {norms_before.mean():.6f}, std: {norms_before.std():.6f}")
    
    # Training steps
    for step in range(10):
        batch = torch.randint(0, 30, (8, 4))
        batch[:, 1] = torch.randint(0, 5, (8,))
        batch[:, 3] = torch.randint(0, 10, (8,))
        
        predictions, factors, time = model.forward(batch)
        loss = loss_fn(predictions, batch[:, 2])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Normalize entities
        model._normalize_entities()
        
        # Check norms
        norms = torch.norm(model.entity_embeddings.weight.data, p=2, dim=1)
        max_deviation = torch.abs(norms - 1.0).max()
        
        if step % 3 == 0:
            print(f"Step {step}: max deviation from unit norm = {max_deviation:.8f}")
    
    # After training
    norms_after = torch.norm(model.entity_embeddings.weight.data, p=2, dim=1)
    print(f"\nAfter training - norm mean: {norms_after.mean():.6f}, std: {norms_after.std():.6f}")
    
    assert torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-5), \
        "Entity norms not maintained during training"
    
    print("✓ Entity normalization maintained throughout training")
    
    return model


def test_training_convergence():
    """Test model convergence on small dataset"""
    print("\n" + "="*60)
    print("TEST 4: Training Convergence")
    print("="*60)
    
    # Small dataset for faster testing
    sizes = (20, 5, 20, 8)
    rank = 16
    model = TNTPairRE(sizes, rank)
    
    # Create small training set
    train_data = torch.randint(0, 20, (100, 4))
    train_data[:, 1] = torch.randint(0, 5, (100,))
    train_data[:, 3] = torch.randint(0, 8, (100,))
    
    # Optimizer and regularizers
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    emb_reg = N3(weight=0.001)
    time_reg = TemporalSmoothness(weight=0.0001)
    
    losses = []
    batch_size = 16
    
    print("\nTraining for 50 epochs...")
    for epoch in range(50):
        epoch_loss = 0
        n_batches = 0
        
        # Shuffle data
        perm = torch.randperm(len(train_data))
        train_data = train_data[perm]
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            
            # Forward
            predictions, factors, time = model.forward(batch)
            truth = batch[:, 2]
            
            # Loss components
            l_fit = loss_fn(predictions, truth)
            l_reg = emb_reg.forward(factors)
            l_time = time_reg.forward(time)
            loss = l_fit + l_reg + l_time
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize entities
            model._normalize_entities()
            
            epoch_loss += l_fit.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Loss = {avg_loss:.4f}")
    
    # Check convergence
    initial_loss = np.mean(losses[:5])
    final_loss = np.mean(losses[-5:])
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"\nInitial loss (avg first 5): {initial_loss:.4f}")
    print(f"Final loss (avg last 5):    {final_loss:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    assert final_loss < initial_loss, "Loss should decrease during training"
    assert improvement > 3, f"Training should show improvement (got {improvement:.1f}%)"
    
    print("✓ Model converges successfully")
    
    return model, losses


def test_time_varying_scores():
    """Test that scores vary appropriately with time"""
    print("\n" + "="*60)
    print("TEST 5: Time-Varying Scores")
    print("="*60)
    
    sizes = (20, 5, 20, 10)
    rank = 16
    model = TNTPairRE(sizes, rank)
    
    # Train briefly to get non-random weights
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    
    train_data = torch.randint(0, 20, (50, 4))
    train_data[:, 1] = torch.randint(0, 5, (50,))
    train_data[:, 3] = torch.randint(0, 10, (50,))
    
    for _ in range(10):
        predictions, factors, time = model.forward(train_data)
        loss = loss_fn(predictions, train_data[:, 2])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model._normalize_entities()
    
    # Test temporal variation
    h, r, t = 5, 2, 10
    scores_over_time = []
    
    for time_idx in range(sizes[3]):
        batch = torch.LongTensor([[h, r, t, time_idx]])
        score = model.score(batch)
        scores_over_time.append(score.item())
    
    print(f"\nScores for (h={h}, r={r}, t={t}) across time:")
    for i, s in enumerate(scores_over_time):
        print(f"  t={i}: {s:.6f}")
    
    score_variance = np.var(scores_over_time)
    score_range = max(scores_over_time) - min(scores_over_time)
    
    print(f"\nVariance: {score_variance:.6f}")
    print(f"Range: {score_range:.6f}")
    
    assert score_variance > 1e-6, "Scores should vary with time"
    
    print("✓ Scores appropriately vary with time")
    
    return model


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "#"*60)
    print("# TNTPairRE Training Integration Tests")
    print("#"*60)
    
    try:
        test_forward_backward()
        test_regularization()
        test_entity_normalization()
        test_training_convergence()
        test_time_varying_scores()
        
        print("\n" + "="*60)
        print("ALL INTEGRATION TESTS PASSED ✓")
        print("="*60)
        print("\nTNTPairRE training pipeline works correctly!")
        print("\nYou can now train the model with:")
        print("  python tkbc/learner.py --dataset ICEWS14 --model TNTPairRE \\")
        print("         --rank 100 --learning_rate 0.001 --emb_reg 0.00001 \\")
        print("         --time_reg 0.0001 --max_epochs 100 --batch_size 1000")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
