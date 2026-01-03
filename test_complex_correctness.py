"""
Test ComplexPairRETNT correctness
"""
import sys
sys.path.insert(0, 'tkbc')

import torch
from models_complex import ComplexPairRETNT
from datasets import TemporalDataset

# Load dataset
dataset = TemporalDataset('ICEWS14')
sizes = (dataset.n_entities, dataset.n_predicates, dataset.n_entities, dataset.n_timestamps)

# Create model
model = ComplexPairRETNT(sizes, rank=20).cuda()
model.eval()

# Sample queries
test_data = torch.from_numpy(dataset.data['test'].astype('int64'))
queries = test_data[:5].cuda()

print("=== Testing ComplexPairRETNT ===")
print(f"Queries shape: {queries.shape}")
print(f"Model: ComplexPairRETNT with rank=20")

# Test 1: forward() vs score() consistency
print("\n=== Test 1: forward() vs score() consistency ===")
with torch.no_grad():
    predictions, _, _ = model.forward(queries)
    
    for i, q in enumerate(queries):
        forward_score = predictions[i, q[2]].item()
        direct_score = model.score(q.unsqueeze(0)).item()
        diff = abs(forward_score - direct_score)
        status = "✓" if diff < 1e-4 else "✗ ERROR"
        print(f"Query {i}: forward={forward_score:.6f}, direct={direct_score:.6f}, diff={diff:.8f} {status}")

# Test 2: get_ranking() correctness
print("\n=== Test 2: get_ranking() correctness (slow, testing 2 queries) ===")
with torch.no_grad():
    for i, q in enumerate(queries[:2]):
        h, r, t, l = q[0].item(), q[1].item(), q[2].item(), q[3].item()
        
        # Manual: score all entities
        target_score = model.score(q.unsqueeze(0)).item()
        
        better_count = 0
        for t_cand in range(sizes[2]):
            q_test = torch.LongTensor([[h, r, t_cand, l]]).cuda()
            score = model.score(q_test).item()
            if score > target_score:
                better_count += 1
        
        rank_manual = 1 + better_count
        
        # Using get_ranking
        filters = {(h, r, l): []}
        ranks = model.get_ranking(q.unsqueeze(0), filters, batch_size=1, chunk_size=2000)
        rank_method = ranks[0].item()
        
        diff = abs(rank_manual - rank_method)
        status = "✓" if diff == 0 else f"✗ ERROR (diff={diff})"
        print(f"Query {i}: manual_rank={rank_manual}, get_ranking={rank_method:.0f} {status}")

# Test 3: Complex multiplication correctness
print("\n=== Test 3: Complex rotation correctness ===")
with torch.no_grad():
    # Test that rotation preserves magnitude for unit complex numbers
    theta = torch.tensor([[0.5, 1.0, 1.5]]).cuda()
    r_real, r_imag = model._angle_to_complex(theta)
    
    # Check |exp(iθ)| = sqrt(cos²θ + sin²θ) = 1
    magnitude = torch.sqrt(r_real**2 + r_imag**2)
    print(f"Rotation magnitude (should be 1.0): {magnitude}")
    if torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5):
        print("✓ Rotation preserves magnitude")
    else:
        print("✗ ERROR: Rotation magnitude not 1.0!")

print("\nDone!")
