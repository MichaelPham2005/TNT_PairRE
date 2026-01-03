"""
Test correctness of TNTPairRE ranking implementation
"""
import sys
sys.path.insert(0, 'tkbc')

import torch
from models import TNTPairRE
from datasets import TemporalDataset

# Load dataset
dataset = TemporalDataset('ICEWS14')
sizes = (dataset.n_entities, dataset.n_predicates, dataset.n_entities, dataset.n_timestamps)

# Create model
model = TNTPairRE(sizes, rank=20).cuda()
model.eval()

# Sample queries
test_data = torch.from_numpy(dataset.data['test'].astype('int64'))
queries = test_data[:10].cuda()

print("Testing ranking correctness...")
print(f"Queries shape: {queries.shape}")

# Method 1: Using score() directly for ALL entities
print("\n=== Method 1: Direct score() for all entities ===")
with torch.no_grad():
    for i, q in enumerate(queries[:3]):  # Test first 3 queries only (slow)
        h, r, t, l = q[0].item(), q[1].item(), q[2].item(), q[3].item()
        
        # Score the correct triple
        true_score = model.score(q.unsqueeze(0)).item()
        
        # Score all possible tails
        better_count = 0
        for t_candidate in range(sizes[2]):
            q_test = torch.LongTensor([[h, r, t_candidate, l]]).cuda()
            score = model.score(q_test).item()
            if score > true_score:
                better_count += 1
        
        rank = 1 + better_count
        print(f"Query {i}: true_score={true_score:.6f}, rank={rank}/{sizes[2]}")

# Method 2: Using get_ranking()
print("\n=== Method 2: get_ranking() ===")
filters = {(q[0].item(), q[1].item(), q[3].item()): [] for q in queries}

with torch.no_grad():
    ranks = model.get_ranking(queries, filters, batch_size=10, chunk_size=2000)
    for i, rank in enumerate(ranks[:10]):
        print(f"Query {i}: rank={rank.item():.0f}")

# Method 3: Check forward() consistency
print("\n=== Method 3: forward() vs score() ===")
with torch.no_grad():
    # Get scores from forward (1-vs-all)
    predictions, _, _ = model.forward(queries)
    
    for i, q in enumerate(queries):
        # Score from forward
        forward_score = predictions[i, q[2]].item()
        
        # Score from score()
        direct_score = model.score(q.unsqueeze(0)).item()
        
        diff = abs(forward_score - direct_score)
        print(f"Query {i}: forward={forward_score:.4f}, direct={direct_score:.4f}, diff={diff:.6f}")
        
        if diff > 1e-4:
            print(f"  WARNING: Large difference!")

print("\nDone!")
