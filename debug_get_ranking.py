"""
Debug get_ranking step by step
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

# Single query
test_data = torch.from_numpy(dataset.data['test'].astype('int64'))
query = test_data[0:1].cuda()  # (1, 4)

print(f"Query: h={query[0,0].item()}, r={query[0,1].item()}, t={query[0,2].item()}, l={query[0,3].item()}")

with torch.no_grad():
    # Get target score
    target_score = model.score(query).item()
    print(f"\nTarget score: {target_score:.6f}")
    
    # Manual ranking: score first 200 entities
    e_h = model.entity_embeddings(query[:, 0])
    tau = model.time_embeddings(query[:, 3])
    
    r_H = model.rel_H(query[:, 1])
    r_T = model.rel_T(query[:, 1])
    r_H_t = model.rel_H_t(query[:, 1])
    r_T_t = model.rel_T_t(query[:, 1])
    
    r_H_l = r_H + r_H_t * tau
    r_T_l = r_T + r_T_t * tau
    
    lhs = e_h * r_H_l  # (1, rank)
    
    # Score all entities
    print("\nScoring first 10 entities manually...")
    for t_idx in range(10):
        e_t = model.entity_embeddings.weight[t_idx:t_idx+1]  # (1, rank)
        rhs = e_t * r_T_l
        diff = lhs - rhs
        score = -torch.sum(torch.abs(diff)).item()
        better = ">" if score > target_score else "<=" 
        print(f"  Entity {t_idx}: score={score:.6f} {better} target")
    
    # Now test chunk computation
    print("\nTesting chunk computation (entities 0-100)...")
    chunk_size = 100
    entities_chunk = model.entity_embeddings.weight[0:chunk_size]  # (100, rank)
    
    lhs_exp = lhs.unsqueeze(1)  # (1, 1, rank)
    r_T_l_exp = r_T_l.unsqueeze(1)  # (1, 1, rank)
    entities_exp = entities_chunk.unsqueeze(0)  # (1, 100, rank)
    
    rhs = entities_exp * r_T_l_exp  # (1, 100, rank)
    diff = lhs_exp - rhs  # (1, 100, rank)
    scores = -torch.sum(torch.abs(diff), dim=2)  # (1, 100)
    
    print(f"  Scores shape: {scores.shape}")
    print(f"  First 10 scores from chunk: {scores[0, :10]}")
    print(f"  Target score: {target_score:.6f}")
    print(f"  Number scoring > target: {(scores[0] > target_score).sum().item()}")
    
    # Check specific target entity score
    target_entity = query[0, 2].item()
    if target_entity < chunk_size:
        print(f"\n  Target entity {target_entity} score from chunk: {scores[0, target_entity].item():.6f}")
        print(f"  Should match target: {target_score:.6f}")
        print(f"  Match? {abs(scores[0, target_entity].item() - target_score) < 1e-5}")
