"""
Full ranking debug
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
query = test_data[0:1].cuda()

print(f"Query: h={query[0,0].item()}, r={query[0,1].item()}, t={query[0,2].item()}, l={query[0,3].item()}")
print(f"Total entities: {sizes[2]}")

with torch.no_grad():
    # Get target score
    target_score = model.score(query).item()
    print(f"Target score: {target_score:.6f}")
    
    # Manual: count entities with better score
    e_h = model.entity_embeddings(query[:, 0])
    tau = model.time_embeddings(query[:, 3])
    
    r_H = model.rel_H(query[:, 1])
    r_T = model.rel_T(query[:, 1])
    r_H_t = model.rel_H_t(query[:, 1])
    r_T_t = model.rel_T_t(query[:, 1])
    
    r_H_l = r_H + r_H_t * tau
    r_T_l = r_T + r_T_t * tau
    
    lhs = e_h * r_H_l
    
    # Score ALL entities
    all_entities = model.entity_embeddings.weight  # (n_entities, rank)
    lhs_exp = lhs.unsqueeze(1)  # (1, 1, rank)
    r_T_l_exp = r_T_l.unsqueeze(1)  # (1, 1, rank)
    entities_exp = all_entities.unsqueeze(0)  # (1, n_entities, rank)
    
    rhs = entities_exp * r_T_l_exp
    diff = lhs_exp - rhs
    all_scores = -torch.sum(torch.abs(diff), dim=2).squeeze(0)  # (n_entities,)
    
    # Count better
    better_count = (all_scores > target_score).sum().item()
    rank_manual = 1 + better_count
    
    print(f"\nManual ranking:")
    print(f"  Entities scoring > target: {better_count}")
    print(f"  Rank: {rank_manual}")
    
    # Verify target entity score
    target_entity = query[0, 2].item()
    print(f"  Target entity {target_entity} score: {all_scores[target_entity].item():.6f}")
    print(f"  Should equal target: {target_score:.6f}")
    print(f"  Match? {abs(all_scores[target_entity].item() - target_score) < 1e-5}")
    
    # Now use get_ranking
    print(f"\nUsing get_ranking with chunk_size=2000:")
    filters = {(query[0,0].item(), query[0,1].item(), query[0,3].item()): []}
    ranks = model.get_ranking(query, filters, batch_size=1, chunk_size=2000)
    print(f"  Rank from get_ranking: {ranks[0].item():.0f}")
    
    print(f"\nDifference: {abs(ranks[0].item() - rank_manual)}")
