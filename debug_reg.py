import torch
from tkbc.models import TNTPairRE, TNTComplEx
from tkbc.regularizers import N3

# Create models
sizes = (100, 10, 100, 10)  # n_entities, n_relations, n_entities, n_timestamps
rank = 20

print("=== Testing TNTComplEx ===")
model_complex = TNTComplEx(sizes, rank)
x = torch.LongTensor([[0, 1, 2, 3], [4, 5, 6, 3], [8, 7, 10, 5]])  # (h, r, t, l) - valid indices
predictions, factors_complex, time = model_complex.forward(x)

print(f"Number of factors: {len(factors_complex)}")
for i, f in enumerate(factors_complex):
    print(f"Factor {i} shape: {f.shape}, mean: {f.mean().item():.4f}, sum: {f.sum().item():.4f}")

reg = N3(0.001)
l_reg_complex = reg.forward(factors_complex)
print(f"Regularization loss: {l_reg_complex.item():.6f}")

print("\n=== Testing TNTPairRE ===")
model_pairre = TNTPairRE(sizes, rank)
predictions, factors_pairre, time = model_pairre.forward(x)

print(f"Number of factors: {len(factors_pairre)}")
for i, f in enumerate(factors_pairre):
    print(f"Factor {i} shape: {f.shape}, mean: {f.mean().item():.4f}, sum: {f.sum().item():.4f}")

l_reg_pairre = reg.forward(factors_pairre)
print(f"Regularization loss: {l_reg_pairre.item():.6f}")
