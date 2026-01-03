import sys
sys.path.insert(0, 'tkbc')

import torch
from datasets import TemporalDataset
from models import TNTPairRE, TNTComplEx
from regularizers import N3

# Load real dataset
dataset = TemporalDataset('ICEWS14')
train_data = torch.from_numpy(dataset.get_train().astype('int64'))

sizes = (dataset.n_entities, dataset.n_predicates, dataset.n_entities, dataset.n_timestamps)

# Sample a batch
batch = train_data[:200].cuda()

print("=== Real Training Batch Test ===")
print(f"Batch shape: {batch.shape}")
print(f"Sizes: {sizes}")

# TNTComplEx
model_complex = TNTComplEx(sizes, rank=64).cuda()
model_complex.train()
predictions, factors_complex, time = model_complex.forward(batch)

print("\n=== TNTComplEx ===")
print(f"Factors shapes: {[f.shape for f in factors_complex]}")
for i, f in enumerate(factors_complex):
    print(f"Factor {i}: mean={f.mean().item():.6f}, max={f.max().item():.6f}, sum={f.sum().item():.2f}")

reg = N3(0.01)
l_reg_complex = reg.forward(factors_complex)
print(f"Regularization loss: {l_reg_complex.item():.6f}")

# TNTPairRE  
model_pairre = TNTPairRE(sizes, rank=64).cuda()
model_pairre.train()
predictions, factors_pairre, time = model_pairre.forward(batch)

print("\n=== TNTPairRE ===")
print(f"Factors shapes: {[f.shape for f in factors_pairre]}")
for i, f in enumerate(factors_pairre):
    print(f"Factor {i}: mean={f.mean().item():.6f}, max={f.max().item():.6f}, sum={f.sum().item():.2f}")

l_reg_pairre = reg.forward(factors_pairre)
print(f"Regularization loss: {l_reg_pairre.item():.6f}")
