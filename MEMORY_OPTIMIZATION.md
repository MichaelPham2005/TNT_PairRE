# TNTPairRE Memory Optimization Guide

## Vấn đề OOM (Out of Memory)

TNTPairRE sử dụng L1 distance nên cần tạo tensor `(batch, n_entities, rank)` trong forward pass, gây ra memory usage cao hơn TNTComplEx.

### Memory Usage So sánh:

**TNTComplEx (ComplEx-based):**

- Forward: `(batch, rank) @ (rank, n_entities)`
- Memory: O(batch × rank + rank × n_entities)
- ICEWS14: ~28MB cho embeddings

**TNTPairRE (L1 distance):**

- Forward: `(batch, n_entities, rank)` tensor
- Memory: O(batch × n_entities × rank)
- ICEWS14 với batch=1000, n_entities=7128, rank=100:
  - Tensor size: 1000 × 7128 × 100 × 4 bytes = **2.7 GB** chỉ cho 1 tensor!

## Giải pháp

### 1. Giảm Batch Size (Đơn giản nhất)

```bash
# Thay vì batch_size=1000
python tkbc/learner.py --dataset ICEWS14 --model TNTPairRE \
       --rank 100 --batch_size 200 \  # Giảm xuống 200
       --learning_rate 0.001
```

**Memory giảm từ 2.7GB → 0.54GB**

### 2. Giảm Rank

```bash
python tkbc/learner.py --dataset ICEWS14 --model TNTPairRE \
       --rank 50 \  # Thay vì 100
       --batch_size 500
```

### 3. Gradient Accumulation

Nếu muốn effective batch size lớn:

```python
# In optimizers.py, accumulate gradients over multiple mini-batches
effective_batch_size = 1000
mini_batch_size = 200
accumulation_steps = effective_batch_size // mini_batch_size  # 5

for step in range(accumulation_steps):
    loss = compute_loss(mini_batch)
    loss = loss / accumulation_steps
    loss.backward()  # Accumulate gradients

optimizer.step()  # Update after accumulation
optimizer.zero_grad()
```

### 4. Mixed Precision Training

```bash
# Use automatic mixed precision (FP16)
# Reduces memory by ~50%
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## Recommended Settings

### ICEWS14 (7128 entities)

```bash
python tkbc/learner.py --dataset ICEWS14 --model TNTPairRE \
       --rank 64 \
       --batch_size 256 \
       --learning_rate 0.001 \
       --emb_reg 0.00001 \
       --time_reg 0.0001 \
       --max_epochs 100
```

**Memory usage: ~0.6 GB**

### ICEWS05-15 (10094 entities)

```bash
python tkbc/learner.py --dataset ICEWS05-15 --model TNTPairRE \
       --rank 64 \
       --batch_size 200 \
       --learning_rate 0.001 \
       --emb_reg 0.00001 \
       --time_reg 0.001 \
       --max_epochs 100
```

**Memory usage: ~0.8 GB**

### YAGO15K (15403 entities)

```bash
python tkbc/learner.py --dataset yago15k --model TNTPairRE \
       --rank 64 \
       --batch_size 128 \
       --learning_rate 0.001 \
       --emb_reg 0.00001 \
       --time_reg 1.0 \
       --max_epochs 100 \
       --no_time_emb
```

**Memory usage: ~0.9 GB**

## Tại sao không thể dùng pure matrix multiplication như TNTComplEx?

**L1 Distance** không decompose được thành dot product:

- L1: `||a - b||₁ = Σ|aᵢ - bᵢ|` ← Cần tính absolute value
- L2: `||a - b||₂² = ||a||² + ||b||² - 2⟨a,b⟩` ← Có thể dùng dot product
- ComplEx: `Re(⟨h⊙r⊙t̄⟩)` ← Trực tiếp là dot product

Nếu đổi sang **L2 distance** hoặc **dot product scoring**, có thể tối ưu như TNTComplEx:

```python
# Alternative: TNTPairRE with L2 (memory efficient)
scores = (lhs.unsqueeze(1) * rhs).sum(dim=2)  # Dot product
# hoặc
scores = -((lhs.unsqueeze(1) - rhs) ** 2).sum(dim=2)  # Negative L2
```

Nhưng điều này thay đổi model semantics và có thể ảnh hưởng performance.

## Performance vs Memory Trade-off

| Config               | Memory  | Training Speed | Expected MRR |
| -------------------- | ------- | -------------- | ------------ |
| rank=100, batch=1000 | **OOM** | -              | -            |
| rank=100, batch=256  | 0.7 GB  | Medium         | Best         |
| rank=64, batch=512   | 0.8 GB  | Fast           | Good         |
| rank=50, batch=256   | 0.4 GB  | Fast           | Acceptable   |

## Monitoring Memory

```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

## Kết luận

TNTPairRE memory-intensive hơn TNTComplEx do bản chất của L1 distance. Để train thành công:

1. **Giảm batch_size** (khuyến nghị: 128-256)
2. **Giảm rank** nếu cần (64 thay vì 100)
3. **Monitor GPU memory** trong quá trình train
4. Xem xét **gradient accumulation** nếu cần effective batch size lớn
