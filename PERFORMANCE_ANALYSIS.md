# TNTPairRE Performance Analysis

## Tại sao TNTPairRE chậm hơn TNTComplEx 80x?

### Root Cause: L1 Distance vs Matrix Multiplication

**TNTComplEx (Fast - 0.2s/epoch)**:

```python
# Single optimized matrix multiplication
scores = (lhs * rel * time) @ all_entities.T
# → cuBLAS optimized, single kernel call
```

**TNTPairRE (Slow - 17s/epoch)**:

```python
# Multiple operations, large intermediate tensors
rhs = entities * r_T_l  # Broadcasting: (batch, n_entities, rank)
diff = lhs - rhs  # Subtraction: (batch, n_entities, rank)
scores = -sum(abs(diff))  # abs() + sum: not optimized like matmul
```

### Performance Bottlenecks

1. **Memory Bandwidth**:

   - Must create `(batch, n_entities, rank)` tensor = 2.7GB for ICEWS14
   - Reading/writing this much data is slow (~17s just for memory I/O)

2. **Kernel Launch Overhead**:

   - Matrix multiply: 1 optimized cuBLAS call
   - L1 distance: Multiple kernels (broadcast → subtract → abs → sum)

3. **No Hardware Optimization**:
   - Matrix multiply uses Tensor Cores on modern GPUs
   - Element-wise abs() uses standard CUDA cores

### Comparison Table

| Operation        | TNTComplEx           | TNTPairRE                   |
| ---------------- | -------------------- | --------------------------- |
| Main computation | `lhs @ rhs.T`        | `sum(abs(lhs - rhs))`       |
| Temp tensor size | `(batch, rank)`      | `(batch, n_entities, rank)` |
| GPU optimization | cuBLAS, Tensor Cores | Standard CUDA               |
| Time per epoch   | **0.2s**             | **17s**                     |
| Memory           | 0.17 GB              | 5.46 GB                     |
| Speedup          | **80x faster**       | -                           |

## Solutions

### Option 1: Accept the Slowdown (Recommended)

TNTPairRE is inherently slower due to L1 distance. Use appropriate hyperparameters:

```bash
# Use smaller batch size
python tkbc/learner.py --dataset ICEWS14 --model TNTPairRE \
       --rank 100 --batch_size 256 \  # Instead of 1000
       --learning_rate 0.001
```

**Time per epoch: ~4-5s** (acceptable)

### Option 2: Reduce Rank

```bash
python tkbc/learner.py --dataset ICEWS14 --model TNTPairRE \
       --rank 64 \  # Instead of 100
       --batch_size 512
```

**Time per epoch: ~3s**

### Option 3: Use L2 Distance (Modify Model)

Replace L1 with negative squared L2 distance to enable matrix multiplication:

```python
# In forward():
# Current (slow):
scores = -torch.sum(torch.abs(lhs - rhs), dim=2)

# Alternative (fast):
# -||lhs - rhs||^2 = 2*lhs@rhs.T - ||lhs||^2 - ||rhs||^2
scores = 2 * torch.bmm(lhs.unsqueeze(1), rhs.transpose(1,2)).squeeze(1)
scores = scores - torch.sum(lhs**2, 1, keepdim=True) - torch.sum(rhs**2, 2)
```

This makes TNTPairRE as fast as TNTComplEx but **changes the model** (may affect accuracy).

### Option 4: Mixed Approach

Use L1 for `score()` (evaluation) but L2 for `forward()` (training):

```python
def forward(self, x):
    # Fast L2-based scoring for training
    ...

def score(self, x):
    # Exact L1 distance for evaluation
    ...
```

## Recommended Configuration

For ICEWS14 (7128 entities):

```bash
python tkbc/learner.py --dataset ICEWS14 --model TNTPairRE \
       --rank 64 \
       --batch_size 256 \
       --learning_rate 0.001 \
       --emb_reg 0.00001 \
       --time_reg 0.0001 \
       --max_epochs 100
```

**Expected performance:**

- Time per epoch: ~3-5s (vs 0.2s for TNTComplEx)
- Total training time: ~5-8 minutes (vs 20 seconds for TNTComplEx)
- This is **acceptable** for research purposes

For ICEWS05-15 (10094 entities):

```bash
--rank 64 --batch_size 200
```

For YAGO15K (15403 entities):

```bash
--rank 64 --batch_size 128
```

## Conclusion

**TNTPairRE will always be slower than TNTComplEx** due to the fundamental difference between L1 distance and matrix multiplication. This is not a bug—it's an inherent property of the model.

**Acceptable slowdown**: 10-20x (with proper hyperparameters)
**Current slowdown**: 80x (with batch_size=1000, need to reduce)

The model is still usable for research, just requires patience and proper configuration.
