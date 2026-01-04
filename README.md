# Baseline PairRE for ICEWS14

> 📖 **[Đọc hướng dẫn tiếng Việt chi tiết tại đây](README_Vietnamese.md)** | Vietnamese Guide Available

## Overview

**Self-contained baseline PairRE** that **ignores timestamps** when training on ICEWS14.

## Quick Start (2 Steps!)

### 1. Download Dataset
```bash
bash download_dataset.sh
```

This downloads ICEWS14 from Facebook AI Research (TKBC dataset).

### 2. Train
```bash
bash train_baseline.sh
```

Or manually:
```bash
python run.py --do_train --cuda --do_valid --do_test \
  --model BaselinePairRE \
  --data_path processed \
  -n 128 -b 512 -d 500 -g 28.0 -adv -dr \
  --max_steps 100000 --save_path checkpoints/baseline
```

**That's it!** 2 commands total.

### 3. Check Results
```bash
cat checkpoints/ICEWS14_BaselinePairRE/train.log | grep "Test MRR"
```

## For Kaggle

```python
# Cell 1: Setup
import shutil, os
shutil.copytree("/kaggle/input/baseline-icews14", "/kaggle/working/my_code")
os.chdir("/kaggle/working/my_code")

# Cell 2: Download data
!bash download_dataset.sh

# Cell 3: Train
!bash train_baseline.sh

# Cell 4: Results
!cat checkpoints/ICEWS14_BaselinePairRE/train.log | grep "Test MRR"
```

## Expected Results

**Baseline PairRE (no time):**
- Test MRR: ~42-45%
- HITS@10: ~62-65%

**Temporal PairRE (with time):**
- Test MRR: ~50-53%
- HITS@10: ~70-73%

**Improvement: +8-10% MRR**

## Hyperparameters (Matched)

Both models use identical hyperparameters for fair comparison:
- Dimension: 500
- Gamma: 28.0
- Learning Rate: 0.0001
- Batch Size: 512
- Negative Samples: 128
- Training Steps: 100k
- Warm-up: 50k steps

## Evaluation Protocol

**Critical**: Both models evaluate with **temporal filtering**
- Filters out `(h,r,t,t')` where `t' ≠ t_test`
- Ensures baseline doesn't get credit for wrong timestamps
- Fair comparison of temporal vs non-temporal modeling

## Comparison with Temporal Model

After training both models, compare:

```bash
# Baseline results
cat checkpoints/ICEWS14_BaselinePairRE/train.log | grep "Test MRR"

# Temporal results  
cat ../temporal/checkpoints/ICEWS14_TemporalPairRE/train.log | grep "Test MRR"
```

The temporal model should significantly outperform the baseline, proving the value of temporal modeling!
