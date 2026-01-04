#!/bin/bash

# Baseline PairRE Training Script for ICEWS14
# NO temporal modeling - timestamps ignored!

echo "=================================================="
echo "BASELINE PairRE - Static KG Approach"
echo "Dataset: ICEWS14 (ignoring timestamps)"
echo "=================================================="

# Step 1: Download and prepare data (if not exists)
if [ ! -d "processed" ]; then
    echo ""
    echo "Downloading ICEWS14 dataset from Facebook AI Research..."
    bash download_dataset.sh
    
    if [ $? -ne 0 ]; then
        echo "❌ Data download failed!"
        exit 1
    fi
else
    echo "✓ Data already prepared in ./processed"
fi

# Configuration
DATA_PATH="processed"
MODEL="BaselinePairRE"
SAVE_DIR="checkpoints/ICEWS14_BaselinePairRE"

# Hyperparameters (matching temporal model for fair comparison)
DIMENSION=500
GAMMA=28.0
LR=0.0001
BATCH_SIZE=512
NEG_SIZE=128
ADV_TEMP=1.0
REG=0.000001
MAX_STEPS=100000
WARMUP=50000
VALID_STEPS=5000
SAVE_STEPS=5000

echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Dimension: $DIMENSION"
echo "  Gamma: $GAMMA"
echo "  Learning Rate: $LR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Negative Samples: $NEG_SIZE"
echo "  Max Steps: $MAX_STEPS"
echo ""
echo "NOTE: This baseline IGNORES timestamps during training!"
echo "      Evaluation uses temporal filtering for fair comparison."
echo ""

# Run training
python -u run.py \
  --do_train \
  --cuda \
  --do_valid \
  --do_test \
  --evaluate_train \
  --model $MODEL \
  --data_path $DATA_PATH \
  -n $NEG_SIZE \
  -b $BATCH_SIZE \
  -d $DIMENSION \
  -g $GAMMA \
  -a $ADV_TEMP \
  -adv \
  -dr \
  -r $REG \
  -lr $LR \
  --max_steps $MAX_STEPS \
  --warm_up_steps $WARMUP \
  --cpu_num 2 \
  --test_batch_size 32 \
  --valid_steps $VALID_STEPS \
  --log_steps 100 \
  --save_checkpoint_steps $SAVE_STEPS \
  --save_path $SAVE_DIR

echo ""
echo "=================================================="
echo "Training Complete!"
echo "Check results in: $SAVE_DIR"
echo "=================================================="
