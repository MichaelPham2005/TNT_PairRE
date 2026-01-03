"""
Demo script for TNTPairRE model training on ICEWS14

This script demonstrates how to train the TNTPairRE model on a real dataset.
For quick testing, it runs for fewer epochs.

Usage:
    python demo_tnt_paire.py
"""

import sys
sys.path.insert(0, 'tkbc')

import torch
from torch import optim
from tkbc.models import TNTPairRE
from tkbc.datasets import TemporalDataset
from tkbc.regularizers import N3, TemporalSmoothness
from tkbc.optimizers import TKBCOptimizer


def main():
    print("\n" + "="*60)
    print("TNTPairRE Demo - Training on ICEWS14")
    print("="*60)
    
    # Hyperparameters
    dataset_name = 'ICEWS14'
    rank = 100
    learning_rate = 0.001
    emb_reg_weight = 0.00001
    time_reg_weight = 0.0001
    batch_size = 1000
    max_epochs = 10  # For demo purposes
    
    print(f"\nHyperparameters:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Rank: {rank}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Embedding regularization: {emb_reg_weight}")
    print(f"  Temporal regularization: {time_reg_weight}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    
    # Load dataset
    print(f"\nLoading {dataset_name} dataset...")
    try:
        dataset = TemporalDataset(dataset_name)
        sizes = dataset.get_shape()
        print(f"Dataset loaded successfully!")
        print(f"  Entities: {sizes[0]}")
        print(f"  Relations: {sizes[1]}")
        print(f"  Timestamps: {sizes[3]}")
        print(f"  Training samples: {len(dataset.get_train())}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"\nMake sure you have run the data preprocessing:")
        print(f"  cd tkbc")
        print(f"  python process_icews.py")
        return
    
    # Initialize model
    print(f"\nInitializing TNTPairRE model...")
    model = TNTPairRE(sizes, rank, no_time_emb=False)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  Using CPU")
    
    # Optimizer and regularizers
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    emb_reg = N3(emb_reg_weight)
    time_reg = TemporalSmoothness(time_reg_weight)
    
    print(f"\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    # Training loop
    for epoch in range(max_epochs):
        examples = torch.from_numpy(dataset.get_train().astype('int64'))
        
        model.train()
        tkbc_optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, optimizer,
            batch_size=batch_size,
            verbose=True
        )
        tkbc_optimizer.epoch(examples)
        
        print(f"\nEpoch {epoch+1}/{max_epochs} completed")
        
        # Validation every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEvaluating on validation set...")
            try:
                valid_mrr, valid_hits = dataset.eval(model, 'valid', 5000)
                valid_both = {
                    'MRR': (valid_mrr['lhs'] + valid_mrr['rhs']) / 2,
                    'hits@[1,3,10]': (valid_hits['lhs'] + valid_hits['rhs']) / 2
                }
                print(f"  Valid MRR: {valid_both['MRR']:.4f}")
                print(f"  Valid Hits@1: {valid_both['hits@[1,3,10]'][0]:.4f}")
                print(f"  Valid Hits@3: {valid_both['hits@[1,3,10]'][1]:.4f}")
                print(f"  Valid Hits@10: {valid_both['hits@[1,3,10]'][2]:.4f}")
            except Exception as e:
                print(f"  Validation error: {e}")
    
    print(f"\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    print(f"\nTo train for full epochs, use:")
    print(f"  cd tkbc")
    print(f"  python learner.py --dataset {dataset_name} --model TNTPairRE \\")
    print(f"         --rank {rank} --learning_rate {learning_rate} \\")
    print(f"         --emb_reg {emb_reg_weight} --time_reg {time_reg_weight} \\")
    print(f"         --max_epochs 100 --batch_size {batch_size}")


if __name__ == "__main__":
    main()
