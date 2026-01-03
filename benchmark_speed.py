"""
Benchmark training speed: TNTComplEx vs TNTPairRE

Measures time per epoch to compare performance.
"""

import sys
sys.path.insert(0, 'tkbc')

import torch
import time
from torch import optim
from tkbc.models import TNTComplEx, TNTPairRE
from tkbc.regularizers import N3, Lambda3, TemporalSmoothness
from tkbc.optimizers import TKBCOptimizer


def benchmark_model(model_class, model_name, sizes, rank, n_samples=10000, batch_size=1000):
    """Benchmark a model's training speed"""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    print(f"Dataset size: {sizes}")
    print(f"Rank: {rank}, Batch size: {batch_size}, Samples: {n_samples}")
    
    # Initialize model
    if model_name == "TNTPairRE":
        model = model_class(sizes, rank)
    else:
        model = model_class(sizes, rank, no_time_emb=False)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        device_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {device_name}")
    else:
        print("Using CPU")
    
    # Create synthetic data
    data = torch.randint(0, sizes[0], (n_samples, 4))
    data[:, 1] = torch.randint(0, sizes[1], (n_samples,))
    data[:, 3] = torch.randint(0, sizes[3], (n_samples,))
    
    # Setup optimizer and regularizers
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    emb_reg = N3(weight=0.0001)
    
    if model_name == "TNTPairRE":
        time_reg = TemporalSmoothness(weight=0.0001)
    else:
        time_reg = Lambda3(weight=0.0001)
    
    # Create optimizer wrapper
    tkbc_opt = TKBCOptimizer(
        model, emb_reg, time_reg, optimizer,
        batch_size=batch_size,
        verbose=False  # Disable progress bar for accurate timing
    )
    
    # Warmup (exclude from timing)
    print("\nWarmup run...")
    tkbc_opt.epoch(data[:batch_size])
    
    # Benchmark: 3 epochs
    print("Timing 3 epochs...")
    times = []
    
    for epoch in range(3):
        start_time = time.time()
        tkbc_opt.epoch(data)
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        print(f"  Epoch {epoch+1}: {epoch_time:.3f}s")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage time per epoch: {avg_time:.3f}s")
    print(f"Samples per second: {n_samples/avg_time:.0f}")
    
    # Memory stats if CUDA
    if torch.cuda.is_available():
        print(f"\nGPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    
    return avg_time


def main():
    print("\n" + "#"*60)
    print("# Training Speed Benchmark")
    print("#"*60)
    
    # ICEWS14-like dataset
    sizes = (7128, 230, 7128, 365)  # entities, relations, entities, timestamps
    rank = 100
    n_samples = 10000  # Smaller for quick benchmark
    batch_size = 1000
    
    print(f"\nDataset configuration (ICEWS14-like):")
    print(f"  Entities: {sizes[0]}")
    print(f"  Relations: {sizes[1]}")
    print(f"  Timestamps: {sizes[3]}")
    print(f"  Training samples: {n_samples}")
    
    # Benchmark TNTComplEx
    time_tntcomplex = benchmark_model(
        TNTComplEx, "TNTComplEx", sizes, rank, n_samples, batch_size
    )
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Benchmark TNTPairRE
    time_tntpaire = benchmark_model(
        TNTPairRE, "TNTPairRE", sizes, rank, n_samples, batch_size
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"TNTComplEx:  {time_tntcomplex:.3f}s per epoch")
    print(f"TNTPairRE:   {time_tntpaire:.3f}s per epoch")
    print(f"Slowdown:    {time_tntpaire/time_tntcomplex:.2f}x")
    
    if time_tntpaire/time_tntcomplex > 10:
        print("\n⚠️  TNTPairRE is significantly slower!")
        print("Recommendations:")
        print("  - Reduce batch_size (try 200-500)")
        print("  - Reduce rank (try 64 instead of 100)")
        print("  - Consider using smaller chunk_size in forward()")
    elif time_tntpaire/time_tntcomplex < 3:
        print("\n✓ TNTPairRE performance is acceptable")
    else:
        print("\n⚠️  TNTPairRE is slower but manageable")
        print("Consider reducing batch_size if OOM occurs")


if __name__ == "__main__":
    main()
