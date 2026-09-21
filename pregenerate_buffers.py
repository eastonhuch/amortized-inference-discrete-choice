"""
Pre-generate replay buffers for multiple K values using Cholesky parameterization.

Usage:
    python pregenerate_buffers.py
    python pregenerate_buffers.py --k-min 3 --k-max 18 --train-buffer-size 500000
    python pregenerate_buffers.py --num-samples 500000 --output-dir my-data
    
File naming conventions:
    - Sobolev mode (default): {k}-train_cholesky.pt, {k}-eval_cholesky.pt
    - Sobolev + Gumbel:        {k}-train_cholesky_gumbel.pt, {k}-eval_cholesky_gumbel.pt
    - Non-Sobolev:             {k}-train_cholesky.pt, {k}-eval_cholesky.pt
    
These files are compatible with:
    - train_model() when lambda_grad is set (Sobolev mode; it looks in
      "simulation-data-checkpoints/")
    - train_model_disk() via DiskBackedBufferManagerCholesky (configurable data_dir)
"""

import os
import argparse
import torch
from tqdm import trange
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional

from utils import get_device
from training import ReplayBuffer, simulate_data


def buffer_to_cpu_dict(buffer: ReplayBuffer, use_sobolev: bool) -> dict:
    """Convert a ReplayBuffer to a CPU dictionary for saving.
    
    Args:
        buffer: ReplayBuffer with data on GPU
        use_sobolev: Whether gradients are stored
    
    Returns:
        Dictionary with all tensors moved to CPU
    """
    save_dict = {
        'v': buffer.v.cpu(),
        'L': buffer.L.cpu(),
        'y': buffer.y.cpu(),
        'size': buffer.size,
        'idx': buffer.idx,
    }
    if use_sobolev:
        save_dict['grad_v'] = buffer.grad_v.cpu()
        save_dict['grad_L'] = buffer.grad_L.cpu()
    return save_dict


def fill_buffer(
    buffer: ReplayBuffer,
    buffer_size: int,
    k: int,
    num_samples: int,
    num_factors: Optional[int],
    fill_batch_size: int,
    use_sobolev: bool,
    tau: float,
    gumbel: bool,
    desc: str,
) -> None:
    """Fill a Cholesky replay buffer with simulated data.
    
    Args:
        buffer: ReplayBuffer to fill
        buffer_size: Total number of examples to generate
        k: Number of alternatives
        num_samples: Monte Carlo samples for choice probabilities
        num_factors: Number of factors for covariance matrix (None for full-rank)
        fill_batch_size: Batch size for data generation
        use_sobolev: Whether to compute gradients
        tau: Temperature for soft gradients
        gumbel: Whether to use Gumbel distribution
        desc: Description for progress bar
    """
    num_batches = buffer_size // fill_batch_size
    remainder = buffer_size % fill_batch_size
    
    for _ in trange(num_batches, desc=desc):
        v, L, y, grad_v, grad_L = simulate_data(
            fill_batch_size, k, num_samples, x_only=False,
            compute_gradients=use_sobolev, tau=tau, gumbel=gumbel,
            num_factors=num_factors
        )
        buffer.add(v, L, y, grad_v, grad_L)
    
    if remainder > 0:
        v, L, y, grad_v, grad_L = simulate_data(
            remainder, k, num_samples, x_only=False,
            compute_gradients=use_sobolev, tau=tau, gumbel=gumbel,
            num_factors=num_factors
        )
        buffer.add(v, L, y, grad_v, grad_L)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate large Cholesky replay buffers for multiple K values"
    )
    parser.add_argument(
        "--k-min", type=int, default=2,
        help="Minimum K value (default: 2)"
    )
    parser.add_argument(
        "--k-max", type=int, default=18,
        help="Maximum K value (default: 18)"
    )
    parser.add_argument(
        "--train-buffer-size", type=int, default=200000,
        help="Number of examples for training buffer (default: 200000)"
    )
    parser.add_argument(
        "--eval-buffer-size", type=int, default=200000,
        help="Number of examples for evaluation buffer (default: 200000)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=1000000,
        help="Monte Carlo samples for choice probabilities (default: 1000000)"
    )
    parser.add_argument(
        "--num-factors", type=int, default=-1,
        help="Number of factors for covariance matrix (default: -1, which gives None for full-rank)"
    )
    parser.add_argument(
        "--tau", type=float, default=5.0,
        help="Temperature for soft gradients (default: 5.0)"
    )
    parser.add_argument(
        "--fill-batch-size", type=int, default=50,
        help="Batch size for data generation (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="simulation-data-checkpoints",
        help="Output directory (default: simulation-data-checkpoints)"
    )
    parser.add_argument(
        "--no-sobolev", action="store_true",
        help="Don't compute gradients (non-Sobolev mode)"
    )
    parser.add_argument(
        "--gumbel", action="store_true",
        help="Generate Gumbel data instead of Gaussian (probit)"
    )

    args = parser.parse_args()
    
    # Convert num_factors=-1 to None (full-rank)
    num_factors = args.num_factors if args.num_factors >= 0 else None
    
    # Enable cuDNN auto-tuner for optimized algorithms
    torch.backends.cudnn.benchmark = True
    
    # Enable TensorFloat32 for faster matmul on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    k_values = list(range(args.k_min, args.k_max + 1))
    use_sobolev = not args.no_sobolev
    
    # File naming follows DiskBackedBufferManagerCholesky:
    # - Sobolev mode: "_cholesky" or "_cholesky_gumbel"
    # - Non-Sobolev mode: "_cholesky"
    # (train_model's non-Sobolev mode uses no suffix, so non-Sobolev buffers written
    # here are read by train_model_disk only.)
    grad_suffix = "_cholesky" + ("_gumbel" if args.gumbel else "")
    
    device = get_device()
    
    print("=" * 60)
    print("Cholesky Buffer Generation")
    print("=" * 60)
    print(f"K values: {args.k_min} to {args.k_max} ({len(k_values)} values)")
    print(f"Train buffer size: {args.train_buffer_size:,}")
    print(f"Eval buffer size: {args.eval_buffer_size:,}")
    print(f"Num samples: {args.num_samples:,}")
    print(f"Num factors: {num_factors if num_factors is not None else 'None (full-rank)'}")
    print(f"Tau: {args.tau}")
    print(f"Fill batch size: {args.fill_batch_size}")
    print(f"Sobolev gradients: {use_sobolev}")
    print(f"Gumbel data: {args.gumbel}")
    print(f"Output directory: {args.output_dir}")
    print(f"File suffix: {grad_suffix}")
    print(f"Device: {device}")
    print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"TF32 matmul: {torch.get_float32_matmul_precision()}")
    print("=" * 60 + "\n")
    
    # Set up async I/O with buffer-level pipelining
    executor = ThreadPoolExecutor(max_workers=2)
    pending_save: Optional[Future] = None
    
    try:
        for k in k_values:
            train_path = os.path.join(args.output_dir, f"{k}-train{grad_suffix}.pt")
            eval_path = os.path.join(args.output_dir, f"{k}-eval{grad_suffix}.pt")
            
            # Check if already exists
            if os.path.exists(train_path) and os.path.exists(eval_path):
                print(f"K={k}: Buffers already exist, skipping.")
                continue
            
            print(f"K={k}: Generating buffers...")
            
            # ========== Train buffer ==========
            # Generate on GPU (may overlap with previous eval buffer's disk save)
            
            print(f"K={k}: Generating train buffer ({args.train_buffer_size:,} examples)...")
            train_buffer = ReplayBuffer(args.train_buffer_size, k, device, store_gradients=use_sobolev)
            
            fill_buffer(
                buffer=train_buffer,
                buffer_size=args.train_buffer_size,
                k=k,
                num_samples=args.num_samples,
                num_factors=num_factors,
                fill_batch_size=args.fill_batch_size,
                use_sobolev=use_sobolev,
                tau=args.tau,
                gumbel=args.gumbel,
                desc=f"K={k} train",
            )
            
            print(f"K={k}: Train buffer has {len(train_buffer):,} examples")
            
            # Wait for previous save to complete before transferring to CPU
            # This ensures only one buffer is in CPU memory at a time
            if pending_save is not None:
                pending_save.result()  # Block until done; raises exception if save failed
            
            # Transfer train buffer to CPU and start async save
            print(f"K={k}: Transferring train buffer to CPU and starting async save...")
            train_dict = buffer_to_cpu_dict(train_buffer, use_sobolev)
            pending_save = executor.submit(torch.save, train_dict, train_path)
            
            # Free GPU memory immediately
            del train_buffer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # ========== Eval buffer ==========
            # Generate on GPU (overlaps with train buffer's disk save)
            
            print(f"K={k}: Generating eval buffer ({args.eval_buffer_size:,} examples, train buffer saving in background)...")
            eval_buffer = ReplayBuffer(args.eval_buffer_size, k, device, store_gradients=use_sobolev)
            
            fill_buffer(
                buffer=eval_buffer,
                buffer_size=args.eval_buffer_size,
                k=k,
                num_samples=args.num_samples,
                num_factors=num_factors,
                fill_batch_size=args.fill_batch_size,
                use_sobolev=use_sobolev,
                tau=args.tau,
                gumbel=args.gumbel,
                desc=f"K={k} eval",
            )
            
            print(f"K={k}: Eval buffer has {len(eval_buffer):,} examples")
            
            # Wait for train buffer save to complete before transferring eval to CPU
            if pending_save is not None:
                pending_save.result()
            
            # Transfer eval buffer to CPU and start async save
            print(f"K={k}: Transferring eval buffer to CPU and starting async save...")
            eval_dict = buffer_to_cpu_dict(eval_buffer, use_sobolev)
            pending_save = executor.submit(torch.save, eval_dict, eval_path)
            
            # Free GPU memory
            del eval_buffer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"K={k}: Generation complete (eval buffer saving in background)\n")
        
        # Wait for final save to complete
        if pending_save is not None:
            print("Waiting for final save to complete...")
            pending_save.result()
        
    finally:
        # Ensure executor shuts down cleanly even if there's an exception
        executor.shutdown(wait=True)
    
    print("=" * 60)
    print("Done! All buffers generated.")
    print("=" * 60)
    
    # Print summary of generated files
    print("\nGenerated files:")
    total_size = 0
    for k in k_values:
        train_path = os.path.join(args.output_dir, f"{k}-train{grad_suffix}.pt")
        eval_path = os.path.join(args.output_dir, f"{k}-eval{grad_suffix}.pt")
        if os.path.exists(train_path) and os.path.exists(eval_path):
            train_size = os.path.getsize(train_path) / (1024 * 1024)  # MB
            eval_size = os.path.getsize(eval_path) / (1024 * 1024)  # MB
            total_size += train_size + eval_size
            print(f"  K={k}: train={train_size:.1f}MB, eval={eval_size:.1f}MB")
    print(f"\nTotal disk usage: {total_size:.1f}MB ({total_size/1024:.2f}GB)")


if __name__ == "__main__":
    main()
