# training.py

import os
import copy
from datetime import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
try:
    from IPython.display import display, clear_output
except ImportError:  # outside IPython; live plotting (plot=True) is meant for Jupyter
    def display(obj):
        print(obj)

    def clear_output(wait=False):
        pass
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from utils import get_device, canonicalize, compute_d_sigma_dt, omega_to_cov
from emulator import Emulator


def _build_L_row_scales(km1, device, dtype=torch.float32):
    """
    Build scaling factors for Cholesky factor rows.
    
    Row j (0-indexed) has j+1 entries, so we scale by 1/sqrt(j+1).
    
    Args:
        km1: K-1, the dimension of L
        device: torch device
        dtype: data type
        
    Returns:
        scales: tensor of shape (num_L_params,) where num_L_params = km1*(km1+1)//2
    """
    scales = []
    for j in range(km1):  # j is 0-indexed row
        row_scale = 1.0 / np.sqrt(j + 1)
        scales.extend([row_scale] * (j + 1))  # j+1 entries in row j
    return torch.tensor(scales, device=device, dtype=dtype)


def _compute_sqrt_L(L, gumbel=False):
    """
    Compute the matrix square root of Sigma given Cholesky factor L of Omega.
    
    For probit (Gaussian): The Cholesky factor of Sigma has a block structure
        with 1 in the (0,0) position and L in the lower-right block.
    For Gumbel: Need symmetric square root, computed via eigendecomposition.
    
    Args:
        L: Lower triangular Cholesky factor of Omega, shape (batch, km1, km1)
        gumbel: Whether to compute symmetric square root
        
    Returns:
        sqrt_Sigma: Matrix square root of Sigma, shape (batch, k, k)
        is_valid: Boolean mask of valid samples, shape (batch,)
    """
    batch_size, km1, _ = L.shape
    k = km1 + 1
    device = L.device
    dtype = L.dtype
    
    # Build Omega = L @ L^T
    Omega = L @ L.transpose(-1, -2)
    
    # Build full Sigma
    Sigma = torch.zeros(batch_size, k, k, device=device, dtype=dtype)
    Sigma[:, 0, 0] = 1.0
    Sigma[:, 1:, 1:] = Omega
    
    if gumbel:
        # Need symmetric square root via eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
        except torch.linalg.LinAlgError:
            # Eigendecomposition failed due to ill-conditioned matrix
            # Mark all samples as invalid and return placeholder
            sqrt_Sigma = torch.zeros(batch_size, k, k, device=device, dtype=dtype)
            is_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
            return sqrt_Sigma, is_valid
        
        # Check validity: eigenvalues should be non-negative and finite
        eigenvalues_valid = (eigenvalues >= -1e-6).all(dim=-1) & (eigenvalues <= 1e8).all(dim=-1)
        
        # Clamp and compute sqrt
        eigenvalues_clamped = torch.clamp(eigenvalues, min=0.0)
        sqrt_eigenvalues = torch.sqrt(eigenvalues_clamped)
        
        # Sigma^{1/2} = Q @ diag(sqrt(lambda)) @ Q^T
        sqrt_Sigma = eigenvectors * sqrt_eigenvalues.unsqueeze(-2) @ eigenvectors.transpose(-1, -2)
        
        # Check for numerical issues
        is_finite = torch.isfinite(sqrt_Sigma).all(dim=(-2, -1))
        is_valid = eigenvalues_valid & is_finite
    else:
        # For probit, use the block Cholesky structure directly
        # chol(Sigma) = [[1, 0], [0, L]] when Sigma = [[1, 0], [0, Omega]] and Omega = L @ L^T
        sqrt_Sigma = torch.zeros(batch_size, k, k, device=device, dtype=dtype)
        sqrt_Sigma[:, 0, 0] = 1.0
        sqrt_Sigma[:, 1:, 1:] = L
        
        # Check that L has positive diagonal (valid Cholesky)
        L_diag = torch.diagonal(L, dim1=-2, dim2=-1)
        is_valid = (L_diag > 0).all(dim=-1) & torch.isfinite(L).all(dim=(-2, -1))
    
    return sqrt_Sigma, is_valid


def simulate_data(batch_size, k, num_samples, x_only=False, dtype=torch.float32,
                           compute_gradients=False, tau=5.0, gumbel=False, num_factors=None):
    """
    Simulate data for training using Cholesky parameterization.
    
    Args:
        batch_size: Number of parameter settings to generate
        k: Number of alternatives
        num_samples: Number of choice samples per parameter setting
        x_only: If True, only return inputs (skip choice simulation)
        dtype: Data type
        compute_gradients: If True, compute gradients of log-probabilities
        tau: Temperature parameter for softmax approximation
        gumbel: Whether to use Gumbel errors (requires symmetric sqrt)
        num_factors: Number of factors for factor model specification. If None, use full-rank Wishart.
    
    Returns:
        v: Centered utilities, shape (batch, K)
        L: Lower triangular Cholesky factor of Omega, shape (batch, K-1, K-1)
        y: Choice frequencies, shape (batch, K)
        grad_v: Jacobian of log P w.r.t. v, shape (batch, K, K)
        grad_L: Jacobian of log P w.r.t. vech(L), shape (batch, K, num_L_params)
    """
    device = get_device()
    km1 = k - 1
    num_L_params = km1 * (km1 + 1) // 2
    df = k + 1
    
    def sample_omega(factors):
        """Sample Omega matrix according to specified factor structure."""
        if factors is None:
            # Full-rank Wishart-like construction
            z = torch.randn(batch_size, km1, df, device=device, dtype=dtype)
            Omega = z @ z.transpose(-1, -2) / df
        else:
            # Factor model: Omega = Z @ Z^T / f + diag(uniquenesses)
            z = torch.randn(batch_size, km1, factors, device=device, dtype=dtype)
            
            # Randomly sample number of factors for each sample: 0, 1, 2, ..., factors
            num_factors_per_sample = torch.randint(0, factors + 1, (batch_size,), device=device)
            
            # Create mask to zero out columns beyond num_factors for each sample
            factor_indices = torch.arange(factors, device=device).unsqueeze(0)  # (1, factors)
            mask = (factor_indices < num_factors_per_sample.unsqueeze(1)).unsqueeze(1)  # (batch, 1, factors)
            z = z * mask  # broadcast to (batch, km1, factors)
            
            # Compute outer product, dividing by num_factors (handle f=0 case)
            outer = z @ z.transpose(-1, -2)  # (batch, km1, km1)
            divisor = num_factors_per_sample.float().clamp(min=1).view(batch_size, 1, 1)
            Omega = outer / divisor
            
            # Sample uniquenesses from chi-squared(1)
            z_uniquenesses = torch.randn(batch_size, km1, device=device, dtype=dtype)
            uniquenesses = z_uniquenesses * z_uniquenesses
            
            # Add uniquenesses to diagonal of Omega
            Omega = Omega + torch.diag_embed(uniquenesses)
        
        return Omega
    
    # Sample Omega and compute its Cholesky factor L
    Omega = sample_omega(factors=num_factors)
    L, info = torch.linalg.cholesky_ex(Omega)
    chol_valid = (info == 0)
    
    # Sample a second Omega for generating diverse v (always full-rank for mixing)
    Omega_prime = sample_omega(factors=None)
    
    # Form convex combination for v generation
    lam = torch.rand(batch_size, 1, 1, device=device, dtype=dtype)
    Omega_mix = lam * Omega + (1 - lam) * Omega_prime
    
    # Build full Sigma_mix for sampling v
    Sigma_mix = torch.zeros(batch_size, k, k, device=device, dtype=dtype)
    Sigma_mix[:, 0, 0] = 1.0
    Sigma_mix[:, 1:, 1:] = Omega_mix
    
    chol_mix, info_mix = torch.linalg.cholesky_ex(Sigma_mix)
    chol_mix_valid = (info_mix == 0)
    
    # Sample centered utilities
    v_independent = 2.0 * torch.randn(batch_size, k, device=device, dtype=dtype)
    v = (chol_mix @ v_independent.unsqueeze(-1)).squeeze(-1)
    v = v - v.mean(dim=-1, keepdim=True)  # Center (optional)
    
    # Compute matrix square root for simulation
    sqrt_Sigma, sqrt_valid = _compute_sqrt_L(L, gumbel=gumbel)
    
    # Combined validity mask
    is_valid = chol_valid & chol_mix_valid & sqrt_valid
    
    # Filter to valid samples
    v = v[is_valid]
    L = L[is_valid]
    sqrt_Sigma = sqrt_Sigma[is_valid]
    
    num_valid = v.shape[0]
    
    if num_valid == 0:
        empty_v = torch.zeros(0, k, device=device, dtype=dtype)
        empty_L = torch.zeros(0, km1, km1, device=device, dtype=dtype)
        empty_y = torch.zeros(0, k, device=device, dtype=dtype)
        empty_grad_v = torch.zeros(0, k, k, device=device, dtype=dtype) if compute_gradients else None
        empty_grad_L = torch.zeros(0, k, num_L_params, device=device, dtype=dtype) if compute_gradients else None
        return empty_v, empty_L, empty_y, empty_grad_v, empty_grad_L
    
    if x_only:
        return v, L, None, None, None
    
    # Enable gradients if needed
    if compute_gradients:
        v = v.detach().clone().requires_grad_(True)
        L = L.detach().clone().requires_grad_(True)
        # Recompute sqrt_Sigma with gradient tracking
        sqrt_Sigma, _ = _compute_sqrt_L(L, gumbel=gumbel)
    
    # Sample errors
    if gumbel:
        euler_mascheroni = 0.57721566490153286060651209
        gumbel_dist = torch.distributions.Gumbel(
            loc=torch.tensor(-euler_mascheroni, device=device, dtype=dtype),
            scale=torch.tensor(1.0, device=device, dtype=dtype)
        )
        z_samples = gumbel_dist.sample((num_valid, num_samples, k)).detach()
    else:
        z_samples = torch.randn(num_valid, num_samples, k, device=device, dtype=dtype).detach()
    
    # Transform errors: epsilon = sqrt_Sigma @ z
    errors = torch.einsum('bij,bsj->bsi', sqrt_Sigma, z_samples)
    
    # Utilities: U = v + epsilon
    U = v.unsqueeze(1) + errors
    
    # Hard choices for probability estimation (detached)
    with torch.no_grad():
        U_detached = U.detach()
        max_U = torch.max(U_detached, dim=2, keepdim=True)[0]
        is_max = (U_detached == max_U).to(dtype)
        y = torch.mean(is_max, dim=1)
    
    # Compute gradients if requested
    grad_v = None
    grad_L = None
    
    if compute_gradients:
        grad_v, grad_L = _compute_gradients(v, L, U, num_samples, k, tau, dtype, device, gumbel)
    
    return v.detach(), L.detach(), y, grad_v, grad_L


def _compute_gradients(v, L, U, num_samples, k, tau, dtype, device, gumbel):
    """
    Compute gradients of log P_j with respect to v and L.
    
    Args:
        v: Centered utilities with requires_grad=True, shape (batch, K)
        L: Cholesky factor with requires_grad=True, shape (batch, K-1, K-1)
        U: Realized utilities, shape (batch, num_samples, K)
        num_samples: Number of MC samples
        k: Number of alternatives
        tau: Temperature for soft choices
        dtype: Data type
        device: Device
        gumbel: Whether using Gumbel errors
        
    Returns:
        grad_v: (batch, K, K) Jacobian
        grad_L: (batch, K, num_L_params) Jacobian
    """
    km1 = k - 1
    num_L_params = km1 * (km1 + 1) // 2
    tril_idx = torch.tril_indices(km1, km1, device=device)
    
    num_valid = v.shape[0]
    
    # Soft choices
    log_soft_choices = F.log_softmax(tau * U, dim=-1)
    log_mean_probs = torch.logsumexp(log_soft_choices, dim=1) - np.log(num_samples)
    
    grad_v_list = []
    grad_L_list = []
    
    for j in range(k):
        log_prob_j = log_mean_probs[:, j]
        grads = torch.autograd.grad(
            outputs=log_prob_j.sum(),
            inputs=(v, L),
            retain_graph=(j < k - 1),
            create_graph=False
        )
        grad_v_list.append(grads[0])
        # Extract lower triangular part of L gradient
        grad_L_list.append(grads[1][:, tril_idx[0], tril_idx[1]])
    
    grad_v = torch.stack(grad_v_list, dim=1)  # (batch, K, K)
    grad_L = torch.stack(grad_L_list, dim=1)  # (batch, K, num_L_params)
    
    return grad_v.detach(), grad_L.detach()


class ReplayBuffer:
    """Replay buffer for Cholesky-parameterized training data."""
    
    def __init__(self, max_size, k, device, store_gradients=False):
        self.max_size = max_size
        self.k = k
        self.km1 = k - 1
        self.device = device
        self.size = 0
        self.idx = 0
        self.store_gradients = store_gradients
        
        num_L_params = self.km1 * (self.km1 + 1) // 2
        
        self.v = torch.zeros(max_size, k, device=device)
        self.L = torch.zeros(max_size, self.km1, self.km1, device=device)
        self.y = torch.zeros(max_size, k, device=device)
        
        if store_gradients:
            self.grad_v = torch.zeros(max_size, k, k, device=device)
            self.grad_L = torch.zeros(max_size, k, num_L_params, device=device)
        else:
            self.grad_v = None
            self.grad_L = None
    
    def __len__(self):
        return self.size
    
    def add(self, v, L, y, grad_v=None, grad_L=None):
        """Add examples to the buffer."""
        batch_size = v.shape[0]
        
        if batch_size == 0:
            return
        
        if batch_size >= self.max_size:
            self.v[:] = v[-self.max_size:]
            self.L[:] = L[-self.max_size:]
            self.y[:] = y[-self.max_size:]
            if self.store_gradients and grad_v is not None and grad_L is not None:
                self.grad_v[:] = grad_v[-self.max_size:]
                self.grad_L[:] = grad_L[-self.max_size:]
            self.size = self.max_size
            self.idx = 0
            return
        
        end_idx = self.idx + batch_size
        
        if end_idx <= self.max_size:
            self.v[self.idx:end_idx] = v
            self.L[self.idx:end_idx] = L
            self.y[self.idx:end_idx] = y
            if self.store_gradients and grad_v is not None and grad_L is not None:
                self.grad_v[self.idx:end_idx] = grad_v
                self.grad_L[self.idx:end_idx] = grad_L
        else:
            first_part = self.max_size - self.idx
            self.v[self.idx:] = v[:first_part]
            self.L[self.idx:] = L[:first_part]
            self.y[self.idx:] = y[:first_part]
            
            second_part = batch_size - first_part
            self.v[:second_part] = v[first_part:]
            self.L[:second_part] = L[first_part:]
            self.y[:second_part] = y[first_part:]
            
            if self.store_gradients and grad_v is not None and grad_L is not None:
                self.grad_v[self.idx:] = grad_v[:first_part]
                self.grad_L[self.idx:] = grad_L[:first_part]
                self.grad_v[:second_part] = grad_v[first_part:]
                self.grad_L[:second_part] = grad_L[first_part:]
        
        self.idx = end_idx % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
    
    def sample(self, batch_size):
        """Sample a batch of examples from the buffer."""
        if self.size == 0:
            raise ValueError("Buffer is empty")
        
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        v = self.v[indices]
        L = self.L[indices]
        y = self.y[indices]
        
        if self.store_gradients:
            grad_v = self.grad_v[indices]
            grad_L = self.grad_L[indices]
        else:
            grad_v = None
            grad_L = None
        
        return v, L, y, grad_v, grad_L
    

def generate_directions(batch_size, k, num_directions, device, dtype=torch.float32):
    """
    Generate random directions in (v, L) space on-the-fly.
    
    Args:
        batch_size: Number of direction sets to generate
        k: Number of alternatives
        num_directions: Number of directions per sample
        device: Torch device
        dtype: Data type
        
    Returns:
        dir_v: Direction for v, shape (batch, num_directions, K), centered
        dir_L: Direction for L (lower triangular entries), shape (batch, num_directions, num_L_params)
    """
    km1 = k - 1
    num_L_params = km1 * (km1 + 1) // 2
    
    # Direction for v: sample and center
    dir_v = torch.randn(batch_size, num_directions, k, device=device, dtype=dtype)
    dir_v = dir_v - dir_v.mean(dim=-1, keepdim=True)
    
    # Direction for L: sample with row scaling
    dir_L_raw = torch.randn(batch_size, num_directions, num_L_params, device=device, dtype=dtype)
    
    # Apply row scaling: entries in row j (0-indexed) get scaled by 1/sqrt(j+1)
    row_scales = _build_L_row_scales(km1, device, dtype)  # (num_L_params,)
    dir_L = dir_L_raw * row_scales.unsqueeze(0).unsqueeze(0)
    
    # Normalize the full direction vector
    dir_full = torch.cat([dir_v, dir_L], dim=-1)
    dir_norms = torch.norm(dir_full, dim=-1, keepdim=True)
    dir_norms = torch.clamp(dir_norms, min=1e-8)
    dir_full = dir_full / dir_norms
    
    # Split back
    dir_v = dir_full[..., :k]
    dir_L = dir_full[..., k:]
    
    return dir_v, dir_L


def directional_derivative_loss(model, v, L, target_probs,
                                          target_grad_v, target_grad_L,
                                          num_directions, lambda_grad,
                                          epsilon=1e-5, gumbel=False):
    """
    Compute gradient matching loss using Cholesky parameterization.
    
    Directions are generated on-the-fly. Invalid perturbations are filtered out.
    
    Args:
        model: The emulator model
        v: Centered utilities, shape (batch, K)
        L: Cholesky factor of Omega, shape (batch, K-1, K-1)
        target_probs: Target choice probabilities, shape (batch, K)
        target_grad_v: Jacobian of log P w.r.t. v, shape (batch, K, K)
        target_grad_L: Jacobian of log P w.r.t. vech(L), shape (batch, K, num_L_params)
        num_directions: Number of random directions to sample
        lambda_grad: Weight for gradient matching loss
        epsilon: Step size for finite differences
        gumbel: Whether using Gumbel errors (affects validity checking)
        
    Returns:
        total_loss: Combined CE + gradient loss
        ce_loss: Cross-entropy loss (detached)
        grad_loss: Gradient matching loss (detached)
    """
    device = v.device
    dtype = v.dtype
    batch_size, k = v.shape
    km1 = k - 1
    num_L_params = km1 * (km1 + 1) // 2
    tril_idx = torch.tril_indices(km1, km1, device=device)
    
    was_training = model.training
    
    # Compute canonical form for base point
    Omega = L @ L.transpose(-1, -2)
    Sigma = torch.zeros(batch_size, k, k, device=device, dtype=dtype)
    Sigma[:, 0, 0] = 1.0
    Sigma[:, 1:, 1:] = Omega
    v_star, cov_star, c_star = canonicalize(v, Sigma)
    
    # CE loss in current mode
    logits_base = model(v_star, cov_star, c_star)
    log_probs_base = F.log_softmax(logits_base, dim=-1)
    
    target_clip = torch.clamp(target_probs, min=1e-8)
    ce_loss = (target_probs * (torch.log(target_clip) - log_probs_base)).mean()
    
    # Gradient loss in eval mode
    model.eval()
    
    # Generate directions on-the-fly
    dir_v, dir_L_flat = generate_directions(batch_size, k, num_directions, device, dtype)
    
    # Build dir_L as lower triangular matrices
    dir_L = torch.zeros(batch_size, num_directions, km1, km1, device=device, dtype=dtype)
    dir_L[:, :, tril_idx[0], tril_idx[1]] = dir_L_flat
    
    # Compute perturbed L matrices
    L_expanded = L.unsqueeze(1)  # (batch, 1, km1, km1)
    L_plus = L_expanded + epsilon * dir_L  # (batch, num_directions, km1, km1)
    L_minus = L_expanded - epsilon * dir_L
    
    # Compute perturbed v
    v_expanded = v.unsqueeze(1)  # (batch, 1, k)
    v_plus = v_expanded + epsilon * dir_v  # (batch, num_directions, k)
    v_minus = v_expanded - epsilon * dir_v
    
    # Check validity of perturbed points
    # Reshape for batch processing
    L_plus_flat = L_plus.reshape(-1, km1, km1)
    L_minus_flat = L_minus.reshape(-1, km1, km1)
    
    _, valid_plus = _compute_sqrt_L(L_plus_flat, gumbel=gumbel)
    _, valid_minus = _compute_sqrt_L(L_minus_flat, gumbel=gumbel)
    
    valid_plus = valid_plus.reshape(batch_size, num_directions)
    valid_minus = valid_minus.reshape(batch_size, num_directions)
    valid_both = valid_plus & valid_minus  # (batch, num_directions)
    
    # Compute canonical forms for valid perturbations
    # Build Omega and Sigma for perturbed points
    Omega_plus = L_plus @ L_plus.transpose(-1, -2)
    Omega_minus = L_minus @ L_minus.transpose(-1, -2)
    
    Sigma_plus = torch.zeros(batch_size, num_directions, k, k, device=device, dtype=dtype)
    Sigma_plus[:, :, 0, 0] = 1.0
    Sigma_plus[:, :, 1:, 1:] = Omega_plus
    
    Sigma_minus = torch.zeros(batch_size, num_directions, k, k, device=device, dtype=dtype)
    Sigma_minus[:, :, 0, 0] = 1.0
    Sigma_minus[:, :, 1:, 1:] = Omega_minus
    
    # Flatten for canonicalization
    v_plus_flat = v_plus.reshape(-1, k)
    v_minus_flat = v_minus.reshape(-1, k)
    Sigma_plus_flat = Sigma_plus.reshape(-1, k, k)
    Sigma_minus_flat = Sigma_minus.reshape(-1, k, k)
    
    v_star_plus, cov_star_plus, c_star_plus = canonicalize(v_plus_flat, Sigma_plus_flat)
    v_star_minus, cov_star_minus, c_star_minus = canonicalize(v_minus_flat, Sigma_minus_flat)
    
    # Forward pass through emulator
    with torch.no_grad():
        logits_plus = model(v_star_plus, cov_star_plus, c_star_plus)
        logits_minus = model(v_star_minus, cov_star_minus, c_star_minus)
        
        log_probs_plus = F.log_softmax(logits_plus, dim=-1)
        log_probs_minus = F.log_softmax(logits_minus, dim=-1)
    
    # Reshape back
    log_probs_plus = log_probs_plus.reshape(batch_size, num_directions, k)
    log_probs_minus = log_probs_minus.reshape(batch_size, num_directions, k)
    
    # Central difference estimate
    model_dir_deriv = (log_probs_plus - log_probs_minus) / (2 * epsilon)
    
    # Target directional derivatives
    # target_grad_v: (batch, K, K) - Jacobian[j, i] = d log P_j / d v_i
    # target_grad_L: (batch, K, num_L_params)
    # dir_v: (batch, num_directions, K)
    # dir_L_flat: (batch, num_directions, num_L_params)
    
    target_dir_deriv_v = torch.einsum('bdk,bjk->bdj', dir_v, target_grad_v)
    target_dir_deriv_L = torch.einsum('bdl,bjl->bdj', dir_L_flat, target_grad_L)
    target_dir_deriv = target_dir_deriv_v + target_dir_deriv_L  # (batch, num_directions, K)
    
    # Initial mask from geometry validity
    valid_mask = valid_both.unsqueeze(-1).float()  # (batch, num_directions, 1)
    
    # Clamp derivatives for stability
    model_dir_deriv = torch.clamp(model_dir_deriv, min=-1e4, max=1e4)
    target_dir_deriv = torch.clamp(target_dir_deriv, min=-1e4, max=1e4)
    
    # Compute weighted squared differences
    weights = target_probs.unsqueeze(1) * valid_mask  # (batch, num_directions, K)
    diff_sq = (model_dir_deriv - target_dir_deriv) ** 2
    weighted_diff_sq = weights * diff_sq
    
    # Check for finite values - exclude directions with any non-finite weighted diff
    finite_per_direction = torch.isfinite(weighted_diff_sq).all(dim=-1)  # (batch, num_directions)
    valid_final = valid_both & finite_per_direction
    
    # Use torch.where to properly zero out invalid entries (NaN * 0 = NaN with multiplication)
    valid_mask_final = valid_final.unsqueeze(-1).expand_as(weighted_diff_sq)
    weighted_diff_sq = torch.where(
        valid_mask_final,
        weighted_diff_sq,
        torch.zeros_like(weighted_diff_sq)
    )
    
    # Compute per-direction loss: sum over k gives probability-weighted squared error per direction
    # Shape: (batch, num_directions)
    per_direction_loss = weighted_diff_sq.sum(dim=-1)
    
    # Count valid directions per example
    num_valid_per_example = valid_final.float().sum(dim=-1)  # (batch,)
    
    # Identify examples with at least one valid direction
    has_valid = num_valid_per_example > 0  # (batch,)
    
    if has_valid.any():
        # Mean loss per example (average over valid directions for that example)
        # per_direction_loss is already zeroed for invalid directions
        per_example_loss = per_direction_loss.sum(dim=-1) / num_valid_per_example.clamp(min=1)
        
        # Average over examples that have valid directions
        # per_example_loss is 0 for invalid examples, so sum over all is sum over valid
        num_valid_examples = has_valid.float().sum()
        grad_loss = (per_example_loss / num_valid_examples).sum()
    else:
        grad_loss = torch.tensor(0.0, device=device, dtype=dtype)
    
    # Restore training mode
    if was_training:
        model.train()
    
    total_loss = ce_loss + lambda_grad * grad_loss
    
    # Final safety check
    if not torch.isfinite(total_loss):
        total_loss = (logits_base * 0).sum()
        return total_loss, 0.0, 0.0
    
    return total_loss, ce_loss.detach().item(), grad_loss.detach().item()


def _add_to_buffers_multi_k(buffers_dict, batch_size, k_values, num_samples,
                                      compute_gradients=False, tau=5.0, gumbel=False):
    """Generate data and add to Cholesky replay buffers for multiple K values."""
    for k in k_values:
        train_buffer, eval_buffer = buffers_dict[k]
        eval_add_size = max(1, batch_size // 10)
        
        for s, buf in zip((batch_size, eval_add_size), (train_buffer, eval_buffer)):
            v, L, y, grad_v, grad_L = simulate_data(
                s, k, num_samples, 
                compute_gradients=compute_gradients, tau=tau, gumbel=gumbel)
            
            buf.add(v, L, y, grad_v, grad_L)


def multinomial_cross_entropy_loss(logits, target, eps=1e-8, log=False):
    """Numerically stable multinomial cross-entropy loss."""
    log_probs = F.log_softmax(logits, dim=-1)
    
    if log:
        loss = torch.exp(target) * (target - log_probs)
    else:
        target_probs_clip = torch.clamp(target, min=eps)
        loss = target * (torch.log(target_probs_clip) - log_probs)
    
    return loss.mean()


def get_model_config(model):
    """Extract configuration from a model instance."""
    
    if isinstance(model, Emulator):
        diag_phi = model.per_alt_encoder.diag_phi
        diag_rho = model.per_alt_encoder.diag_rho
        offdiag_phi = model.per_alt_encoder.offdiag_phi
        offdiag_rho = model.per_alt_encoder.offdiag_rho
        
        config = {
            'name': model.name,
            'include_c_star': model.include_c_star,
            'diag_phi_hidden_dim': diag_phi.network[0].out_features if diag_phi.num_layers > 0 else diag_phi.input_dim,
            'diag_phi_output_dim': model.per_alt_encoder.diag_phi_output_dim,
            'diag_phi_num_layers': diag_phi.num_layers,
            'diag_rho_hidden_dim': diag_rho.network[0].out_features if diag_rho.num_layers > 0 else diag_rho.input_dim,
            'diag_rho_output_dim': model.per_alt_encoder.diag_rho_output_dim,
            'diag_rho_num_layers': diag_rho.num_layers,
            'offdiag_phi_hidden_dim': offdiag_phi.network[0].out_features if offdiag_phi.num_layers > 0 else offdiag_phi.input_dim,
            'offdiag_phi_output_dim': model.per_alt_encoder.offdiag_phi_output_dim,
            'offdiag_phi_num_layers': offdiag_phi.num_layers,
            'offdiag_rho_hidden_dim': offdiag_rho.network[0].out_features if offdiag_rho.num_layers > 0 else offdiag_rho.input_dim,
            'offdiag_rho_output_dim': model.per_alt_encoder.offdiag_rho_output_dim,
            'offdiag_rho_num_layers': offdiag_rho.num_layers,
            'combine_hidden_dim': model.per_alt_encoder.combine_mlp.hidden_dim,
            'combine_output_dim': model.per_alt_encoder.combine_output_dim,
            'combine_num_layers': model.per_alt_encoder.combine_mlp.num_layers,
            'num_equiv_layers': len(model.equiv_layers),
            'dropout_rate': model.dropout_rate,
            'include_k': model.include_k,
        }
        
        if len(model.equiv_layers) > 0:
            config['equiv_hidden_dim'] = model.equiv_layers[0].equiv_layer.A.weight.shape[0]
        else:
            config['equiv_hidden_dim'] = model.feature_dim
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
    return config


def save_checkpoint(model, optimizer, epoch, loss, k_str, checkpoint_dir="model-checkpoints"):
    """Save a model checkpoint including full configuration.
    
    Args:
        model: The model to save
        optimizer: The optimizer
        epoch: Current epoch
        loss: Current loss
        k_str: String identifier for K value(s) (e.g., "3" or "3_4_5")
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config = get_model_config(model)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_class': model.__class__.__name__,
        'config': config,
    }
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_path = os.path.join(checkpoint_dir, f"checkpoint_{model.name}_{k_str}_{date_str}.pt")
    torch.save(checkpoint, date_path)
    k_path = os.path.join(checkpoint_dir, f"checkpoint_{model.name}_{k_str}.pt")
    torch.save(checkpoint, k_path)


def load_checkpoint(checkpoint_path, device=None, optimizer=None):
    """Load a model checkpoint, reconstructing the model from saved configuration."""
    
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    model_class = checkpoint.get('model_class', 'Emulator')
    
    if model_class == 'Emulator':
        name = config.pop('name', 'emulator')
        model_kwargs = {k_: v for k_, v in config.items() if v is not None}
        model = Emulator(name=name, **model_kwargs)
    else:
        raise ValueError(f"Unknown model class: {model_class}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, epoch, loss

class DiskBackedBufferManagerCholesky:
    """Manages in-memory replay buffers backed by larger on-disk buffers for Cholesky parameterization.
    
    Periodically refreshes in-memory buffers by loading random chunks from disk.
    Uses background prefetching to overlap disk I/O with GPU computation.
    """
    
    def __init__(
        self,
        data_dir: str,
        k_values: List[int],
        in_memory_size: int,
        device: torch.device,
        use_sobolev: bool = True,
        gumbel: bool = False,
        num_workers: int = 4,
    ):
        """Initialize the buffer manager.
        
        Args:
            data_dir: Directory containing the pre-generated buffer files
            k_values: List of K values to load
            in_memory_size: Size of in-memory buffers (chunk size)
            device: PyTorch device for in-memory data
            use_sobolev: Whether buffers contain gradient data
            gumbel: Whether using Gumbel errors (affects file naming)
            num_workers: Number of worker threads for parallel disk I/O
        """
        self.data_dir = data_dir
        self.k_values = sorted(k_values)
        self.in_memory_size = in_memory_size
        self.device = device
        self.use_sobolev = use_sobolev
        self.gumbel = gumbel
        self.grad_suffix = "_cholesky" + ("_gumbel" if gumbel else "") if use_sobolev else "_cholesky"
        self.num_workers = num_workers
        
        # Validate that all buffer files exist and get their sizes
        self.disk_sizes = self._validate_and_get_sizes()
        
        # Create in-memory GPU buffers
        self.train_buffers: Dict[int, ReplayBuffer] = {}
        self.eval_buffers: Dict[int, ReplayBuffer] = {}
        
        for k in self.k_values:
            self.train_buffers[k] = ReplayBuffer(
                in_memory_size, k, device, store_gradients=use_sobolev
            )
            self.eval_buffers[k] = ReplayBuffer(
                in_memory_size, k, device, store_gradients=use_sobolev
            )
        
        # Threading infrastructure for async prefetch
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=num_workers)
        self._prefetch_futures: List[Future] = []
        self._closed = False
        
        # Initial synchronous load
        print(f"Loading initial chunks into memory (size={in_memory_size:,} per K)...")
        self._load_all_sync()
        print("Initial load complete.")
        
        # Start first background prefetch
        print(f"Starting background prefetch with {num_workers} workers...")
        self._start_prefetch()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def close(self):
        if self._closed:
            return
        
        self._closed = True
        
        for future in self._prefetch_futures:
            try:
                future.result(timeout=60)
            except Exception:
                pass
        
        self._prefetch_futures = []
        
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def _get_paths(self, k: int) -> Tuple[str, str]:
        train_path = os.path.join(self.data_dir, f"{k}-train{self.grad_suffix}.pt")
        eval_path = os.path.join(self.data_dir, f"{k}-eval{self.grad_suffix}.pt")
        return train_path, eval_path
    
    def _validate_and_get_sizes(self) -> Dict[int, Dict[str, int]]:
        sizes = {}
        print(f"Validating Cholesky buffer files in {self.data_dir}...")
        
        for k in self.k_values:
            train_path, eval_path = self._get_paths(k)
            
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Missing train buffer: {train_path}")
            if not os.path.exists(eval_path):
                raise FileNotFoundError(f"Missing eval buffer: {eval_path}")
            
            train_data = torch.load(train_path, map_location='cpu', weights_only=True)
            eval_data = torch.load(eval_path, map_location='cpu', weights_only=True)
            
            sizes[k] = {
                'train': train_data['size'],
                'eval': eval_data['size'],
            }
            
            del train_data, eval_data
        
        total_train = sum(s['train'] for s in sizes.values())
        total_eval = sum(s['eval'] for s in sizes.values())
        print(f"Found {len(self.k_values)} K values, "
              f"total examples: {total_train:,} train, {total_eval:,} eval")
        
        return sizes
    
    def _load_random_chunk_to_device(self, k: int, split: str) -> Tuple[torch.Tensor, ...]:
        train_path, eval_path = self._get_paths(k)
        path = train_path if split == 'train' else eval_path
        
        data = torch.load(path, map_location='cpu', weights_only=True)
        
        disk_size = data['size']
        chunk_size = min(self.in_memory_size, disk_size)
        
        indices = torch.randperm(disk_size)[:chunk_size]
        
        v = data['v'][indices].to(self.device)
        L = data['L'][indices].to(self.device)
        y = data['y'][indices].to(self.device)
        
        if self.use_sobolev:
            grad_v = data['grad_v'][indices].to(self.device)
            grad_L = data['grad_L'][indices].to(self.device)
        else:
            grad_v = None
            grad_L = None
        
        del data
        
        return v, L, y, grad_v, grad_L
    
    def _load_chunk_to_cpu_pinned(
        self, k: int, split: str
    ) -> Tuple[int, str, Tuple[torch.Tensor, ...]]:
        train_path, eval_path = self._get_paths(k)
        path = train_path if split == 'train' else eval_path
        
        data = torch.load(path, map_location='cpu', weights_only=True)
        
        disk_size = data['size']
        chunk_size = min(self.in_memory_size, disk_size)
        
        indices = torch.randperm(disk_size)[:chunk_size]
        
        v = data['v'][indices]
        L = data['L'][indices]
        y = data['y'][indices]

        if torch.cuda.is_available():
            v = v.pin_memory()
            L = L.pin_memory()
            y = y.pin_memory()
        
        if self.use_sobolev:
            grad_v = data['grad_v'][indices]
            grad_L = data['grad_L'][indices]
            
            if torch.cuda.is_available():
                grad_v = grad_v.pin_memory()
                grad_L = grad_L.pin_memory()
        else:
            grad_v = None
            grad_L = None
        
        del data
        
        return k, split, (v, L, y, grad_v, grad_L)
    
    def _load_all_sync(self):
        for k in self.k_values:
            for split in ['train', 'eval']:
                v, L, y, grad_v, grad_L = self._load_random_chunk_to_device(k, split)
                
                buffer = self.train_buffers[k] if split == 'train' else self.eval_buffers[k]
                buffer.size = 0
                buffer.idx = 0
                buffer.add(v, L, y, grad_v, grad_L)
    
    def _start_prefetch(self):
        if self._closed or self._executor is None:
            return
        
        self._prefetch_futures = []
        for k in self.k_values:
            for split in ['train', 'eval']:
                future = self._executor.submit(self._load_chunk_to_cpu_pinned, k, split)
                self._prefetch_futures.append(future)
    
    def _wait_for_prefetch(self) -> Dict[int, Dict[str, Tuple[torch.Tensor, ...]]]:
        staging: Dict[int, Dict[str, Tuple[torch.Tensor, ...]]] = {}
        
        for future in as_completed(self._prefetch_futures):
            k, split, tensors = future.result()
            if k not in staging:
                staging[k] = {}
            staging[k][split] = tensors
        
        self._prefetch_futures = []
        return staging
    
    def _transfer_staging_to_gpu(
        self, 
        staging: Dict[int, Dict[str, Tuple[torch.Tensor, ...]]], 
        splits: List[str]
    ):
        for k in self.k_values:
            for split in splits:
                if k not in staging or split not in staging[k]:
                    continue
                
                v, L, y, grad_v, grad_L = staging[k][split]
                
                buffer = self.train_buffers[k] if split == 'train' else self.eval_buffers[k]
                
                buffer.size = 0
                buffer.idx = 0
                
                v_gpu = v.to(self.device, non_blocking=True)
                L_gpu = L.to(self.device, non_blocking=True)
                y_gpu = y.to(self.device, non_blocking=True)
                
                if self.use_sobolev and grad_v is not None and grad_L is not None:
                    grad_v_gpu = grad_v.to(self.device, non_blocking=True)
                    grad_L_gpu = grad_L.to(self.device, non_blocking=True)
                else:
                    grad_v_gpu = None
                    grad_L_gpu = None
                
                buffer.add(v_gpu, L_gpu, y_gpu, grad_v_gpu, grad_L_gpu)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
    
    def refresh_all(self, splits: Optional[List[str]] = None):
        if splits is None:
            splits = ['train', 'eval']
        
        staging = self._wait_for_prefetch()
        self._transfer_staging_to_gpu(staging, splits)
        del staging
        self._start_prefetch()
    
    def sample(self, k: int, batch_size: int, split: str = 'train') -> Tuple[torch.Tensor, ...]:
        buffer = self.train_buffers[k] if split == 'train' else self.eval_buffers[k]
        return buffer.sample(batch_size)
    
    def __repr__(self) -> str:
        return (f"DiskBackedBufferManagerCholesky(data_dir='{self.data_dir}', "
                f"k_values={self.k_values}, in_memory_size={self.in_memory_size:,}, "
                f"use_sobolev={self.use_sobolev}, gumbel={self.gumbel}, num_workers={self.num_workers})")


def train_model(model, ks, optimizer, num_epochs, batch_size, num_samples, eval_every,
                eval_batch_size, plot=False, scheduler=None,
                buffer_size=100000, buffer_add_per_step=None, buffer_add_every=None,
                lambda_grad=None, tau=5.0, num_directions=1, epsilon=1e-5):
    """Train the emulator model on one or more K values.
    
    Replay buffers are loaded from "simulation-data-checkpoints/" if present;
    otherwise they are simulated and saved there.
    
    Args:
        model: The emulator model to train
        ks: Integer or list/tuple of K values to train on
        optimizer: PyTorch optimizer (SGD-like, e.g. AdamW; LBFGS not supported)
        num_epochs: Number of training epochs/steps
        batch_size: Training batch size (per K value)
        num_samples: Number of Monte Carlo samples for simulating choice probabilities
        eval_every: Evaluate every N epochs
        eval_batch_size: Batch size for evaluation (per K value)
        plot: Whether to show live plots
        scheduler: Optional learning rate scheduler
        buffer_size: Size of replay buffer (per K value)
        buffer_add_per_step: Number of examples to add to buffer per step (per K value)
        buffer_add_every: Add to buffer every N steps
        lambda_grad: Weight for gradient matching loss (enables Sobolev-style training)
        tau: Temperature for soft gradient computation in data generation
        num_directions: Number of random directions for directional derivative loss
        epsilon: Finite difference step size for directional derivative loss
    
    Returns:
        (model, df) tuple with trained model and training history DataFrame
    """
    device = get_device()
    model.to(device)
    
    if isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError("LBFGS is not supported by train_model. "
                         "Use an SGD-like optimizer instead.")
    
    # Normalize ks to a sorted list
    if isinstance(ks, int):
        k_values = [ks]
    else:
        k_values = sorted(list(ks))
    
    use_sobolev = lambda_grad is not None
    
    # Create string identifier for K values
    k_str = "_".join(str(k) for k in k_values)
    
    train_loss_history = []
    eval_ce_loss_history = []
    eval_rmse_loss_history = []
    dropout_history = []
    grad_norm_history = []
    eval_ce_loss_train_history = []
    eval_rmse_loss_train_history = []

    if use_sobolev:
        train_ce_loss_history = []
        train_grad_loss_history = []
        eval_total_loss_history = []
        eval_grad_loss_history = []
        eval_total_loss_train_history = []
    
    mse_loss = nn.MSELoss()
    date_str = datetime.now().strftime("%Y-%m-%d")
    df_filename = os.path.join("fitting-results", f"train_{model.name}_{k_str}.csv")
    df_filename_date = os.path.join("fitting-results", f"train_{model.name}_{k_str}_{date_str}.csv")
    batch_size_curr = batch_size
    
    data_checkpoint_dir = "simulation-data-checkpoints"
    os.makedirs(data_checkpoint_dir, exist_ok=True)
    checkpoint_dir = "model-checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("fitting-results", exist_ok=True)
    
    update_buffers = buffer_add_every is not None and buffer_add_per_step is not None
    
    # Load or create replay buffers for each K
    # Determine if using Gumbel (non-factorization-invariant) from model
    gumbel = model.include_c_star
    grad_suffix = "_cholesky" + ("_gumbel" if gumbel else "") if use_sobolev else ""
    buffers_dict = {}  # K -> (train_buffer, eval_buffer)
    
    for k in k_values:
        train_buffer_path = os.path.join(data_checkpoint_dir, f"{k}-train{grad_suffix}.pt")
        eval_buffer_path = os.path.join(data_checkpoint_dir, f"{k}-eval{grad_suffix}.pt")
        
        if os.path.exists(train_buffer_path) and os.path.exists(eval_buffer_path):
            print(f"Loading replay buffers for K={k} from {data_checkpoint_dir}/...")
            
            train_data = torch.load(train_buffer_path, map_location=device, weights_only=True)
            eval_data = torch.load(eval_buffer_path, map_location=device, weights_only=True)
            
            if use_sobolev:
                # Load Cholesky buffers
                train_replay_buffer = ReplayBuffer(buffer_size, k, device, store_gradients=True)
                train_replay_buffer.v = train_data['v'].to(device)
                train_replay_buffer.L = train_data['L'].to(device)
                train_replay_buffer.y = train_data['y'].to(device)
                train_replay_buffer.size = train_data['size']
                train_replay_buffer.idx = train_data['idx']
                train_replay_buffer.grad_v = train_data['grad_v'].to(device)
                train_replay_buffer.grad_L = train_data['grad_L'].to(device)
                
                eval_replay_buffer = ReplayBuffer(buffer_size, k, device, store_gradients=True)
                eval_replay_buffer.v = eval_data['v'].to(device)
                eval_replay_buffer.L = eval_data['L'].to(device)
                eval_replay_buffer.y = eval_data['y'].to(device)
                eval_replay_buffer.size = eval_data['size']
                eval_replay_buffer.idx = eval_data['idx']
                eval_replay_buffer.grad_v = eval_data['grad_v'].to(device)
                eval_replay_buffer.grad_L = eval_data['grad_L'].to(device)
            else:
                # Load standard buffers (no gradients)
                train_replay_buffer = ReplayBuffer(buffer_size, k, device, store_gradients=False)
                train_replay_buffer.v = train_data['v'].to(device)
                train_replay_buffer.L = train_data['L'].to(device)
                train_replay_buffer.y = train_data['y'].to(device)
                train_replay_buffer.size = train_data['size']
                train_replay_buffer.idx = train_data['idx']
                
                eval_replay_buffer = ReplayBuffer(buffer_size, k, device, store_gradients=False)
                eval_replay_buffer.v = eval_data['v'].to(device)
                eval_replay_buffer.L = eval_data['L'].to(device)
                eval_replay_buffer.y = eval_data['y'].to(device)
                eval_replay_buffer.size = eval_data['size']
                eval_replay_buffer.idx = eval_data['idx']
            
            print(f"  Loaded train buffer with {len(train_replay_buffer)} examples")
            print(f"  Loaded eval buffer with {len(eval_replay_buffer)} examples")
            
            buffers_dict[k] = (train_replay_buffer, eval_replay_buffer)
        else:
            print(f"Creating replay buffers for K={k}...")
            train_replay_buffer = ReplayBuffer(buffer_size, k, device, store_gradients=use_sobolev)
            eval_replay_buffer = ReplayBuffer(buffer_size, k, device, store_gradients=use_sobolev)
            buffers_dict[k] = (train_replay_buffer, eval_replay_buffer)
    
    # Fill any empty buffers
    empty_k_values = [k for k in k_values if len(buffers_dict[k][0]) == 0]
    if empty_k_values:
        print(f"Pre-filling replay buffers for K={empty_k_values}...")
        
        # Determine batch size for filling
        if buffer_add_per_step is not None:
            fill_batch_size = buffer_add_per_step
        else:
            # Default batch size when buffer_add_per_step is None
            fill_batch_size = 100
        
        initial_fill = buffer_size
        num_batches = initial_fill // fill_batch_size
        
        # Temporarily create a buffers_dict with only empty buffers for filling
        empty_buffers_dict = {k: buffers_dict[k] for k in empty_k_values}
        
        for _ in trange(num_batches, desc="Filling buffers"):
            _add_to_buffers_multi_k(
                empty_buffers_dict, fill_batch_size, empty_k_values, num_samples,
                compute_gradients=use_sobolev, tau=tau, gumbel=gumbel
            )        
        # Save newly created buffers
        for k in empty_k_values:
            train_replay_buffer, eval_replay_buffer = buffers_dict[k]
            print(f"  K={k}: Train buffer has {len(train_replay_buffer)} examples, "
                  f"Eval buffer has {len(eval_replay_buffer)} examples")
            
            train_buffer_path = os.path.join(data_checkpoint_dir, f"{k}-train{grad_suffix}.pt")
            eval_buffer_path = os.path.join(data_checkpoint_dir, f"{k}-eval{grad_suffix}.pt")
            
            train_save_dict = {
                'v': train_replay_buffer.v,
                'L': train_replay_buffer.L,
                'y': train_replay_buffer.y,
                'size': train_replay_buffer.size,
                'idx': train_replay_buffer.idx,
            }
            if use_sobolev:
                train_save_dict['grad_v'] = train_replay_buffer.grad_v
                train_save_dict['grad_L'] = train_replay_buffer.grad_L
            torch.save(train_save_dict, train_buffer_path)
            
            eval_save_dict = {
                'v': eval_replay_buffer.v,
                'L': eval_replay_buffer.L,
                'y': eval_replay_buffer.y,
                'size': eval_replay_buffer.size,
                'idx': eval_replay_buffer.idx,
            }
            if use_sobolev:
                eval_save_dict['grad_v'] = eval_replay_buffer.grad_v
                eval_save_dict['grad_L'] = eval_replay_buffer.grad_L
            torch.save(eval_save_dict, eval_buffer_path)
        
        print("Replay buffers saved.")
    
    # No direction replay buffers needed - directions are generated on-the-fly
    best_eval_loss = float('inf')
    eval_forward_times = []
    
    def get_df():
        df_dict = {
            "train_loss": train_loss_history,
            "eval_ce_loss": eval_ce_loss_history,
            "eval_rmse_loss": eval_rmse_loss_history,
            "eval_ce_loss_train": eval_ce_loss_train_history,
            "eval_rmse_loss_train": eval_rmse_loss_train_history,
            "dropout": dropout_history
        }
        if use_sobolev:
            df_dict["train_ce_loss"] = train_ce_loss_history
            df_dict["train_grad_loss"] = train_grad_loss_history
            df_dict["eval_total_loss"] = eval_total_loss_history
            df_dict["eval_grad_loss"] = eval_grad_loss_history
            df_dict["eval_total_loss_train"] = eval_total_loss_train_history
        return pd.DataFrame(df_dict)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax0 = ax[0]
        ax1 = ax[1]
        ax0_twin = ax0.twinx()
        plt.ion()
        
        # Colors for different K values
        k_colors = {k: f'C{i}' for i, k in enumerate(k_values)}
    
    # Print optimizer type
    optimizer_name = type(optimizer).__name__
    print(f"Training with {optimizer_name} optimizer")
    
    for i in trange(num_epochs):
        model.train()
        
        # Update buffers for all K values
        if update_buffers and i % buffer_add_every == 0:
            _add_to_buffers_multi_k(
                buffers_dict, buffer_add_per_step, k_values, num_samples,
                compute_gradients=use_sobolev, tau=tau, gumbel=gumbel
            )
        
        # Training step: sample from buffer
        optimizer.zero_grad()
        
        # Accumulate losses across K values
        total_loss = 0.0
        total_ce_loss = 0.0
        total_grad_loss = 0.0
        
        for k in k_values:
            train_buffer, _ = buffers_dict[k]
            v, L, y, grad_v, grad_L = train_buffer.sample(batch_size_curr)
            
            if use_sobolev:
                target_probs = y
                
                loss, ce_loss, grad_loss = directional_derivative_loss(
                    model, v, L, target_probs, grad_v, grad_L,
                    num_directions=num_directions,
                    lambda_grad=lambda_grad,
                    epsilon=epsilon,
                    gumbel=gumbel
                )
                
                total_ce_loss += ce_loss
                total_grad_loss += grad_loss
            else:
                # Reconstruct canonical form for model forward pass
                Omega = L @ L.transpose(-1, -2)
                Sigma = omega_to_cov(Omega)
                v_star, cov_star, c_star = canonicalize(v, Sigma)
                
                logits = model(v_star, cov_star, c_star)
                loss = multinomial_cross_entropy_loss(
                    logits, y, eps=1. / num_samples, log=False)
            
            total_loss += loss
        
        # Average loss across K values
        avg_loss_tensor = total_loss / len(k_values)
        avg_loss_tensor.backward()
        
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        grad_norm_history.append(total_norm.item())
        optimizer.step()
        
        avg_loss = avg_loss_tensor.detach().cpu().item()
        
        if use_sobolev:
            train_ce_loss_history.append(total_ce_loss / len(k_values))
            train_grad_loss_history.append(total_grad_loss / len(k_values))
    
        # Model step (if applicable)
        if hasattr(model, 'step'):
            model.step()
        
        # Scheduler step (if provided)
        if scheduler is not None:
            scheduler.step()
        
        train_loss_history.append(avg_loss)
        
        # Record dropout
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                dropout_history.append(module.p)
                break
        else:
            dropout_history.append(0.0)

        # Evaluation
        if (i + 1) % eval_every == 0:
            
            # Sample data once for both eval and train mode metrics
            eval_samples = {}  # K -> (v, L, y, grad_v, grad_L)
            
            for k in k_values:
                _, eval_buffer = buffers_dict[k]
                v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval = eval_buffer.sample(eval_batch_size)
                eval_samples[k] = (v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval)
            
            # === Eval mode metrics ===
            model.eval()
            
            eval_total_loss_sum = 0.0
            eval_ce_loss_sum = 0.0
            eval_grad_loss_sum = 0.0
            eval_rmse_sum = 0.0
            
            # For plotting: collect predictions and targets per K
            plot_data = {}  # K -> (yhat, y_probs)
            
            for k in k_values:
                v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval = eval_samples[k]
                
                # Reconstruct canonical form for model forward pass
                Omega_eval = L_eval @ L_eval.transpose(-1, -2)
                Sigma_eval = omega_to_cov(Omega_eval)
                v_star_eval, cov_star_eval, c_star_eval = canonicalize(v_eval, Sigma_eval)
                
                with torch.no_grad():
                    logits_eval = model(v_star_eval, cov_star_eval, c_star_eval)
                
                    if use_sobolev:
                        target_probs_eval = y_eval
                        
                        eval_total, eval_ce, eval_grad = directional_derivative_loss(
                            model, v_eval, L_eval, target_probs_eval,
                            grad_v_eval, grad_L_eval,
                            num_directions=num_directions,
                            lambda_grad=lambda_grad,
                            epsilon=epsilon,
                            gumbel=gumbel
                        )
                        
                        eval_total_loss_sum += eval_total.item()
                        eval_ce_loss_sum += eval_ce
                        eval_grad_loss_sum += eval_grad
                    else:
                        eval_ce = multinomial_cross_entropy_loss(
                            logits_eval, y_eval, eps=1. / num_samples, log=False)
                        eval_ce_loss_sum += eval_ce.item()
                    
                    yhat_eval = F.softmax(logits_eval, dim=-1)
                    y_eval_probs = y_eval
                    rmse = torch.sqrt(mse_loss(y_eval_probs, yhat_eval)).item()
                    eval_rmse_sum += rmse
                    
                    plot_data[k] = (yhat_eval.cpu(), y_eval_probs.cpu())
            
            # Average across K values (eval mode)
            num_k = len(k_values)
            avg_eval_ce = eval_ce_loss_sum / num_k
            avg_eval_rmse = eval_rmse_sum / num_k
            
            eval_ce_loss_history.append(avg_eval_ce)
            eval_rmse_loss_history.append(avg_eval_rmse)
            
            if use_sobolev:
                avg_eval_total = eval_total_loss_sum / num_k
                avg_eval_grad = eval_grad_loss_sum / num_k
                eval_total_loss_history.append(avg_eval_total)
                eval_grad_loss_history.append(avg_eval_grad)
                eval_comparison_loss = avg_eval_total
            else:
                eval_comparison_loss = avg_eval_ce
            
            # Train mode metrics (same samples)
            model.train()
            
            eval_ce_loss_train_sum = 0.0
            eval_rmse_train_sum = 0.0
            eval_total_loss_train_sum = 0.0
            
            for k in k_values:
                v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval = eval_samples[k]
                
                # Reconstruct canonical form for model forward pass
                Omega_eval = L_eval @ L_eval.transpose(-1, -2)
                Sigma_eval = omega_to_cov(Omega_eval)
                v_star_eval, cov_star_eval, c_star_eval = canonicalize(v_eval, Sigma_eval)
                
                with torch.no_grad():
                    logits_eval_train = model(v_star_eval, cov_star_eval, c_star_eval)
                    
                    if use_sobolev:
                        target_probs_eval = y_eval
                        
                        eval_total_train, eval_ce_train, _ = directional_derivative_loss(
                            model, v_eval, L_eval, target_probs_eval,
                            grad_v_eval, grad_L_eval,
                            num_directions=num_directions,
                            lambda_grad=lambda_grad,
                            epsilon=epsilon,
                            gumbel=gumbel
                        )
                        
                        eval_total_loss_train_sum += eval_total_train.item()
                        eval_ce_loss_train_sum += eval_ce_train
                    else:
                        eval_ce_train = multinomial_cross_entropy_loss(
                            logits_eval_train, y_eval, eps=1. / num_samples, log=False)
                        eval_ce_loss_train_sum += eval_ce_train.item()
                    
                    yhat_eval_train = F.softmax(logits_eval_train, dim=-1)
                    y_eval_probs = y_eval
                    rmse_train = torch.sqrt(mse_loss(y_eval_probs, yhat_eval_train)).item()
                    eval_rmse_train_sum += rmse_train
            
            # Average across K values (train mode)
            eval_ce_loss_train_history.append(eval_ce_loss_train_sum / num_k)
            eval_rmse_loss_train_history.append(eval_rmse_train_sum / num_k)
            
            if use_sobolev:
                eval_total_loss_train_history.append(eval_total_loss_train_sum / num_k)
            
            # CPU timing (use first K for timing)
            k_for_timing = k_values[0]
            v_timing, L_timing, _, _, _ = eval_samples[k_for_timing]
            
            # Reconstruct canonical form for timing
            Omega_timing = L_timing @ L_timing.transpose(-1, -2)
            Sigma_timing = omega_to_cov(Omega_timing)
            v_star_timing, cov_star_timing, c_star_timing = canonicalize(v_timing, Sigma_timing)

            model_cpu = copy.deepcopy(model).cpu()
            model_cpu.eval()

            with torch.no_grad():
                v_star_cpu = v_star_timing.cpu()
                cov_star_cpu = cov_star_timing.cpu()
                c_star_cpu = c_star_timing.cpu()
                _ = model_cpu(v_star_cpu, cov_star_cpu, c_star_cpu)
                forward_start = time.time()
                _ = model_cpu(v_star_cpu, cov_star_cpu, c_star_cpu)
                forward_time = time.time() - forward_start
            eval_forward_times.append(forward_time)
            del model_cpu

            df = get_df()
            df.to_csv(df_filename)
            df.to_csv(df_filename_date)
            
            if eval_comparison_loss < best_eval_loss:
                best_eval_loss = eval_comparison_loss
                save_checkpoint(model, optimizer, i + 1, avg_eval_ce, k_str, checkpoint_dir)

            if plot:
                clear_output(wait=True)
                ax0.clear()
                ax0_twin.clear()
                ax0_twin.yaxis.set_label_position('right')
                ax0_twin.yaxis.tick_right()
                
                epoch_idx = np.arange(1, i + 2)
                ax0.plot(epoch_idx, train_loss_history, label='Train Loss', color='C0', alpha=0.7)
                eval_indices = [j + 1 for j, val in enumerate(eval_ce_loss_history) if val is not None]
                eval_values = [val for val in eval_ce_loss_history if val is not None]
                ax0.plot(eval_indices, eval_values, label='Eval CE Loss', color='C2', alpha=0.7)
                ax0.plot(epoch_idx, train_loss_history[0] / epoch_idx, label='O(1/n)', color='C3', alpha=0.7)
                ax0.set_xscale('log')
                ax0.set_yscale('log')
                ax0.set_xlabel('Epoch')
                ax0.set_ylabel('Loss', color='C0')
                ax0.set_title(f'Training Loss & Grad Norm (Epoch {i + 1}, K={k_str})')
                ax0_twin.plot(epoch_idx, grad_norm_history, label='Grad Norm', color='C1', alpha=0.7)
                ax0_twin.set_yscale('log')
                ax0_twin.set_ylabel('Grad Norm', color='C1')
                lines1, labels1 = ax0.get_legend_handles_labels()
                lines2, labels2 = ax0_twin.get_legend_handles_labels()
                ax0.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                ax1.clear()
                for k in k_values:
                    yhat, y_probs = plot_data[k]
                    ax1.scatter(yhat.numpy(), y_probs.numpy(),
                               color=k_colors[k], label=f"K={k}", alpha=0.2, s=10)
                ax1.plot([0, 1], [0, 1], color="black", linestyle="--", 
                        label="Perfect", alpha=0.7)
                ax1.set_xlabel("Predicted Probability")
                ax1.set_ylabel("Simulated Probability")
                ax1.legend()
                ax1.set_title("Predicted vs. Actual (All K)")
                
                fig.tight_layout()
                display(fig)
            
            current_time_ms = forward_time * 1000
            avg_time_ms = np.mean(eval_forward_times) * 1000
            time_per_sample_us = (forward_time / eval_batch_size) * 1e6
            avg_time_per_sample_us = (np.mean(eval_forward_times) / eval_batch_size) * 1e6
            
            print(f"\nCPU Forward Pass Timing (batch_size={eval_batch_size}, K={k_for_timing}):")
            print(f"  Current:  {current_time_ms:.2f} ms total, {time_per_sample_us:.2f} µs/sample")
            print(f"  Average:  {avg_time_ms:.2f} ms total, {avg_time_per_sample_us:.2f} µs/sample")
            print(f"  Optimizer: {optimizer_name}")
            print(f"  Training on K values: {k_values}")
            for k in k_values:
                train_buf, eval_buf = buffers_dict[k]
                print(f"    K={k}: Train buffer={len(train_buf)}, Eval buffer={len(eval_buf)}")
            if use_sobolev:
                print(f"  Directional derivative training: lambda_grad={lambda_grad}, "
                      f"num_directions={num_directions}, epsilon={epsilon}")
                print(f"  Last train CE loss: {train_ce_loss_history[-1]:.6f}, "
                      f"grad loss: {train_grad_loss_history[-1]:.6f}")

            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        else:
            eval_ce_loss_history.append(None)
            eval_rmse_loss_history.append(None)
            eval_ce_loss_train_history.append(None)
            eval_rmse_loss_train_history.append(None)
            if use_sobolev:
                eval_total_loss_history.append(None)
                eval_grad_loss_history.append(None)
                eval_total_loss_train_history.append(None)
    
    df = get_df()
    return model, df


def train_model_disk(
    model,
    manager: DiskBackedBufferManagerCholesky,
    optimizer,
    num_epochs: int,
    num_samples: int,
    batch_size: int = 2000,
    eval_every: int = 100,
    eval_batch_size: int = 2000,
    scheduler=None,
    plot: bool = False,
    lambda_grad: Optional[float] = None,
    num_directions: int = 1,
    epsilon: float = 1e-5,
    refresh_every: int = 100,
):
    """Train the emulator model using disk-backed Cholesky replay buffers.
    
    This function is designed for training on large pre-generated datasets that
    don't fit in GPU memory. It periodically refreshes in-memory buffers by
    loading random chunks from disk. Uses Cholesky parameterization for the
    covariance matrix.
    
    Args:
        model: The emulator model to train
        manager: DiskBackedBufferManagerCholesky containing pre-generated training data
                 with Cholesky-parameterized covariances
        optimizer: PyTorch optimizer (SGD-like, LBFGS not supported)
        num_epochs: Number of training steps
        num_samples: Number of Monte Carlo samples used when generating the data
                     (used for eps in cross-entropy loss)
        batch_size: Training batch size per K value (default: 2000)
        eval_every: Evaluate every N epochs (default: 100)
        eval_batch_size: Batch size for evaluation per K value (default: 2000)
        scheduler: Optional learning rate scheduler
        plot: Whether to show live plots (default: False)
        lambda_grad: Weight for gradient matching loss. If None, only use CE loss.
                     If provided, manager.use_sobolev must be True.
        num_directions: Number of random directions for directional derivative loss
                        (generated on-the-fly)
        epsilon: Finite difference step size for directional derivative loss
        refresh_every: Refresh in-memory buffers from disk every N steps (default: 100)
    
    Returns:
        (model, df) tuple with trained model and training history DataFrame
    """
    device = get_device()
    model.to(device)
    
    # Validate optimizer type (no LBFGS support)
    if isinstance(optimizer, torch.optim.LBFGS):
        raise ValueError("LBFGS is not supported for disk-backed training. "
                         "Use an SGD-like optimizer instead.")
    
    # Extract configuration from manager
    k_values = manager.k_values
    use_sobolev = lambda_grad is not None
    
    # Validate Sobolev configuration
    if use_sobolev and not manager.use_sobolev:
        raise ValueError("lambda_grad is set but manager.use_sobolev is False. "
                         "The pre-generated data must include gradients for Sobolev training.")
    
    # Create string identifier for K values
    k_str = "_".join(str(k) for k in k_values)
    
    # Initialize history lists
    train_loss_history = []
    eval_ce_loss_history = []
    eval_rmse_loss_history = []
    dropout_history = []
    grad_norm_history = []
    eval_ce_loss_train_history = []
    eval_rmse_loss_train_history = []

    if use_sobolev:
        train_ce_loss_history = []
        train_grad_loss_history = []
        eval_total_loss_history = []
        eval_grad_loss_history = []
        eval_total_loss_train_history = []
    
    mse_loss = nn.MSELoss()
    date_str = datetime.now().strftime("%Y-%m-%d")
    df_filename = os.path.join("fitting-results", f"train_{model.name}_{k_str}.csv")
    df_filename_date = os.path.join("fitting-results", f"train_{model.name}_{k_str}_{date_str}.csv")
    batch_size_curr = batch_size
    
    checkpoint_dir = "model-checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("fitting-results", exist_ok=True)
    
    manager.refresh_all()
    gumbel = manager.gumbel  # Get gumbel flag from manager
    
    best_eval_loss = float('inf')
    eval_forward_times = []
    
    def get_df():
        df_dict = {
            "train_loss": train_loss_history,
            "eval_ce_loss": eval_ce_loss_history,
            "eval_rmse_loss": eval_rmse_loss_history,
            "eval_ce_loss_train": eval_ce_loss_train_history,
            "eval_rmse_loss_train": eval_rmse_loss_train_history,
            "dropout": dropout_history
        }
        if use_sobolev:
            df_dict["train_ce_loss"] = train_ce_loss_history
            df_dict["train_grad_loss"] = train_grad_loss_history
            df_dict["eval_total_loss"] = eval_total_loss_history
            df_dict["eval_grad_loss"] = eval_grad_loss_history
            df_dict["eval_total_loss_train"] = eval_total_loss_train_history
        return pd.DataFrame(df_dict)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax0 = ax[0]
        ax1 = ax[1]
        ax0_twin = ax0.twinx()
        plt.ion()
        
        # Colors for different K values
        k_colors = {k: f'C{i}' for i, k in enumerate(k_values)}
    
    optimizer_name = type(optimizer).__name__
    print(f"Training with {optimizer_name} optimizer")
    print(f"Training on K values: {k_values}")
    print(f"Refresh every {refresh_every} steps, batch_size={batch_size}")
    if use_sobolev:
        print(f"Sobolev training: lambda_grad={lambda_grad}, num_directions={num_directions}, epsilon={epsilon}")
    
    for i in trange(num_epochs):
        model.train()
                
        # Refresh buffers from disk (directions generated on-the-fly, no need to regenerate)
        if i > 0 and i % refresh_every == 0:
            manager.refresh_all()
        
        # Training step
        optimizer.zero_grad()
        
        # Accumulate losses across K values
        total_loss = 0.0
        total_ce_loss = 0.0
        total_grad_loss = 0.0
        
        for k in k_values:
            # Sample from Cholesky buffer
            v, L, y, grad_v, grad_L = manager.sample(k, batch_size_curr, split='train')
            
            if use_sobolev:
                target_probs = y  # Already probabilities from the buffer
                
                loss, ce_loss, grad_loss = directional_derivative_loss(
                    model, v, L, target_probs, grad_v, grad_L,
                    num_directions=num_directions,
                    lambda_grad=lambda_grad,
                    epsilon=epsilon,
                    gumbel=gumbel
                )
                total_ce_loss += ce_loss
                total_grad_loss += grad_loss
            else:
                # Reconstruct canonical form for model forward pass
                Omega = L @ L.transpose(-1, -2)
                Sigma = omega_to_cov(Omega)
                v_star, cov_star, c_star = canonicalize(v, Sigma)
                
                logits = model(v_star, cov_star, c_star)
                loss = multinomial_cross_entropy_loss(
                    logits, y, eps=1./num_samples, log=False)
            
            total_loss += loss
        
        # Average loss across K values
        avg_loss_tensor = total_loss / len(k_values)
        avg_loss_tensor.backward()
        
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        grad_norm_history.append(total_norm.item())

        # Take step if grad is valid
        grad_is_valid = all(
            torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None
        )

        if grad_is_valid:
            optimizer.step()
        else:
            print(f"Epoch {i}: Skipping update due to NaN/Inf gradients")
            optimizer.zero_grad()
        
        avg_loss = avg_loss_tensor.detach().cpu().item()
        
        if use_sobolev:
            train_ce_loss_history.append(total_ce_loss / len(k_values))
            train_grad_loss_history.append(total_grad_loss / len(k_values))
        
        # Model step (if applicable)
        if hasattr(model, 'step'):
            model.step()
        
        # Scheduler step (if provided)
        if scheduler is not None:
            scheduler.step()
        
        train_loss_history.append(avg_loss)
        
        # Record dropout
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                dropout_history.append(module.p)
                break
        else:
            dropout_history.append(0.0)

        # Evaluation
        if (i + 1) % eval_every == 0:
            
            # Sample data once for both eval and train mode metrics
            eval_samples = {}  # K -> (v, L, y, grad_v, grad_L)
            
            for k in k_values:
                v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval = manager.sample(
                    k, eval_batch_size, split='eval')
                
                eval_samples[k] = (v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval)
            
            # Eval mode metrics
            model.eval()
            
            eval_total_loss_sum = 0.0
            eval_ce_loss_sum = 0.0
            eval_grad_loss_sum = 0.0
            eval_rmse_sum = 0.0
            
            # For plotting: collect predictions and targets per K
            plot_data = {}  # K -> (yhat, y_probs)
            
            for k in k_values:
                v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval = eval_samples[k]
                
                # Reconstruct canonical form for model forward pass
                Omega_eval = L_eval @ L_eval.transpose(-1, -2)
                Sigma_eval = omega_to_cov(Omega_eval)
                v_star_eval, cov_star_eval, c_star_eval = canonicalize(v_eval, Sigma_eval)
                
                with torch.no_grad():
                    logits_eval = model(v_star_eval, cov_star_eval, c_star_eval)
                
                    if use_sobolev:
                        target_probs_eval = y_eval
                        
                        eval_total, eval_ce, eval_grad = directional_derivative_loss(
                            model, v_eval, L_eval, target_probs_eval,
                            grad_v_eval, grad_L_eval,
                            num_directions=num_directions,
                            lambda_grad=lambda_grad,
                            epsilon=epsilon,
                            gumbel=gumbel
                        )
                        eval_total_loss_sum += eval_total.item()
                        eval_ce_loss_sum += eval_ce
                        eval_grad_loss_sum += eval_grad
                    else:
                        eval_ce = multinomial_cross_entropy_loss(
                            logits_eval, y_eval, eps=1. / num_samples, log=False)
                        eval_ce_loss_sum += eval_ce.item()
                    
                    yhat_eval = F.softmax(logits_eval, dim=-1)
                    y_eval_probs = y_eval
                    rmse = torch.sqrt(mse_loss(y_eval_probs, yhat_eval)).item()
                    eval_rmse_sum += rmse
                    
                    plot_data[k] = (yhat_eval.cpu(), y_eval_probs.cpu())
            
            # Average across K values (eval mode)
            num_k = len(k_values)
            avg_eval_ce = eval_ce_loss_sum / num_k
            avg_eval_rmse = eval_rmse_sum / num_k
            
            eval_ce_loss_history.append(avg_eval_ce)
            eval_rmse_loss_history.append(avg_eval_rmse)
            
            if use_sobolev:
                avg_eval_total = eval_total_loss_sum / num_k
                avg_eval_grad = eval_grad_loss_sum / num_k
                eval_total_loss_history.append(avg_eval_total)
                eval_grad_loss_history.append(avg_eval_grad)
                eval_comparison_loss = avg_eval_total
            else:
                eval_comparison_loss = avg_eval_ce
            
            # Train mode metrics (same samples)
            model.train()
            
            eval_ce_loss_train_sum = 0.0
            eval_rmse_train_sum = 0.0
            eval_total_loss_train_sum = 0.0
            
            for k in k_values:
                v_eval, L_eval, y_eval, grad_v_eval, grad_L_eval = eval_samples[k]
                
                # Reconstruct canonical form for model forward pass
                Omega_eval = L_eval @ L_eval.transpose(-1, -2)
                Sigma_eval = omega_to_cov(Omega_eval)
                v_star_eval, cov_star_eval, c_star_eval = canonicalize(v_eval, Sigma_eval)
                
                with torch.no_grad():
                    logits_eval_train = model(v_star_eval, cov_star_eval, c_star_eval)
                    
                    if use_sobolev:
                        target_probs_eval = y_eval
                        
                        eval_total_train, eval_ce_train, _ = directional_derivative_loss(
                            model, v_eval, L_eval, target_probs_eval,
                            grad_v_eval, grad_L_eval,
                            num_directions=num_directions,
                            lambda_grad=lambda_grad,
                            epsilon=epsilon,
                            gumbel=gumbel
                        )
                        
                        eval_total_loss_train_sum += eval_total_train.item()
                        eval_ce_loss_train_sum += eval_ce_train
                    else:
                        eval_ce_train = multinomial_cross_entropy_loss(
                            logits_eval_train, y_eval, eps=1. / num_samples, log=False)
                        eval_ce_loss_train_sum += eval_ce_train.item()
                    
                    yhat_eval_train = F.softmax(logits_eval_train, dim=-1)
                    y_eval_probs = y_eval
                    rmse_train = torch.sqrt(mse_loss(y_eval_probs, yhat_eval_train)).item()
                    eval_rmse_train_sum += rmse_train
            
            # Average across K values (train mode)
            eval_ce_loss_train_history.append(eval_ce_loss_train_sum / num_k)
            eval_rmse_loss_train_history.append(eval_rmse_train_sum / num_k)
            
            if use_sobolev:
                eval_total_loss_train_history.append(eval_total_loss_train_sum / num_k)
            
            # CPU timing (use first K for timing)
            k_for_timing = k_values[0]
            v_timing, L_timing, _, _, _ = eval_samples[k_for_timing]
            
            # Reconstruct canonical form for timing
            Omega_timing = L_timing @ L_timing.transpose(-1, -2)
            Sigma_timing = omega_to_cov(Omega_timing)
            v_star_timing, cov_star_timing, c_star_timing = canonicalize(v_timing, Sigma_timing)
            
            model_cpu = copy.deepcopy(model).cpu()
            model_cpu.eval()
            
            with torch.no_grad():
                v_star_cpu = v_star_timing.cpu()
                cov_star_cpu = cov_star_timing.cpu()
                c_star_cpu = c_star_timing.cpu()
                _ = model_cpu(v_star_cpu, cov_star_cpu, c_star_cpu)
                forward_start = time.time()
                _ = model_cpu(v_star_cpu, cov_star_cpu, c_star_cpu)
                forward_time = time.time() - forward_start
            eval_forward_times.append(forward_time)
            del model_cpu

            df = get_df()
            df.to_csv(df_filename)
            df.to_csv(df_filename_date)
            
            if eval_comparison_loss < best_eval_loss:
                best_eval_loss = eval_comparison_loss
                save_checkpoint(model, optimizer, i + 1, avg_eval_ce, k_str, checkpoint_dir)

            if plot:
                clear_output(wait=True)
                ax0.clear()
                ax0_twin.clear()
                ax0_twin.yaxis.set_label_position('right')
                ax0_twin.yaxis.tick_right()
                
                epoch_idx = np.arange(1, i + 2)
                ax0.plot(epoch_idx, train_loss_history, label='Train Loss', color='C0', alpha=0.7)
                eval_indices = [j + 1 for j, val in enumerate(eval_ce_loss_history) if val is not None]
                eval_values = [val for val in eval_ce_loss_history if val is not None]
                ax0.plot(eval_indices, eval_values, label='Eval CE Loss', color='C2', alpha=0.7)
                ax0.plot(epoch_idx, train_loss_history[0] / epoch_idx, label='O(1/n)', color='C3', alpha=0.7)
                ax0.set_xscale('log')
                ax0.set_yscale('log')
                ax0.set_xlabel('Epoch')
                ax0.set_ylabel('Loss', color='C0')
                ax0.set_title(f'Training Loss & Grad Norm (Epoch {i + 1}, K={k_str})')
                ax0_twin.plot(epoch_idx, grad_norm_history, label='Grad Norm', color='C1', alpha=0.7)
                ax0_twin.set_yscale('log')
                ax0_twin.set_ylabel('Grad Norm', color='C1')
                lines1, labels1 = ax0.get_legend_handles_labels()
                lines2, labels2 = ax0_twin.get_legend_handles_labels()
                ax0.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

                ax1.clear()
                for k in k_values:
                    yhat, y_probs = plot_data[k]
                    ax1.scatter(yhat.numpy(), y_probs.numpy(),
                               color=k_colors[k], label=f"K={k}", alpha=0.2, s=10)
                ax1.plot([0, 1], [0, 1], color="black", linestyle="--", 
                        label="Perfect", alpha=0.7)
                ax1.set_xlabel("Predicted Probability")
                ax1.set_ylabel("Simulated Probability")
                ax1.legend(loc="upper left")
                ax1.set_title("Predicted vs. Actual (All K)")
                
                fig.tight_layout()
                display(fig)
            
            current_time_ms = forward_time * 1000
            avg_time_ms = np.mean(eval_forward_times) * 1000
            time_per_sample_us = (forward_time / eval_batch_size) * 1e6
            avg_time_per_sample_us = (np.mean(eval_forward_times) / eval_batch_size) * 1e6
            
            print(f"\nCPU Forward Pass Timing (batch_size={eval_batch_size}, K={k_for_timing}):")
            print(f"  Current:  {current_time_ms:.2f} ms total, {time_per_sample_us:.2f} µs/sample")
            print(f"  Average:  {avg_time_ms:.2f} ms total, {avg_time_per_sample_us:.2f} µs/sample")
            print(f"  Optimizer: {optimizer_name}")
            print(f"  Training on K values: {k_values}")
            print(f"  In-memory buffer size: {manager.in_memory_size:,}")
            if use_sobolev:
                print(f"  Directional derivative training: lambda_grad={lambda_grad}, "
                      f"num_directions={num_directions}, epsilon={epsilon}")
                print(f"  Last train CE loss: {train_ce_loss_history[-1]:.6f}, "
                      f"grad loss: {train_grad_loss_history[-1]:.6f}")

            if torch.backends.mps.is_available():
                torch.mps.synchronize()
        else:
            eval_ce_loss_history.append(None)
            eval_rmse_loss_history.append(None)
            eval_ce_loss_train_history.append(None)
            eval_rmse_loss_train_history.append(None)
            if use_sobolev:
                eval_total_loss_history.append(None)
                eval_grad_loss_history.append(None)
                eval_total_loss_train_history.append(None)
    
    df = get_df()
    return model, df


class InverseSqrtScheduler:
    """Learning rate scheduler with linear warmup, then inverse square-root decay."""
    
    def __init__(self, optimizer, warmup_steps=1000, min_lr_ratio=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0
    
    def get_lr_multiplier(self):
        step = self.step_count + 1
        
        if step < self.warmup_steps:
            # Linear warmup
            return step / self.warmup_steps
        else:
            # Inverse square-root decay with minimum ratio
            # At step == warmup_steps, multiplier = 1.0
            # Then decays as sqrt(1/step) relative to warmup_steps
            decay = (self.warmup_steps / step)**0.5
            return max(decay, self.min_lr_ratio)
    
    def step(self):
        self.step_count += 1
        multiplier = self.get_lr_multiplier()
        
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group['lr'] = base_lr * multiplier
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        return {
            'step_count': self.step_count,
            'warmup_steps': self.warmup_steps,
            'min_lr_ratio': self.min_lr_ratio
        }
    
    def load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        if 'warmup_steps' in state_dict:
            self.warmup_steps = state_dict['warmup_steps']
        if 'min_lr_ratio' in state_dict:
            self.min_lr_ratio = state_dict['min_lr_ratio']