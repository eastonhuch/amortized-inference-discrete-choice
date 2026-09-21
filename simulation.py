# simulation.py
"""
Core simulation functions for multinomial probit model estimation.

Model Setup (general k alternatives):
- Alternative k (index k-1) is the baseline with zero utility and zero error variance
- Each non-baseline alternative j has its own covariate vector X[i,j,:] for observation i
- V[i,j] = X[i,j,:] @ beta for j = 0, ..., k-2; V[i,k-1] = 0
- beta: (p,) coefficient vector (shared across alternatives)

We support both a Dense and Factor-structured covariance matrix

Correlated-Gumbel extension (dense covariance only):
- The outside option (index k-1, V[i,k-1] = 0) is the base alternative: it has its own
  Gumbel(-gamma, 1) error with unit scale and zero covariance with the inside alternatives,
  so the full scale matrix is blockdiag(Omega, 1) instead of blockdiag(Omega, 0).
- Omega (inside alternatives) keeps the probit dense parameterization, including Omega_11 = 1.
- Errors are eps = Sigma^{1/2} z with the symmetric square root and z_j ~ iid Gumbel(-gamma, 1),
  as in training.simulate_data.
- The Gumbel emulator expects the base alternative at index 0 (its is_first indicator), so
  alternatives are reordered to put the outside option first only when the emulator is called.
- GHK is probit-only; the Gumbel study uses softmax-smoothed sample averages instead.
"""

import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.special import gammaln, multigammaln
import math
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Dict, Any, Optional, Callable, Tuple, List, TypedDict, Union
import os

from utils import canonicalize
from training import load_checkpoint


# =============================================================================
# Constants
# =============================================================================

DEFAULT_EIGENVALUE_FLOOR = 1e-8
DEFAULT_LOG_L_DIAG_LOWER_BOUND = -8.0
DEFAULT_BARRIER_STRENGTH = 0.1
DEFAULT_COV_PRIOR_STRENGTH = 0.1
DEFAULT_FACTOR_BARRIER_STRENGTH = 0.1
CHOLESKY_REGULARIZATIONS = [0.0, 1e-6, 1e-4, 1e-2, 1.0]
LOG_PROB_CLAMP_MIN = -30.0
Z_CLAMP_MIN = -8.0
Z_CLAMP_MAX = 8.0
CONFIDENCE_Z_95 = 1.96
DEFAULT_FACTOR_LOG_PSI_STD = 0.1
DEFAULT_FACTOR_GAMMA_STD = 0.2
DEFAULT_LOG_PSI_LOWER_BOUND = -8.0

# Correlated-Gumbel errors and the softmax-smoothed simulator
ERROR_DISTRIBUTIONS = ("probit", "gumbel")
EULER_MASCHERONI = 0.57721566490153286060651209  # same constant as training.simulate_data
SQRT_EIGENVALUE_FLOOR = 1e-12  # eigenvalue floor inside the symmetric square root
DEFAULT_SIM_TAU = 5.0
DEFAULT_SIM_NUM_SAMPLES = 1000
DEFAULT_SIM_MEMORY_GB = 8.0  # memory budget for one smoothed-simulator fit
DEFAULT_SIM_CHUNK_ELEMENTS = 2**23  # rows x draws x K per simulation chunk
SIM_MEMORY_RESERVE_GB = 1.0  # part of the budget kept free for everything else
SIM_POOL_MEMORY_SHARE = 0.6  # share of the remaining budget the draw pool may use
SIM_GRAPH_SAFETY_FACTOR = 1.25  # allowance on the estimated autograd memory
SIM_POOL_BLOCK_ELEMENTS = 2**24  # draws are generated in blocks of this many floats
SIM_SEED_OFFSET = 500  # per-replication seed offset for the draw pool (same as GHK)

# =============================================================================
# TypedDicts
# =============================================================================

class CholeskyParamsDict(TypedDict):
    beta: torch.Tensor
    log_L_diag: Optional[torch.Tensor]
    L_lower: Optional[torch.Tensor]


class CovarianceParamsDict(TypedDict):
    beta: torch.Tensor
    variances: Optional[torch.Tensor]
    covariances: Optional[torch.Tensor]


class FactorNativeParamsDict(TypedDict):
    """Native (optimization) scale parameters for anchored factor model."""
    beta: torch.Tensor
    log_psi: Optional[torch.Tensor]  # Log idiosyncratic variances log(psi_2), ..., log(psi_{K-1})
    gamma: Optional[torch.Tensor]  # Factor loadings gamma_2, ..., gamma_{K-1}


class FactorTransformedParamsDict(TypedDict):
    """Transformed (reported) scale parameters for anchored factor model."""
    beta: torch.Tensor
    psi: Optional[torch.Tensor]  # Idiosyncratic variances psi_2, ..., psi_{K-1}
    gamma: Optional[torch.Tensor]  # Factor loadings gamma_2, ..., gamma_{K-1}
    

class SimulationStatistics(TypedDict):
    rms_bias: Tuple[float, float]
    rmse: Tuple[float, float]
    empirical_se: Tuple[float, float]
    sqrt_avg_se_sq: Tuple[float, float]
    se_ratio: Tuple[float, float]
    coverage: Tuple[float, float]
    conv_rate: Tuple[float, float]
    grad_norm: Tuple[float, float]
    elapsed_time: Tuple[float, float]
    n_iter: Tuple[float, float]
    n: int
    dgp: str
    estimation_method: str


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class SimulationConfig:
    sample_sizes: List[int] = field(default_factory=lambda: [1000, 10000, 100000])
    num_reps: int = 100
    p: int = 2
    k: int = 3
    checkpoint_path: str = "./model-checkpoints/checkpoint_emulator_3.pt"
    seed: Optional[int] = 1
    covariance_structure: str = "dense"  # "dense" or "factor"
    error_distribution: str = "probit"  # "probit" or "gumbel"


@dataclass
class OptimizationConfig:
    max_iter: int = 10000
    loss_tol_base: float = 1e-8
    grad_tol: float = 1e-9
    history_size: int = 50


@dataclass
class GHKConfig:
    num_draws: int = 100
    seed: Optional[int] = None


@dataclass
class SmoothedSimConfig:
    """
    Softmax-smoothed sample-average simulator (correlated-Gumbel errors).

    tau: softmax temperature, applied to the level-scale utilities (outside option has
        unit Gumbel scale), as in training._compute_gradients.
    num_samples: number of error draws R per observation.
    seed: seed for the pregenerated draw pool (set per replication by run_simulation_study).
    memory_gb: memory budget; decides the pool size and whether chunks are checkpointed.
    chunk_elements: rows x draws x K per simulation chunk.
    max_pool_rows: optional cap on the number of draw sets (forces cycling; for testing).
    use_checkpoint: None decides from the memory budget; True/False forces it.
    """
    tau: float = DEFAULT_SIM_TAU
    num_samples: int = DEFAULT_SIM_NUM_SAMPLES
    seed: Optional[int] = None
    memory_gb: float = DEFAULT_SIM_MEMORY_GB
    chunk_elements: int = DEFAULT_SIM_CHUNK_ELEMENTS
    max_pool_rows: Optional[int] = None
    use_checkpoint: Optional[bool] = None


def smoothed_method_label(tau: float, num_samples: int) -> str:
    """Display label for the smoothed simulator, e.g. Sim(5,500)."""
    return f"Sim({float(tau):g},{int(num_samples)})"


def smoothed_method_file_label(tau: float, num_samples: int) -> str:
    """File-name label for the smoothed simulator, e.g. Sim_tau5_R500."""
    return f"Sim_tau{float(tau):g}_R{int(num_samples)}"


@dataclass
class SmoothedMemoryPlan:
    pool_rows: int  # number of draw sets; observation i uses draw set i mod pool_rows
    use_checkpoint: bool  # checkpoint simulation chunks on the optimizer path
    pool_bytes: int
    graph_bytes: int  # estimated autograd memory without checkpointing
    available_bytes: int


def plan_smoothed_memory(n: int, k: int, config: SmoothedSimConfig) -> SmoothedMemoryPlan:
    """
    Decide the draw-pool size and whether to checkpoint, from the memory budget.

    The pool holds one set of R x K draws per observation when it fits in
    SIM_POOL_MEMORY_SHARE of the budget (after SIM_MEMORY_RESERVE_GB); otherwise draw sets
    are cycled (observation i uses set i mod pool_rows). Without checkpointing, autograd
    keeps log_softmax(tau U) (K floats per draw) and the chosen column (1 float per draw) for
    every observation, plus gathered draw copies when the pool cycles. If pool plus that
    estimate exceeds the budget, chunks are checkpointed (recomputed in the backward pass).
    """
    bytes_per_float = 4
    per_row = int(config.num_samples) * int(k)
    budget = int(float(config.memory_gb) * 2**30)
    available = max(0, budget - int(SIM_MEMORY_RESERVE_GB * 2**30))

    if config.max_pool_rows is not None:
        pool_rows = max(1, min(n, int(config.max_pool_rows)))
    elif n * per_row * bytes_per_float <= SIM_POOL_MEMORY_SHARE * available:
        pool_rows = n
    else:
        pool_rows = int(SIM_POOL_MEMORY_SHARE * available) // (per_row * bytes_per_float)
        pool_rows = max(1, min(n, pool_rows))
    pool_bytes = pool_rows * per_row * bytes_per_float

    graph_floats = n * int(config.num_samples) * (int(k) + 1)
    if pool_rows < n:
        graph_floats += n * per_row  # gathered copies of the cycled draws
    graph_bytes = int(SIM_GRAPH_SAFETY_FACTOR * graph_floats * bytes_per_float)

    if config.use_checkpoint is None:
        use_checkpoint = pool_bytes + graph_bytes > available
    else:
        use_checkpoint = bool(config.use_checkpoint)

    return SmoothedMemoryPlan(
        pool_rows=pool_rows,
        use_checkpoint=use_checkpoint,
        pool_bytes=pool_bytes,
        graph_bytes=graph_bytes,
        available_bytes=available,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def build_lower_triangular_indices(size: int) -> List[Tuple[int, int]]:
    """
    Build list of (i, j) indices for strictly lower triangular elements,
    excluding the first row/column.
    
    Returns indices for positions (1,0), (2,0), (2,1), (3,0), (3,1), (3,2), ...
    """
    indices = []
    for i in range(1, size):
        for j in range(i):
            indices.append((i, j))
    return indices


def safe_cholesky(Sigma: torch.Tensor) -> torch.Tensor:
    """
    Attempt Cholesky decomposition with increasingly aggressive regularization.
    """
    for reg in CHOLESKY_REGULARIZATIONS:
        try:
            if reg > 0:
                warnings.warn(f"Cholesky failed, adding {reg} to diagonal")
                Sigma_reg = Sigma + reg * torch.eye(Sigma.shape[0], device=Sigma.device, dtype=Sigma.dtype)
            else:
                Sigma_reg = Sigma
            return torch.linalg.cholesky(Sigma_reg)
        except RuntimeError:
            continue
    
    warnings.warn("Cholesky decomposition failed after all regularization attempts, returning NaNs")
    return torch.full_like(Sigma, float('nan'))


def robust_matrix_inverse(
    A: torch.Tensor, 
    eigenvalue_floor: float = DEFAULT_EIGENVALUE_FLOOR
) -> torch.Tensor:
    """
    Compute a robust matrix inverse using spectral decomposition.
    """
    original_device = A.device
    if A.device.type == 'mps':
        A = A.cpu()
    
    A = (A + A.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    valid_mask = torch.isfinite(eigenvalues) & (eigenvalues > eigenvalue_floor)
    eigenvalues_inv = torch.zeros_like(eigenvalues)
    eigenvalues_inv[valid_mask] = 1.0 / eigenvalues[valid_mask]
    A_inv = eigenvectors @ torch.diag(eigenvalues_inv) @ eigenvectors.T
    
    return A_inv.to(original_device)


def compute_log_det_jacobian_cholesky_to_cov(
    log_L_diag: torch.Tensor,
    L_lower: torch.Tensor,
    k_minus_1: int,
    L_lower_indices: List[Tuple[int, int]]
) -> torch.Tensor:
    """Compute log|det(J)| for Cholesky -> Covariance transformation using autodiff."""
    n_diag = k_minus_1 - 1
    n_lower = (k_minus_1 - 1) * k_minus_1 // 2
    
    if n_diag == 0 and n_lower == 0:
        return torch.tensor(0.0, device=log_L_diag.device if log_L_diag is not None else 'cpu')
    
    device = log_L_diag.device if log_L_diag is not None else L_lower.device
    dtype = log_L_diag.dtype if log_L_diag is not None else L_lower.dtype
    
    def transform_fn(params_flat: torch.Tensor) -> torch.Tensor:
        log_L_diag_in = params_flat[:n_diag] if n_diag > 0 else None
        L_lower_in = params_flat[n_diag:] if n_lower > 0 else None
        
        # Build Cholesky factor
        L = torch.zeros(k_minus_1, k_minus_1, device=device, dtype=dtype)
        L[0, 0] = 1.0
        
        if log_L_diag_in is not None:
            for i in range(1, k_minus_1):
                L[i, i] = torch.exp(log_L_diag_in[i - 1])
        else:
            for i in range(1, k_minus_1):
                L[i, i] = 1.0
        
        if L_lower_in is not None:
            for idx, (i, j) in enumerate(L_lower_indices):
                L[i, j] = L_lower_in[idx]
        
        # Compute covariance
        Sigma = L @ L.T
        
        # Extract free parameters: variances[1:] and lower triangular covariances
        variances = torch.diag(Sigma)[1:]
        
        covariances = []
        for i, j in L_lower_indices:
            covariances.append(Sigma[i, j])
        
        if covariances:
            covariances = torch.stack(covariances)
            return torch.cat([variances, covariances])
        else:
            return variances
    
    # Build input parameter vector
    parts = []
    if log_L_diag is not None:
        parts.append(log_L_diag)
    if L_lower is not None:
        parts.append(L_lower)
    params_flat = torch.cat(parts) if parts else torch.tensor([], device=device, dtype=dtype)
    
    if params_flat.numel() == 0:
        return torch.tensor(0.0, device=device, dtype=dtype)
    
    J = torch.func.jacfwd(transform_fn)(params_flat)
    _, logabsdet = torch.linalg.slogdet(J)
    return logabsdet


def sample_cholesky_params(k: int, device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sample Cholesky parameters by generating a random covariance matrix
    and extracting its Cholesky factor.
    """
    k_minus_1 = k - 1
    n_L_diag_params = max(0, k_minus_1 - 1)
    n_L_lower_params = (k_minus_1 - 1) * k_minus_1 // 2 if k_minus_1 > 1 else 0
    
    if n_L_diag_params == 0 and n_L_lower_params == 0:
        return None, None
    
    max_attempts = 100
    for _ in range(max_attempts):
        z = torch.randn(k_minus_1, k+10, device=device)
        cov = z @ z.T
        
        scale = cov[0, 0]
        if scale <= 0:
            continue
        cov = cov / (scale + 1e-8)
        
        L, info = torch.linalg.cholesky_ex(cov)
        
        if info == 0:
            log_L_diag = None
            if n_L_diag_params > 0:
                L_diag = torch.diag(L)[1:]
                log_L_diag = torch.log(L_diag)
            
            L_lower = None
            if n_L_lower_params > 0:
                tril_idx = torch.tril_indices(k_minus_1, k_minus_1, offset=-1, device=device)
                L_lower = L[tril_idx[0], tril_idx[1]]
            
            return log_L_diag, L_lower
    
    raise RuntimeError(f"Failed to sample valid Cholesky parameters after {max_attempts} attempts")


def sample_factor_params(
    k: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sample factor model parameters on native (optimization) scale.
    
    Uses anchored normalization: psi_1 = 0, gamma_1 = 1.
    Samples log_psi ~ N(0, 0.1^2) and gamma ~ N(0, 0.2^2).
    
    Args:
        k: Number of alternatives
        device: Torch device
    
    Returns:
        log_psi: (K-2,) log idiosyncratic variances for psi_2, ..., psi_{K-1}
        gamma: (K-2,) factor loadings gamma_2, ..., gamma_{K-1}
    """
    k_minus_1 = k - 1
    n_params = max(0, k_minus_1 - 1)
    
    if n_params == 0:
        return None, None
    
    # log_psi: sample from normal
    log_psi = DEFAULT_FACTOR_LOG_PSI_STD * torch.randn(n_params, device=device, dtype=torch.float32)
    
    # gamma: sample from normal
    gamma = DEFAULT_FACTOR_GAMMA_STD * torch.randn(n_params, device=device, dtype=torch.float32)
    
    return log_psi, gamma


def generate_covariates(n: int, k: int, p: int, device: torch.device) -> torch.Tensor:
    X = torch.randn(n, k, p, device=device)
    X[:, k-1, :] = 0.0
    return X


def setup_emulator(
    checkpoint_path: str, 
    device: torch.device,
) -> nn.Module:
    """
    Load an emulator from a checkpoint file.
    """    
    print(f"Loading emulator from {checkpoint_path}")
    
    model, epoch, loss = load_checkpoint(checkpoint_path, device=device)
    print(f"  Loaded checkpoint from epoch {epoch} with loss {loss:.6f}")
    
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    
    return model


# =============================================================================
# Correlated-Gumbel Helpers
# =============================================================================

class _SymmetricSqrtFunction(torch.autograd.Function):
    """
    Symmetric PSD square root A^{1/2} = Q diag(s) Q^T, s = sqrt(eigenvalues).

    torch.linalg.eigh's own derivative is not finite when eigenvalues repeat (e.g. Omega = I,
    the optimizer's starting point). The square root itself is smooth there, and its
    derivative in the eigenbasis (Daleckii-Krein) is

        d(A^{1/2}) = Q [ (Q^T dA Q)_ij / (s_i + s_j) ] Q^T,

    which is finite whenever A is positive definite. Both reverse mode (backward) and forward
    mode (jvp, used by torch.func.jacfwd in compute_standard_errors) use this formula; the
    map is self-adjoint, so the two coincide. generate_vmap_rule lets torch.func batch it.
    """
    generate_vmap_rule = True

    @staticmethod
    def forward(A):
        A_sym = 0.5 * (A + A.transpose(-1, -2))
        eigenvalues, Q = torch.linalg.eigh(A_sym)
        s = torch.sqrt(torch.clamp(eigenvalues, min=SQRT_EIGENVALUE_FLOOR))
        root = (Q * s.unsqueeze(-2)) @ Q.transpose(-1, -2)
        return root, Q, s

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, Q, s = output
        ctx.mark_non_differentiable(Q, s)
        ctx.save_for_backward(Q, s)
        ctx.save_for_forward(Q, s)

    @staticmethod
    def _frechet(Q, s, dA):
        dA_sym = 0.5 * (dA + dA.transpose(-1, -2))
        G = Q.transpose(-1, -2) @ dA_sym @ Q
        denom = s.unsqueeze(-1) + s.unsqueeze(-2)
        return Q @ (G / denom) @ Q.transpose(-1, -2)

    @staticmethod
    def backward(ctx, grad_root, grad_Q, grad_s):
        Q, s = ctx.saved_tensors
        if grad_root is None:
            return None
        return _SymmetricSqrtFunction._frechet(Q, s, grad_root)

    @staticmethod
    def jvp(ctx, dA):
        Q, s = ctx.saved_tensors
        return _SymmetricSqrtFunction._frechet(Q, s, dA), None, None


def symmetric_sqrt(A: torch.Tensor) -> torch.Tensor:
    """Symmetric PSD square root of a symmetric PSD matrix (or batch), with safe derivatives."""
    return _SymmetricSqrtFunction.apply(A)[0]


def gumbel_full_scale_matrix(S: torch.Tensor) -> torch.Tensor:
    """
    Gumbel scale matrix blockdiag(Omega, 1) from the probit-style blockdiag(Omega, 0).

    The outside option (index k-1) gets unit variance and zero covariance with the inside
    alternatives. Written additively so it works under autograd and torch.func.
    """
    k = S.shape[-1]
    outside_unit = torch.zeros(k, k, device=S.device, dtype=S.dtype)
    outside_unit[k - 1, k - 1] = 1.0
    return S + outside_unit


def gumbel_error_root(S: torch.Tensor) -> torch.Tensor:
    """
    Full-size symmetric root blockdiag(Omega^{1/2}, 1) of the Gumbel scale matrix.

    Only the inside block of S is used, so S may be either the probit-style
    blockdiag(Omega, 0) or the Gumbel blockdiag(Omega, 1).
    """
    k = S.shape[-1]
    root_inside = symmetric_sqrt(S[..., :k - 1, :k - 1])
    root_padded = F.pad(root_inside, (0, 1, 0, 1), mode='constant', value=0)
    outside_unit = torch.zeros(k, k, device=S.device, dtype=S.dtype)
    outside_unit[k - 1, k - 1] = 1.0
    return root_padded + outside_unit


def outside_first_permutation(k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reordering that moves the outside option (index k-1) to index 0 for the Gumbel emulator.

    perm = [k-1, 0, 1, ..., k-2]: emulator position e holds original alternative perm[e].
    inv_perm = [1, 2, ..., k-1, 0]: original alternative j sits at emulator position inv_perm[j].
    So V_emu = V[:, perm], and log_probs = log_probs_emu[:, inv_perm] restores the original order.
    """
    perm = torch.cat([torch.tensor([k - 1]), torch.arange(k - 1)]).to(device)
    inv_perm = torch.cat([torch.arange(1, k), torch.tensor([0])]).to(device)
    return perm, inv_perm


def standard_gumbel_distribution(device: torch.device, dtype: torch.dtype = torch.float32):
    """Gumbel(-gamma, 1), the standardized error distribution used in training.simulate_data."""
    return torch.distributions.Gumbel(
        loc=torch.tensor(-EULER_MASCHERONI, device=device, dtype=dtype),
        scale=torch.tensor(1.0, device=device, dtype=dtype),
    )


def generate_gumbel_pool(
    pool_rows: int,
    num_samples: int,
    k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    block_elements: int = SIM_POOL_BLOCK_ELEMENTS,
) -> torch.Tensor:
    """
    Pool of standardized Gumbel(-gamma, 1) draws of shape (pool_rows, num_samples, k), drawn
    from the global RNG in blocks of whole rows so sampling temporaries stay small.
    """
    pool = torch.empty(pool_rows, num_samples, k, device=device, dtype=dtype)
    gumbel = standard_gumbel_distribution(device, dtype)
    rows_per_block = max(1, int(block_elements) // (num_samples * k))
    for start in range(0, pool_rows, rows_per_block):
        end = min(start + rows_per_block, pool_rows)
        pool[start:end] = gumbel.sample((end - start, num_samples, k))
    return pool


# =============================================================================
# Data Generating Processes
# =============================================================================

class DataGeneratingProcess(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def sample_choices(
        self, 
        V: torch.Tensor, 
        S: torch.Tensor, 
        S_chol: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        pass


class ExactProbitDGP(DataGeneratingProcess):
    @property
    def name(self) -> str:
        return "Probit"
    
    def sample_choices(
        self, 
        V: torch.Tensor, 
        S: torch.Tensor, 
        S_chol: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        device = V.device
        n, k = V.shape
        
        eps = torch.zeros(n, k, device=device)
        S_chol_reduced = S_chol[:k-1, :k-1]
        eps_iid = torch.randn(n, k-1, device=device)
        eps_reduced = eps_iid @ S_chol_reduced.T
        eps[:, :k-1] = eps_reduced
        
        U = V + eps
        y = torch.argmax(U, dim=1)
        return y


class EmulatorDGP(DataGeneratingProcess):
    """DGP that uses the neural network emulator."""
    
    @property
    def name(self) -> str:
        return "Emulator"
    
    def sample_choices(
        self, 
        V: torch.Tensor, 
        S: torch.Tensor, 
        S_chol: torch.Tensor,
        model: nn.Module = None,
        **kwargs
    ) -> torch.Tensor:
        if model is None:
            raise ValueError("EmulatorDGP requires a model argument")
        
        n = V.shape[0]
        S_expanded = S.unsqueeze(0).expand(n, -1, -1)
        
        with torch.no_grad():
            V_star, S_star, C_star = canonicalize(V, S_expanded)
            logits = model(V_star, S_star, C_star)
            probs = F.softmax(logits, dim=-1)
            categorical = dist.Categorical(probs=probs)
            y = categorical.sample()
        return y


class GumbelDGP(DataGeneratingProcess):
    """
    Exact correlated-Gumbel RUM: U = V + Sigma^{1/2} z, z_j ~ iid Gumbel(-gamma, 1).

    Sigma = blockdiag(Omega, 1) with the symmetric square root, so the inside errors are
    Omega^{1/2} z_inside and the outside option's error is its own z (not zero as in probit).
    S may be the probit-style blockdiag(Omega, 0) passed by run_single_replication; only its
    inside block is used. S_chol is ignored: the Cholesky factor is not a valid root here.
    """
    
    @property
    def name(self) -> str:
        return "Gumbel"
    
    def sample_choices(
        self, 
        V: torch.Tensor, 
        S: torch.Tensor, 
        S_chol: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        n, k = V.shape
        with torch.no_grad():
            root = gumbel_error_root(S).to(dtype=V.dtype)
            z = standard_gumbel_distribution(V.device, V.dtype).sample((n, k))
            eps = z @ root.T
            U = V + eps
            y = torch.argmax(U, dim=1)
        return y


class GumbelEmulatorDGP(EmulatorDGP):
    """
    Emulator DGP for correlated-Gumbel errors.

    Builds blockdiag(Omega, 1), moves the outside option to index 0 (the Gumbel emulator's
    base position), samples from the emulator's probabilities, and maps the sampled position
    back to the original alternative index.
    """
    
    def sample_choices(
        self, 
        V: torch.Tensor, 
        S: torch.Tensor, 
        S_chol: torch.Tensor,
        model: nn.Module = None,
        **kwargs
    ) -> torch.Tensor:
        if model is None:
            raise ValueError("GumbelEmulatorDGP requires a model argument")
        if not getattr(model, 'include_c_star', False):
            raise ValueError("GumbelEmulatorDGP requires a Gumbel emulator (include_c_star=True)")
        
        n, k = V.shape
        perm, _ = outside_first_permutation(k, V.device)
        S_emu = gumbel_full_scale_matrix(S).index_select(0, perm).index_select(1, perm)
        V_emu = V.index_select(1, perm)
        S_expanded = S_emu.unsqueeze(0).expand(n, -1, -1)
        
        with torch.no_grad():
            V_star, S_star, C_star = canonicalize(V_emu, S_expanded)
            logits = model(V_star, S_star, C_star)
            probs = F.softmax(logits, dim=-1)
            categorical = dist.Categorical(probs=probs)
            y_emu = categorical.sample()
        return perm[y_emu]


# =============================================================================
# Parameter Specification Classes
# =============================================================================

class ParameterSpec(ABC):
    """Abstract base class for parameter specifications."""
    
    @abstractmethod
    def get_true_values(self) -> torch.Tensor:
        pass
    
    @abstractmethod
    def initialize(self, device: torch.device) -> torch.Tensor:
        pass
    
    @abstractmethod
    def unflatten(self, params: torch.Tensor) -> Dict[str, Any]:
        pass
    
    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
    
    @abstractmethod
    def compute_utilities(self, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_reduced_covariance_matrix(self, *args, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def build_full_covariance_matrix(self, *args, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_param_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_param_types(self) -> List[str]:
        pass


class CovarianceParameterSpec(ParameterSpec):
    """
    Parameter specification using variance/covariance parameterization.
    """
    
    def __init__(
        self,
        p: int,
        k: int,
        beta_true: torch.Tensor,
        variances_true: Optional[torch.Tensor],
        covariances_true: Optional[torch.Tensor],
        device: torch.device
    ):
        self.p = p
        self.k = k
        self.k_minus_1 = k - 1
        self.beta_true = beta_true
        self.variances_true = variances_true
        self.covariances_true = covariances_true
        self.device = device
        
        self.n_beta_params = p
        self.n_variance_params = max(0, self.k_minus_1 - 1)
        self.n_covariance_params = (self.k_minus_1 - 1) * self.k_minus_1 // 2 if self.k_minus_1 > 1 else 0
        
        self.covariance_indices = build_lower_triangular_indices(self.k_minus_1)
    
    def get_true_values(self) -> torch.Tensor:
        parts = [self.beta_true.flatten()]
        if self.variances_true is not None:
            parts.append(self.variances_true.flatten())
        if self.covariances_true is not None:
            parts.append(self.covariances_true.flatten())
        return torch.cat(parts)
    
    def initialize(self, device: torch.device) -> torch.Tensor:
        parts = [torch.zeros(self.n_beta_params, device=device)]
        if self.n_variance_params > 0:
            parts.append(torch.ones(self.n_variance_params, device=device))
        if self.n_covariance_params > 0:
            parts.append(torch.zeros(self.n_covariance_params, device=device))
        return torch.cat(parts)
    
    def unflatten(self, params: torch.Tensor) -> CovarianceParamsDict:
        idx = 0
        beta = params[idx:idx + self.n_beta_params]
        idx += self.n_beta_params
        
        variances = None
        if self.n_variance_params > 0:
            variances = params[idx:idx + self.n_variance_params]
            idx += self.n_variance_params
        
        covariances = None
        if self.n_covariance_params > 0:
            covariances = params[idx:idx + self.n_covariance_params]
        
        return {'beta': beta, 'variances': variances, 'covariances': covariances}
    
    @property
    def num_params(self) -> int:
        return self.n_beta_params + self.n_variance_params + self.n_covariance_params
    
    def compute_utilities(self, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return torch.einsum('nkp,p->nk', X, beta)
    
    def build_reduced_covariance_matrix(
        self,
        variances: Optional[torch.Tensor],
        covariances: Optional[torch.Tensor]
    ) -> torch.Tensor:
        device = self.device
        dtype = torch.float32
        
        if variances is not None:
            device = variances.device
            dtype = variances.dtype
        elif covariances is not None:
            device = covariances.device
            dtype = covariances.dtype
        
        Sigma = torch.zeros(self.k_minus_1, self.k_minus_1, device=device, dtype=dtype)
        
        Sigma[0, 0] = 1.0
        if self.k_minus_1 > 1 and variances is not None:
            for i in range(1, self.k_minus_1):
                Sigma[i, i] = variances[i - 1]
        
        if covariances is not None:
            for idx, (i, j) in enumerate(self.covariance_indices):
                Sigma[i, j] = covariances[idx]
                Sigma[j, i] = covariances[idx]
        
        return Sigma
    
    def build_full_covariance_matrix(
        self,
        variances: Optional[torch.Tensor],
        covariances: Optional[torch.Tensor]
    ) -> torch.Tensor:
        S_reduced = self.build_reduced_covariance_matrix(variances, covariances)
        return F.pad(S_reduced, (0, 1, 0, 1), mode='constant', value=0)
    
    def get_param_names(self) -> List[str]:
        names = []
        for i in range(self.p):
            names.append(f'beta[{i}]')
        for i in range(self.n_variance_params):
            names.append(f'var[{i + 1}]')
        for idx, (i, j) in enumerate(self.covariance_indices):
            names.append(f'cov[{i},{j}]')
        return names
    
    def get_param_types(self) -> List[str]:
        types = []
        for _ in range(self.n_beta_params):
            types.append('beta')
        for _ in range(self.n_variance_params):
            types.append('variance')
        for _ in range(self.n_covariance_params):
            types.append('covariance')
        return types
    
    @staticmethod
    def from_cholesky_spec(chol_spec: 'CholeskyParameterSpec') -> 'CovarianceParameterSpec':
        Sigma = chol_spec.build_reduced_covariance_matrix(
            chol_spec.log_L_diag_true, chol_spec.L_lower_true
        )
        
        variances_true = None
        if chol_spec.k - 1 > 1:
            variances_true = torch.diag(Sigma)[1:].clone()
        
        covariances_true = None
        n_cov = (chol_spec.k - 2) * (chol_spec.k - 1) // 2 if chol_spec.k > 2 else 0
        if n_cov > 0:
            covariance_indices = build_lower_triangular_indices(chol_spec.k - 1)
            covariances_true = torch.zeros(n_cov, device=chol_spec.device)
            for idx, (i, j) in enumerate(covariance_indices):
                covariances_true[idx] = Sigma[i, j]
        
        return CovarianceParameterSpec(
            p=chol_spec.p,
            k=chol_spec.k,
            beta_true=chol_spec.beta_true,
            variances_true=variances_true,
            covariances_true=covariances_true,
            device=chol_spec.device
        )
    
    def to_cholesky_params(
        self,
        variances: Optional[torch.Tensor],
        covariances: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.n_variance_params == 0 and self.n_covariance_params == 0:
            return None, None
        
        Sigma = self.build_reduced_covariance_matrix(variances, covariances)
        L = safe_cholesky(Sigma)
        
        if torch.isnan(L).any():
            log_L_diag = torch.full((self.n_variance_params,), float('nan'), device=self.device) if self.n_variance_params > 0 else None
            L_lower = torch.full((self.n_covariance_params,), float('nan'), device=self.device) if self.n_covariance_params > 0 else None
            return log_L_diag, L_lower
        
        log_L_diag = None
        if self.n_variance_params > 0:
            L_diag = torch.diag(L)[1:]
            log_L_diag = torch.log(L_diag)
        
        L_lower = None
        if self.n_covariance_params > 0:
            L_lower = torch.zeros(self.n_covariance_params, device=L.device, dtype=L.dtype)
            for idx, (i, j) in enumerate(self.covariance_indices):
                L_lower[idx] = L[i, j]
        
        return log_L_diag, L_lower


class CholeskyParameterSpec(ParameterSpec):
    """
    Parameter specification using Cholesky parameterization for dense covariance.
    """
    
    def __init__(
        self, 
        p: int,
        k: int,
        beta_true: torch.Tensor,
        log_L_diag_true: Optional[torch.Tensor],
        L_lower_true: Optional[torch.Tensor],
        device: torch.device
    ):
        self.p = p
        self.k = k
        self.k_minus_1 = k - 1
        self.beta_true = beta_true
        self.log_L_diag_true = log_L_diag_true
        self.L_lower_true = L_lower_true
        self.device = device
        
        self.n_beta_params = p
        self.n_L_diag_params = max(0, self.k_minus_1 - 1)
        self.n_L_lower_params = (self.k_minus_1 - 1) * self.k_minus_1 // 2 if self.k_minus_1 > 1 else 0
        
        self.L_lower_indices = build_lower_triangular_indices(self.k_minus_1)
    
    def get_true_values(self) -> torch.Tensor:
        parts = [self.beta_true.flatten()]
        if self.log_L_diag_true is not None:
            parts.append(self.log_L_diag_true.flatten())
        if self.L_lower_true is not None:
            parts.append(self.L_lower_true.flatten())
        return torch.cat(parts)
    
    def initialize(self, device: torch.device) -> torch.Tensor:
        parts = [torch.zeros(self.n_beta_params, device=device)]
        if self.n_L_diag_params > 0:
            parts.append(torch.zeros(self.n_L_diag_params, device=device))
        if self.n_L_lower_params > 0:
            parts.append(torch.zeros(self.n_L_lower_params, device=device))
        return torch.cat(parts)
    
    def unflatten(self, params: torch.Tensor) -> CholeskyParamsDict:
        idx = 0
        beta = params[idx:idx + self.n_beta_params]
        idx += self.n_beta_params
        
        log_L_diag = None
        if self.n_L_diag_params > 0:
            log_L_diag = params[idx:idx + self.n_L_diag_params]
            idx += self.n_L_diag_params
        
        L_lower = None
        if self.n_L_lower_params > 0:
            L_lower = params[idx:idx + self.n_L_lower_params]
        
        return {'beta': beta, 'log_L_diag': log_L_diag, 'L_lower': L_lower}
    
    @property
    def num_params(self) -> int:
        return self.n_beta_params + self.n_L_diag_params + self.n_L_lower_params
    
    def compute_utilities(self, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return torch.einsum('nkp,p->nk', X, beta)
    
    def build_cholesky_factor(
        self, 
        log_L_diag: Optional[torch.Tensor], 
        L_lower: Optional[torch.Tensor]
    ) -> torch.Tensor:
        device = self.device
        dtype = torch.float32
        
        if log_L_diag is not None:
            device = log_L_diag.device
            dtype = log_L_diag.dtype
        elif L_lower is not None:
            device = L_lower.device
            dtype = L_lower.dtype
        
        L = torch.zeros(self.k_minus_1, self.k_minus_1, device=device, dtype=dtype)
        L[0, 0] = 1.0
        
        for i in range(1, self.k_minus_1):
            if log_L_diag is not None:
                L[i, i] = torch.exp(log_L_diag[i-1])
            else:
                L[i, i] = 1.0
        
        if L_lower is not None:
            for idx, (i, j) in enumerate(self.L_lower_indices):
                L[i, j] = L_lower[idx]
        
        return L
    
    def build_reduced_covariance_matrix(
        self, 
        log_L_diag: Optional[torch.Tensor], 
        L_lower: Optional[torch.Tensor]
    ) -> torch.Tensor:
        L = self.build_cholesky_factor(log_L_diag, L_lower)
        return L @ L.T
    
    def build_full_covariance_matrix(
        self, 
        log_L_diag: Optional[torch.Tensor], 
        L_lower: Optional[torch.Tensor]
    ) -> torch.Tensor:
        S_reduced = self.build_reduced_covariance_matrix(log_L_diag, L_lower)
        return F.pad(S_reduced, (0, 1, 0, 1), mode='constant', value=0)
    
    def get_cholesky_factor_full(
        self, 
        log_L_diag: Optional[torch.Tensor], 
        L_lower: Optional[torch.Tensor]
    ) -> torch.Tensor:
        L_reduced = self.build_cholesky_factor(log_L_diag, L_lower)
        return F.pad(L_reduced, (0, 1, 0, 1), mode='constant', value=0)
    
    def get_param_names(self) -> List[str]:
        names = []
        for i in range(self.p):
            names.append(f'beta[{i}]')
        for i in range(self.n_L_diag_params):
            names.append(f'log_L_diag[{i}]')
        for idx, (i, j) in enumerate(self.L_lower_indices):
            names.append(f'L_lower[{i},{j}]')
        return names
    
    def get_param_types(self) -> List[str]:
        types = []
        for _ in range(self.n_beta_params):
            types.append('beta')
        for _ in range(self.n_L_diag_params):
            types.append('L_diag')
        for _ in range(self.n_L_lower_params):
            types.append('L_lower')
        return types
    
    def to_covariance_params(
        self,
        log_L_diag: Optional[torch.Tensor],
        L_lower: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.n_L_diag_params == 0 and self.n_L_lower_params == 0:
            return None, None
        
        Sigma = self.build_reduced_covariance_matrix(log_L_diag, L_lower)
        
        variances = None
        if self.n_L_diag_params > 0:
            variances = torch.diag(Sigma)[1:].clone()
        
        covariances = None
        if self.n_L_lower_params > 0:
            covariances = torch.zeros(self.n_L_lower_params, device=Sigma.device, dtype=Sigma.dtype)
            for idx, (i, j) in enumerate(self.L_lower_indices):
                covariances[idx] = Sigma[i, j]
        
        return variances, covariances


class FactorParameterSpec(ParameterSpec):
    """
    Parameter specification using anchored factor normalization.
    
    Covariance structure: Sigma = Psi + gamma @ gamma.T
    
    Anchored normalization:
    - psi_1 = 0 (fixed)
    - gamma_1 = 1 (fixed)
    - This implies Sigma_11 = 1
    
    Native (optimization) parameterization:
    - beta: (p,) coefficients
    - log_psi: (K-2,) log idiosyncratic variances log(psi_2), ..., log(psi_{K-1})
    - gamma: (K-2,) factor loadings gamma_2, ..., gamma_{K-1}
    
    Transformed (reported) parameterization:
    - beta: (p,) coefficients  
    - psi: (K-2,) idiosyncratic variances psi_2, ..., psi_{K-1}
    - gamma: (K-2,) factor loadings gamma_2, ..., gamma_{K-1} (same as native)
    
    Note: psi_1 = 0 and gamma_1 = 1 are fixed, not estimated or reported.
    """
    
    def __init__(
        self,
        p: int,
        k: int,
        beta_true: torch.Tensor,
        log_psi_true: Optional[torch.Tensor],
        gamma_true: Optional[torch.Tensor],
        device: torch.device,
    ):
        self.p = p
        self.k = k
        self.k_minus_1 = k - 1
        self.beta_true = beta_true
        self.log_psi_true = log_psi_true
        self.gamma_true = gamma_true
        self.device = device
        
        self.n_beta_params = p
        # For K alternatives, we have K-1 dimensional covariance
        # We estimate K-2 log_psi params (psi_1 = 0 is fixed)
        # We estimate K-2 gamma params (gamma_1 = 1 is fixed)
        self.n_log_psi_params = max(0, self.k_minus_1 - 1)
        self.n_gamma_params = max(0, self.k_minus_1 - 1)
    
    def get_true_values(self) -> torch.Tensor:
        """Get true values on native (optimization) scale: (beta, log_psi, gamma)."""
        parts = [self.beta_true.flatten()]
        if self.log_psi_true is not None:
            parts.append(self.log_psi_true.flatten())
        if self.gamma_true is not None:
            parts.append(self.gamma_true.flatten())
        return torch.cat(parts)
    
    def get_true_values_transformed(self) -> torch.Tensor:
        """Get true values on transformed (reported) scale: (beta, psi, gamma)."""
        parts = [self.beta_true.flatten()]
        if self.log_psi_true is not None and self.gamma_true is not None:
            psi, gamma = self.native_to_transformed(self.log_psi_true, self.gamma_true)
            if psi is not None:
                parts.append(psi.flatten())
            if gamma is not None:
                parts.append(gamma.flatten())
        return torch.cat(parts)
    
    def initialize(self, device: torch.device) -> torch.Tensor:
        """Initialize parameters on native scale."""
        parts = [torch.zeros(self.n_beta_params, device=device, dtype=torch.float32)]
        if self.n_log_psi_params > 0:
            # Initialize log_psi to 0 -> psi = 1
            parts.append(torch.zeros(self.n_log_psi_params, device=device, dtype=torch.float32))
        if self.n_gamma_params > 0:
            # Initialize gamma to 0
            parts.append(torch.zeros(self.n_gamma_params, device=device, dtype=torch.float32))
        return torch.cat(parts)
    
    def unflatten(self, params: torch.Tensor) -> FactorNativeParamsDict:
        """Unflatten parameter vector to native scale dictionary."""
        idx = 0
        beta = params[idx:idx + self.n_beta_params]
        idx += self.n_beta_params
        
        log_psi = None
        if self.n_log_psi_params > 0:
            log_psi = params[idx:idx + self.n_log_psi_params]
            idx += self.n_log_psi_params
        
        gamma = None
        if self.n_gamma_params > 0:
            gamma = params[idx:idx + self.n_gamma_params]
        
        return {'beta': beta, 'log_psi': log_psi, 'gamma': gamma}
    
    def unflatten_transformed(self, params: torch.Tensor) -> FactorTransformedParamsDict:
        """Unflatten parameter vector to transformed scale dictionary."""
        idx = 0
        beta = params[idx:idx + self.n_beta_params]
        idx += self.n_beta_params
        
        psi = None
        if self.n_log_psi_params > 0:
            psi = params[idx:idx + self.n_log_psi_params]
            idx += self.n_log_psi_params
        
        gamma = None
        if self.n_gamma_params > 0:
            gamma = params[idx:idx + self.n_gamma_params]
        
        return {'beta': beta, 'psi': psi, 'gamma': gamma}
    
    def native_to_transformed(
        self,
        log_psi: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Transform from native (log_psi, gamma) to reported (psi, gamma) parameters.
        
        Returns psi_2, ..., psi_{K-1} and gamma_2, ..., gamma_{K-1}.
        """
        if self.k_minus_1 <= 1:
            return None, None
        
        # Transform log_psi -> psi
        psi = None
        if log_psi is not None:
            psi = torch.exp(log_psi)
        
        # gamma is unchanged
        return psi, gamma
    
    def transformed_to_native(
        self,
        psi: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Transform from reported (psi, gamma) to native (log_psi, gamma) parameters.
        
        Expects psi_2, ..., psi_{K-1} and gamma_2, ..., gamma_{K-1}.
        """
        if self.k_minus_1 <= 1:
            return None, None
        
        # Transform psi -> log_psi
        log_psi = None
        if psi is not None:
            log_psi = torch.log(psi)
        
        # gamma is unchanged
        return log_psi, gamma

    def get_all_psi(
        self,
        log_psi: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Get all idiosyncratic variances psi_1, ..., psi_{K-1}.
        
        psi_1 = 0 is fixed.
        """
        device = log_psi.device if log_psi is not None else self.device
        dtype = log_psi.dtype if log_psi is not None else torch.float32
        
        if self.k_minus_1 == 1:
            # Only psi_1 = 0
            return torch.tensor([0.0], device=device, dtype=dtype)
        
        if log_psi is not None:
            psi_rest = torch.exp(log_psi)  # psi_2, ..., psi_{K-1}
            psi_1 = torch.tensor([0.0], device=device, dtype=dtype)
            return torch.cat([psi_1, psi_rest])
        else:
            return torch.zeros(self.k_minus_1, device=device, dtype=dtype)

    def get_all_gamma(
        self,
        gamma: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Get all factor loadings gamma_1, ..., gamma_{K-1}.
        
        gamma_1 = 1 is fixed.
        """
        device = gamma.device if gamma is not None else self.device
        dtype = gamma.dtype if gamma is not None else torch.float32
        
        if self.k_minus_1 == 1:
            # Only gamma_1 = 1
            return torch.tensor([1.0], device=device, dtype=dtype)
        
        if gamma is not None:
            gamma_1 = torch.tensor([1.0], device=device, dtype=dtype)
            return torch.cat([gamma_1, gamma])
        else:
            result = torch.zeros(self.k_minus_1, device=device, dtype=dtype)
            result[0] = 1.0
            return result
    
    @property
    def num_params(self) -> int:
        """Number of parameters on native scale."""
        return self.n_beta_params + self.n_log_psi_params + self.n_gamma_params
    
    @property
    def num_params_transformed(self) -> int:
        """Number of parameters on transformed scale (same count, different meaning)."""
        return self.num_params
    
    def compute_utilities(self, X: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return torch.einsum('nkp,p->nk', X, beta)
    
    def build_reduced_covariance_matrix(
        self,
        log_psi: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Build (K-1) x (K-1) covariance matrix Sigma = Psi + gamma @ gamma.T."""
        psi_all = self.get_all_psi(log_psi)
        gamma_all = self.get_all_gamma(gamma)
        
        Psi = torch.diag(psi_all)
        Sigma = Psi + torch.outer(gamma_all, gamma_all)
        
        return Sigma
    
    def build_reduced_covariance_matrix_transformed(
        self,
        psi: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Build covariance matrix from transformed parameters."""
        log_psi, gamma = self.transformed_to_native(psi, gamma)
        return self.build_reduced_covariance_matrix(log_psi, gamma)
    
    def build_full_covariance_matrix(
        self,
        log_psi: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Build full (K, K) covariance matrix with zeros for baseline alternative."""
        S_reduced = self.build_reduced_covariance_matrix(log_psi, gamma)
        return F.pad(S_reduced, (0, 1, 0, 1), mode='constant', value=0)
    
    def build_full_covariance_matrix_transformed(
        self,
        psi: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Build full covariance matrix from transformed parameters."""
        S_reduced = self.build_reduced_covariance_matrix_transformed(psi, gamma)
        return F.pad(S_reduced, (0, 1, 0, 1), mode='constant', value=0)
    
    def get_cholesky_factor_full(
        self,
        log_psi: Optional[torch.Tensor] = None,
        gamma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Cholesky factor of full covariance matrix."""
        S = self.build_full_covariance_matrix(log_psi, gamma)
        S_reduced = S[:self.k_minus_1, :self.k_minus_1]
        L_reduced = safe_cholesky(S_reduced)
        return F.pad(L_reduced, (0, 1, 0, 1), mode='constant', value=0)
    
    def get_param_names(self) -> List[str]:
        """Parameter names on native scale."""
        names = []
        for i in range(self.p):
            names.append(f'beta[{i}]')
        for i in range(self.n_log_psi_params):
            names.append(f'log_psi[{i + 2}]')
        for i in range(self.n_gamma_params):
            names.append(f'gamma[{i + 2}]')
        return names
    
    def get_param_names_transformed(self) -> List[str]:
        """Parameter names on transformed scale."""
        names = []
        for i in range(self.p):
            names.append(f'beta[{i}]')
        for i in range(self.n_log_psi_params):
            names.append(f'psi[{i + 2}]')
        for i in range(self.n_gamma_params):
            names.append(f'gamma[{i + 2}]')
        return names
    
    def get_param_types(self) -> List[str]:
        """Parameter types on native scale."""
        types = []
        for _ in range(self.n_beta_params):
            types.append('beta')
        for _ in range(self.n_log_psi_params):
            types.append('log_psi')
        for _ in range(self.n_gamma_params):
            types.append('gamma')
        return types
    
    def get_param_types_transformed(self) -> List[str]:
        """Parameter types on transformed scale."""
        types = []
        for _ in range(self.n_beta_params):
            types.append('beta')
        for _ in range(self.n_log_psi_params):
            types.append('psi')
        for _ in range(self.n_gamma_params):
            types.append('gamma')
        return types


# Type alias for parameter specs
AnyParameterSpec = Union[CholeskyParameterSpec, CovarianceParameterSpec, FactorParameterSpec]

# =============================================================================
# Negative Log-Likelihood Classes
# =============================================================================


class NegativeLogLikelihood(ABC):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        param_spec: AnyParameterSpec,
        log_L_diag_lower_bound: float = DEFAULT_LOG_L_DIAG_LOWER_BOUND,
        barrier_strength: float = DEFAULT_BARRIER_STRENGTH,
        cov_prior_scale: Optional[torch.Tensor] = None,
        cov_prior_df: Optional[float] = None,
        cov_prior_strength: float = DEFAULT_COV_PRIOR_STRENGTH,
        log_psi_lower_bound: float = DEFAULT_LOG_PSI_LOWER_BOUND,
        factor_barrier_strength: float = DEFAULT_FACTOR_BARRIER_STRENGTH,
    ):
        self.X = X
        self.y = y
        self.param_spec = param_spec
        self.n = X.shape[0]
        self.k = param_spec.k
        self.n_idx = torch.arange(self.n, device=X.device)
        self.device = X.device
        self.log_L_diag_lower_bound = log_L_diag_lower_bound
        self.barrier_strength = barrier_strength
        self.cov_prior_strength = cov_prior_strength
        self.log_psi_lower_bound = log_psi_lower_bound
        self.factor_barrier_strength = factor_barrier_strength
        
        k_minus_1 = self.k - 1
        if cov_prior_scale is None:
            self.cov_prior_scale = torch.eye(k_minus_1, device=self.device)
        else:
            self.cov_prior_scale = cov_prior_scale
        
        if cov_prior_df is None:
            self.cov_prior_df = float(self.k + 10)
        else:
            self.cov_prior_df = cov_prior_df
        
        self._is_factor_model = isinstance(param_spec, FactorParameterSpec)
        
        if self._is_factor_model:
            self._setup_factor_prior()
        else:
            self._setup_cov_prior()
            self.cov_param_spec = CovarianceParameterSpec.from_cholesky_spec(param_spec)
            self.L_lower_indices = build_lower_triangular_indices(self.k - 1)
    
    def _setup_cov_prior(self):
        """Precompute constants for the normalized Wishart prior (dense model)."""
        p = self.k - 1
        n = self.cov_prior_df
        Sigma = self.cov_prior_scale
        
        self.prior_p = p
        self.prior_n = n
        self.prior_Sigma_inv = torch.linalg.inv(Sigma)
        
        self.prior_log_const = (
            gammaln(n * p / 2)
            - (n * p / 2) * np.log(2)
            - (n / 2) * torch.linalg.slogdet(Sigma)[1].item()
            - multigammaln(n / 2, p)
        )
    
    def _setup_factor_prior(self):
        """Setup prior constants for anchored factor model.
        
        Priors:
        - log_psi ~ N(0, log_psi_std^2)
        - gamma ~ N(0, gamma_std^2)
        """
        self.factor_log_psi_std = DEFAULT_FACTOR_LOG_PSI_STD
        self.factor_gamma_std = DEFAULT_FACTOR_GAMMA_STD
    
    def _compute_factor_prior_penalty(
        self,
        log_psi: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute negative log prior for factor model parameters on native scale.
        
        For log_psi: Gaussian prior N(0, log_psi_std^2)
        For gamma: Gaussian prior N(0, gamma_std^2)
        """
        device = self.device
        dtype = torch.float32
        
        if log_psi is not None:
            device = log_psi.device
            dtype = log_psi.dtype
        elif gamma is not None:
            device = gamma.device
            dtype = gamma.dtype
        
        penalty = torch.tensor(0.0, device=device, dtype=dtype)
        
        # log_psi prior: Gaussian
        if log_psi is not None:
            log_psi_var = self.factor_log_psi_std ** 2
            penalty = penalty + 0.5 * (log_psi ** 2).sum() / log_psi_var
        
        # gamma prior: Gaussian
        if gamma is not None:
            gamma_var = self.factor_gamma_std ** 2
            penalty = penalty + 0.5 * (gamma ** 2).sum() / gamma_var
        
        return self.cov_prior_strength * penalty

    def _compute_factor_prior_penalty_transformed(
        self,
        psi: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute negative log prior for factor model parameters on transformed scale.
        
        For psi: Convert to log_psi, apply Gaussian prior, add Jacobian correction
        For gamma: Apply Gaussian prior directly (same as native)
        """
        device = self.device
        dtype = torch.float32
        
        if psi is not None:
            device = psi.device
            dtype = psi.dtype
        elif gamma is not None:
            device = gamma.device
            dtype = gamma.dtype
        
        penalty = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Psi prior: convert to log_psi, apply Gaussian, add Jacobian
        if psi is not None:
            log_psi = torch.log(psi)
            
            # Gaussian prior on log_psi
            log_psi_var = self.factor_log_psi_std ** 2
            penalty_log_psi = 0.5 * (log_psi ** 2).sum() / log_psi_var
            
            # Jacobian correction for psi -> log_psi transformation
            # p(psi) = p(log_psi) * |d(log_psi)/d(psi)| = p(log_psi) / psi
            # -log p(psi) = -log p(log_psi) + log(psi) = -log p(log_psi) + log_psi
            # So we add log_psi.sum() to the penalty
            jacobian_correction = log_psi.sum()
            
            penalty = penalty + penalty_log_psi + jacobian_correction
        
        # Gamma prior: Gaussian (same as native, no transformation)
        if gamma is not None:
            gamma_var = self.factor_gamma_std ** 2
            penalty = penalty + 0.5 * (gamma ** 2).sum() / gamma_var
        
        return self.cov_prior_strength * penalty

    def _compute_factor_barrier_penalty(
        self,
        log_psi: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute barrier penalty to keep log_psi above lower bound.
        
        Uses log barrier: -strength * log(log_psi - lower)
        
        Note: gamma is unconstrained, so no barrier is applied.
        """
        device = self.device
        dtype = torch.float32
        
        if log_psi is not None:
            device = log_psi.device
            dtype = log_psi.dtype
        
        penalty = torch.tensor(0.0, device=device, dtype=dtype)
        lower = torch.tensor(self.log_psi_lower_bound, device=device, dtype=dtype)
        
        # Barrier on log_psi
        if log_psi is not None and log_psi.numel() > 0:
            log_psi_lower_dist = log_psi - lower
            log_psi_penalty = -torch.log(log_psi_lower_dist).mean()
            penalty = penalty + log_psi_penalty
        
        return self.factor_barrier_strength * penalty

    def _compute_cov_prior_penalty(
        self, 
        log_L_diag: Optional[torch.Tensor], 
        L_lower: Optional[torch.Tensor],
        apply_jacobian: bool = False
    ) -> torch.Tensor:
        """
        Compute negative log prior for the normalized Wishart distribution (dense model).
        """
        if self.prior_p <= 1:
            return torch.tensor(0.0, device=self.device)
        
        Sigma_current = self.param_spec.build_reduced_covariance_matrix(log_L_diag, L_lower)
        V = Sigma_current / Sigma_current[0, 0]
        
        p = self.prior_p
        n = self.prior_n
        
        log_det_V = torch.linalg.slogdet(V)[1]
        trace_term = torch.trace(self.prior_Sigma_inv @ V)
        
        log_prior = (
            self.prior_log_const
            + ((n - p - 1) / 2) * log_det_V
            - (n * p / 2) * torch.log(0.5 * trace_term)
        )
        
        result = -self.cov_prior_strength * log_prior
        
        if apply_jacobian:
            log_det_J = compute_log_det_jacobian_cholesky_to_cov(
                log_L_diag, L_lower, self.k - 1, self.L_lower_indices
            )
            result = result + self.cov_prior_strength * log_det_J
        
        return result
    
    def _compute_barrier_penalty(self, log_diag_params: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute barrier penalty for log diagonal elements (dense models only)."""
        if log_diag_params is None:
            return torch.tensor(0.0, device=self.device)
        
        shifted = log_diag_params - self.log_L_diag_lower_bound
        return -self.barrier_strength * torch.log(shifted).mean()
    
    def _compute_total_penalty(
        self,
        params_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute total penalty on native parameterization."""
        if self._is_factor_model:
            log_psi = params_dict.get('log_psi')
            gamma = params_dict.get('gamma')
            prior_penalty = self._compute_factor_prior_penalty(log_psi, gamma)
            barrier_penalty = self._compute_factor_barrier_penalty(log_psi)
            return prior_penalty + barrier_penalty
        else:
            log_L_diag = params_dict.get('log_L_diag')
            L_lower = params_dict.get('L_lower')
            barrier = self._compute_barrier_penalty(log_L_diag)
            cov_prior = self._compute_cov_prior_penalty(log_L_diag, L_lower, apply_jacobian=False)
            return barrier + cov_prior
        
    def _compute_total_penalty_transformed(
        self,
        params_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute total penalty on transformed scale with Jacobian corrections."""
        if self._is_factor_model:
            psi = params_dict.get('psi')
            gamma = params_dict.get('gamma')
            
            # Prior penalty with Jacobian correction
            prior_penalty = self._compute_factor_prior_penalty_transformed(psi, gamma)
            
            # Barrier penalty on native scale
            log_psi = torch.log(psi) if psi is not None else None
            barrier_penalty = self._compute_factor_barrier_penalty(log_psi)
            
            return prior_penalty + barrier_penalty
        else:
            variances = params_dict.get('variances')
            covariances = params_dict.get('covariances')
            log_L_diag, L_lower = self.cov_param_spec.to_cholesky_params(variances, covariances)
            barrier = self._compute_barrier_penalty(log_L_diag)
            cov_prior = self._compute_cov_prior_penalty(log_L_diag, L_lower, apply_jacobian=True)
            return barrier + cov_prior
      
    def _unpack_params(self, params: torch.Tensor) -> Dict[str, Any]:
        return self.param_spec.unflatten(params)
    
    def _unpack_params_transformed(self, params: torch.Tensor) -> Dict[str, Any]:
        if self._is_factor_model:
            return self.param_spec.unflatten_transformed(params)
        else:
            raise ValueError("Transformed params only supported for factor models")
    
    def _compute_utilities(self, beta: torch.Tensor) -> torch.Tensor:
        return self.param_spec.compute_utilities(self.X, beta)
    
    def _get_covariance_matrix_from_params(
        self, 
        params: torch.Tensor, 
        use_cov_param: bool = False,
        use_transformed: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract beta and build full covariance matrix from parameters."""
        if self._is_factor_model:
            if use_transformed:
                params_dict = self.param_spec.unflatten_transformed(params)
                beta = params_dict['beta']
                psi = params_dict['psi']
                gamma = params_dict['gamma']
                S = self.param_spec.build_full_covariance_matrix_transformed(psi, gamma)
            else:
                params_dict = self.param_spec.unflatten(params)
                beta = params_dict['beta']
                log_psi = params_dict['log_psi']
                gamma = params_dict['gamma']
                S = self.param_spec.build_full_covariance_matrix(log_psi, gamma)
            return beta, S
        else:
            if use_cov_param:
                params_dict = self.cov_param_spec.unflatten(params)
                beta = params_dict['beta']
                variances = params_dict['variances']
                covariances = params_dict['covariances']
                S = self.cov_param_spec.build_full_covariance_matrix(variances, covariances)
            else:
                params_dict = self.param_spec.unflatten(params)
                beta = params_dict['beta']
                log_L_diag = params_dict['log_L_diag']
                L_lower = params_dict['L_lower']
                variances, covariances = self.param_spec.to_covariance_params(log_L_diag, L_lower)
                S = self.cov_param_spec.build_full_covariance_matrix(variances, covariances)
            return beta, S
                    
    def _get_cholesky_params_from_params(
        self,
        params: torch.Tensor,
        use_cov_param: bool = False,
        use_transformed: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract beta and Cholesky parameters from params."""
        if self._is_factor_model:
            if use_transformed:
                params_dict = self.param_spec.unflatten_transformed(params)
                beta = params_dict['beta']
                psi = params_dict['psi']
                gamma = params_dict['gamma']
                S_reduced = self.param_spec.build_reduced_covariance_matrix_transformed(psi, gamma)
            else:
                params_dict = self.param_spec.unflatten(params)
                beta = params_dict['beta']
                log_psi = params_dict['log_psi']
                gamma = params_dict['gamma']
                S_reduced = self.param_spec.build_reduced_covariance_matrix(log_psi, gamma)
            
            L = safe_cholesky(S_reduced)
            
            if torch.isnan(L).any():
                return beta, None, None
            
            k_minus_1 = self.k - 1
            n_L_diag = max(0, k_minus_1 - 1)
            n_L_lower = (k_minus_1 - 1) * k_minus_1 // 2 if k_minus_1 > 1 else 0
            
            log_L_diag = None
            if n_L_diag > 0:
                log_L_diag = torch.log(torch.diag(L)[1:])
            
            L_lower = None
            if n_L_lower > 0:
                L_lower_indices = build_lower_triangular_indices(k_minus_1)
                L_lower = torch.zeros(n_L_lower, device=L.device, dtype=L.dtype)
                for idx, (i, j) in enumerate(L_lower_indices):
                    L_lower[idx] = L[i, j]
            
            return beta, log_L_diag, L_lower
        else:
            if use_cov_param:
                params_dict = self.cov_param_spec.unflatten(params)
                beta = params_dict['beta']
                variances = params_dict['variances']
                covariances = params_dict['covariances']
                log_L_diag, L_lower = self.cov_param_spec.to_cholesky_params(variances, covariances)
            else:
                params_dict = self.param_spec.unflatten(params)
                beta = params_dict['beta']
                log_L_diag = params_dict['log_L_diag']
                L_lower = params_dict['L_lower']
            
            return beta, log_L_diag, L_lower
    
    @abstractmethod
    def _compute_observation_nlls(self, params: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _compute_observation_nlls_cov_param(self, params: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def _compute_observation_nlls_transformed(self, params: torch.Tensor) -> torch.Tensor:
        """Compute per-observation NLLs using transformed parameterization."""
        pass
    
    @abstractmethod
    def compute_observation_nlls_for_batch(
        self,
        params: torch.Tensor,
        start_idx: int,
        end_idx: int,
        use_cov_param: bool = False,
        use_transformed: bool = False
    ) -> torch.Tensor:
        pass
    
    def _compute_mean_nll(self, params: torch.Tensor) -> torch.Tensor:
        return self._compute_observation_nlls(params).mean()
    
    def __call__(self, params: torch.Tensor) -> torch.Tensor:
        params_dict = self._unpack_params(params)
        mean_nll = self._compute_mean_nll(params)
        penalty = self._compute_total_penalty(params_dict)
        return mean_nll + penalty / self.n
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class EmulatorNegativeLogLikelihood(NegativeLogLikelihood):
    """NLL using the neural network emulator with smooth canonicalization."""
    
    def __init__(
        self, 
        X: torch.Tensor,
        y: torch.Tensor,
        model: nn.Module,
        param_spec: AnyParameterSpec,
        log_L_diag_lower_bound: float = DEFAULT_LOG_L_DIAG_LOWER_BOUND,
        barrier_strength: float = DEFAULT_BARRIER_STRENGTH,
        cov_prior_scale: Optional[torch.Tensor] = None,
        cov_prior_df: Optional[float] = None,
        cov_prior_strength: float = DEFAULT_COV_PRIOR_STRENGTH,
        log_psi_lower_bound: float = DEFAULT_LOG_PSI_LOWER_BOUND,
        factor_barrier_strength: float = DEFAULT_FACTOR_BARRIER_STRENGTH,
    ):
        super().__init__(
            X, y, param_spec, log_L_diag_lower_bound, barrier_strength,
            cov_prior_scale, cov_prior_df, cov_prior_strength,
            log_psi_lower_bound, factor_barrier_strength
        )
        self.model = model
    
    @property
    def name(self) -> str:
        return "Emulator"
    
    def _compute_emulator_log_probs(
        self, 
        V: torch.Tensor, 
        S: torch.Tensor, 
        n_obs: int
    ) -> torch.Tensor:
        S_expanded = S.unsqueeze(0).expand(n_obs, -1, -1)
        V_star, S_star, C_star = canonicalize(V, S_expanded)
        logits = self.model(V_star, S_star, C_star)
        return F.log_softmax(logits, dim=-1)
    
    def _compute_observation_nlls(self, params: torch.Tensor) -> torch.Tensor:
        params_dict = self._unpack_params(params)
        beta = params_dict['beta']
        
        if self._is_factor_model:
            log_psi = params_dict['log_psi']
            gamma = params_dict['gamma']
            S = self.param_spec.build_full_covariance_matrix(log_psi, gamma)
        else:
            log_L_diag = params_dict['log_L_diag']
            L_lower = params_dict['L_lower']
            variances, covariances = self.param_spec.to_covariance_params(log_L_diag, L_lower)
            S = self.cov_param_spec.build_full_covariance_matrix(variances, covariances)
        
        V = self._compute_utilities(beta)
        log_probs = self._compute_emulator_log_probs(V, S, self.n)
        log_probs_chosen = log_probs[self.n_idx, self.y]
        
        return -log_probs_chosen
    
    def _compute_observation_nlls_cov_param(self, params: torch.Tensor) -> torch.Tensor:
        if self._is_factor_model:
            return self._compute_observation_nlls(params)
        
        params_dict = self.cov_param_spec.unflatten(params)
        beta = params_dict['beta']
        variances = params_dict['variances']
        covariances = params_dict['covariances']
        
        V = self._compute_utilities(beta)
        S = self.cov_param_spec.build_full_covariance_matrix(variances, covariances)
        
        log_probs = self._compute_emulator_log_probs(V, S, self.n)
        log_probs_chosen = log_probs[self.n_idx, self.y]
        
        return -log_probs_chosen
    
    def _compute_observation_nlls_transformed(self, params: torch.Tensor) -> torch.Tensor:
        if self._is_factor_model:
            params_dict = self.param_spec.unflatten_transformed(params)
            beta = params_dict['beta']
            psi = params_dict['psi']
            gamma = params_dict['gamma']
            S = self.param_spec.build_full_covariance_matrix_transformed(psi, gamma)
        else:
            raise ValueError("Transformed NLLs for dense models use _compute_observation_nlls_cov_param")
        
        V = self._compute_utilities(beta)
        log_probs = self._compute_emulator_log_probs(V, S, self.n)
        log_probs_chosen = log_probs[self.n_idx, self.y]
        
        return -log_probs_chosen
    
    def compute_observation_nlls_for_batch(
        self,
        params: torch.Tensor,
        start_idx: int,
        end_idx: int,
        use_cov_param: bool = False,
        use_transformed: bool = False
    ) -> torch.Tensor:
        batch_n = end_idx - start_idx
        X_batch = self.X[start_idx:end_idx]
        y_batch = self.y[start_idx:end_idx]
        batch_n_idx = torch.arange(batch_n, device=self.device)
        
        beta, S = self._get_covariance_matrix_from_params(params, use_cov_param, use_transformed)
        
        V = torch.einsum('nkp,p->nk', X_batch, beta)
        log_probs = self._compute_emulator_log_probs(V, S, batch_n)
        log_probs_chosen = log_probs[batch_n_idx, y_batch]
        
        return -log_probs_chosen


class GHKNegativeLogLikelihood(NegativeLogLikelihood):
    """GHK simulator for multinomial probit likelihood."""
    
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        param_spec: AnyParameterSpec,
        ghk_config: GHKConfig,
        log_L_diag_lower_bound: float = DEFAULT_LOG_L_DIAG_LOWER_BOUND,
        barrier_strength: float = DEFAULT_BARRIER_STRENGTH,
        cov_prior_scale: Optional[torch.Tensor] = None,
        cov_prior_df: Optional[float] = None,
        cov_prior_strength: float = DEFAULT_COV_PRIOR_STRENGTH,
        log_psi_lower_bound: float = DEFAULT_LOG_PSI_LOWER_BOUND,
        factor_barrier_strength: float = DEFAULT_FACTOR_BARRIER_STRENGTH,
    ):
        super().__init__(
            X, y, param_spec, log_L_diag_lower_bound, barrier_strength,
            cov_prior_scale, cov_prior_df, cov_prior_strength,
            log_psi_lower_bound, factor_barrier_strength
        )
        self.num_draws = ghk_config.num_draws
        self.ghk_seed = ghk_config.seed
        self.k_minus_1 = self.k - 1
        
        rng_state = torch.get_rng_state()
        if self.ghk_seed is not None:
            torch.manual_seed(self.ghk_seed)
        self.uniforms = torch.rand(self.n, self.num_draws, self.k_minus_1, device=self.device)
        torch.set_rng_state(rng_state)
        
        self.choice_indices = {
            j: torch.where(self.y == j)[0] for j in range(self.k)
        }
        
        self._precompute_A_matrices()
    
    def _precompute_A_matrices(self):
        k = self.k
        k_minus_1 = self.k_minus_1
        
        A_list = []
        for chosen in range(k_minus_1):
            A = torch.zeros(k_minus_1, k_minus_1, device=self.device)
            row = 0
            for m in range(k):
                if m == chosen:
                    continue
                if m < k_minus_1:
                    A[row, chosen] = 1.0
                    A[row, m] = -1.0
                else:
                    A[row, chosen] = 1.0
                row += 1
            A_list.append(A)
        
        self.A_stacked = torch.stack(A_list, dim=0)
    
    def _compute_all_L_eta(self, L: torch.Tensor) -> torch.Tensor:
        AL_all = torch.matmul(self.A_stacked, L)
        Q, R = torch.linalg.qr(AL_all.transpose(-2, -1), mode='reduced')
        signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        L_eta_all = R.transpose(-2, -1) * signs.unsqueeze(-2)
        return L_eta_all
    
    @property
    def name(self) -> str:
        return f"GHK({self.num_draws})"
    
    def _build_cholesky_factor(
        self,
        log_L_diag: Optional[torch.Tensor],
        L_lower: Optional[torch.Tensor]
    ) -> torch.Tensor:
        k_minus_1 = self.k_minus_1
        device = self.device
        dtype = torch.float32
        
        if log_L_diag is not None:
            device = log_L_diag.device
            dtype = log_L_diag.dtype
        elif L_lower is not None:
            device = L_lower.device
            dtype = L_lower.dtype
        
        L = torch.zeros(k_minus_1, k_minus_1, device=device, dtype=dtype)
        L[0, 0] = 1.0
        
        for i in range(1, k_minus_1):
            if log_L_diag is not None:
                L[i, i] = torch.exp(log_L_diag[i-1])
            else:
                L[i, i] = 1.0
        
        if L_lower is not None:
            L_lower_indices = build_lower_triangular_indices(k_minus_1)
            for idx, (i, j) in enumerate(L_lower_indices):
                L[i, j] = L_lower[idx]
        
        return L
    
    def _get_cholesky_from_factor_params(
        self,
        log_psi: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Get Cholesky factor from factor model parameters."""
        S_reduced = self.param_spec.build_reduced_covariance_matrix(log_psi, gamma)
        L = safe_cholesky(S_reduced)
        return L
    
    def _get_cholesky_from_factor_params_transformed(
        self,
        psi: Optional[torch.Tensor],
        gamma: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Get Cholesky factor from transformed factor model parameters."""
        S_reduced = self.param_spec.build_reduced_covariance_matrix_transformed(psi, gamma)
        L = safe_cholesky(S_reduced)
        return L
    
    def _compute_observation_nlls(self, params: torch.Tensor) -> torch.Tensor:
        params_dict = self._unpack_params(params)
        beta = params_dict['beta']
        
        if self._is_factor_model:
            log_psi = params_dict['log_psi']
            gamma = params_dict['gamma']
            L = self._get_cholesky_from_factor_params(log_psi, gamma)
            
            if torch.isnan(L).any():
                return torch.full((self.n,), float('nan'), device=self.device)
        else:
            log_L_diag = params_dict['log_L_diag']
            L_lower = params_dict['L_lower']
            L = self._build_cholesky_factor(log_L_diag, L_lower)
        
        V = self._compute_utilities(beta)
        log_probs = self._compute_all_log_probs(V, L)
        
        return -log_probs
    
    def _compute_observation_nlls_transformed(self, params: torch.Tensor) -> torch.Tensor:
        if self._is_factor_model:
            params_dict = self.param_spec.unflatten_transformed(params)
            beta = params_dict['beta']
            psi = params_dict['psi']
            gamma = params_dict['gamma']
            L = self._get_cholesky_from_factor_params_transformed(psi, gamma)
            
            if torch.isnan(L).any():
                return torch.full((self.n,), float('nan'), device=self.device)
        else:
            raise ValueError("Transformed NLLs for dense models use _compute_observation_nlls_cov_param")
        
        V = self._compute_utilities(beta)
        log_probs = self._compute_all_log_probs(V, L)
        
        return -log_probs
    
    def _compute_observation_nlls_cov_param(self, params: torch.Tensor) -> torch.Tensor:
        if self._is_factor_model:
            return self._compute_observation_nlls(params)
        
        params_dict = self.cov_param_spec.unflatten(params)
        beta = params_dict['beta']
        variances = params_dict['variances']
        covariances = params_dict['covariances']
        
        log_L_diag, L_lower = self.cov_param_spec.to_cholesky_params(variances, covariances)
        
        if log_L_diag is not None and torch.isnan(log_L_diag).any():
            return torch.full((self.n,), float('nan'), device=self.device)
        if L_lower is not None and torch.isnan(L_lower).any():
            return torch.full((self.n,), float('nan'), device=self.device)
        
        L = self._build_cholesky_factor(log_L_diag, L_lower)
        V = self._compute_utilities(beta)
        log_probs = self._compute_all_log_probs(V, L)
        
        return -log_probs
    
    def compute_observation_nlls_for_batch(
        self,
        params: torch.Tensor,
        start_idx: int,
        end_idx: int,
        use_cov_param: bool = False,
        use_transformed: bool = False
    ) -> torch.Tensor:
        batch_n = end_idx - start_idx
        X_batch = self.X[start_idx:end_idx]
        y_batch = self.y[start_idx:end_idx]
        uniforms_batch = self.uniforms[start_idx:end_idx]
        
        beta, log_L_diag, L_lower = self._get_cholesky_params_from_params(
            params, use_cov_param, use_transformed
        )
        
        if log_L_diag is not None and torch.isnan(log_L_diag).any():
            return torch.full((batch_n,), float('nan'), device=self.device)
        if L_lower is not None and torch.isnan(L_lower).any():
            return torch.full((batch_n,), float('nan'), device=self.device)
        
        V = torch.einsum('nkp,p->nk', X_batch, beta)
        L = self._build_cholesky_factor(log_L_diag, L_lower)
        
        log_probs = self._compute_batch_log_probs(V, L, y_batch, uniforms_batch)
        
        return -log_probs

    def _compute_batch_log_probs(
        self,
        V: torch.Tensor,
        L: torch.Tensor,
        y_batch: torch.Tensor,
        uniforms_batch: torch.Tensor
    ) -> torch.Tensor:
        k = self.k
        k_minus_1 = self.k_minus_1
        batch_n = V.shape[0]
        
        L_eta_all = self._compute_all_L_eta(L)
        
        log_probs = torch.zeros(batch_n, device=self.device)
        
        batch_choice_indices = {
            j: torch.where(y_batch == j)[0] for j in range(k)
        }
        
        ref_indices = batch_choice_indices[k - 1]
        if len(ref_indices) > 0:
            V_ref = V[ref_indices]
            uniforms_ref = uniforms_batch[ref_indices]
            upper_bounds = -V_ref[:, :k_minus_1]
            log_p = self._ghk_rectangle_prob(upper_bounds, L, uniforms_ref)
            log_probs = log_probs.scatter(0, ref_indices, log_p)
        
        for j in range(k_minus_1):
            indices = batch_choice_indices[j]
            if len(indices) == 0:
                continue
            
            V_subset = V[indices]
            uniforms_subset = uniforms_batch[indices]
            upper_bounds = self._compute_upper_bounds(V_subset, j)
            
            log_p = self._ghk_rectangle_prob(upper_bounds, L_eta_all[j], uniforms_subset)
            log_probs = log_probs.scatter(0, indices, log_p)
        
        return log_probs
    
    def _compute_all_log_probs(self, V: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        k = self.k
        k_minus_1 = self.k_minus_1
        
        L_eta_all = self._compute_all_L_eta(L)
        
        log_probs = torch.zeros(self.n, device=self.device)
        
        ref_indices = self.choice_indices[k - 1]
        if len(ref_indices) > 0:
            V_ref = V[ref_indices]
            uniforms_ref = self.uniforms[ref_indices]
            upper_bounds = -V_ref[:, :k_minus_1]
            log_p = self._ghk_rectangle_prob(upper_bounds, L, uniforms_ref)
            log_probs = log_probs.scatter(0, ref_indices, log_p)
        
        for j in range(k_minus_1):
            indices = self.choice_indices[j]
            if len(indices) == 0:
                continue
            
            V_subset = V[indices]
            uniforms_subset = self.uniforms[indices]
            upper_bounds = self._compute_upper_bounds(V_subset, j)
            
            log_p = self._ghk_rectangle_prob(upper_bounds, L_eta_all[j], uniforms_subset)
            log_probs = log_probs.scatter(0, indices, log_p)
        
        return log_probs
    
    def _compute_upper_bounds(self, V_subset: torch.Tensor, chosen: int) -> torch.Tensor:
        n_j = V_subset.shape[0]
        k = self.k
        k_minus_1 = self.k_minus_1
        
        upper_bounds = torch.zeros(n_j, k_minus_1, device=self.device)
        
        row = 0
        for m in range(k):
            if m == chosen:
                continue
            if m < k_minus_1:
                upper_bounds[:, row] = V_subset[:, chosen] - V_subset[:, m]
            else:
                upper_bounds[:, row] = V_subset[:, chosen]
            row += 1
        
        return upper_bounds
    
    def _ghk_rectangle_prob(
        self,
        upper_bounds: torch.Tensor,
        L: torch.Tensor,
        uniforms: torch.Tensor
    ) -> torch.Tensor:
        n_j = upper_bounds.shape[0]
        k_minus_1 = upper_bounds.shape[1]
        R = uniforms.shape[1]
        
        log_prob_contributions = []
        z_samples_list = []
        
        for m in range(k_minus_1):
            if m == 0:
                contrib_prev = torch.zeros(n_j, R, device=self.device)
            else:
                z_prev = torch.stack(z_samples_list, dim=2)
                L_row = L[m, :m]
                contrib_prev = (z_prev * L_row).sum(dim=2)
            
            upper_m = (upper_bounds[:, m:m+1] - contrib_prev) / L[m, m]
            
            log_Phi_upper = torch.special.log_ndtr(upper_m)
            log_Phi_upper = torch.clamp(log_Phi_upper, min=LOG_PROB_CLAMP_MIN)
            log_prob_contributions.append(log_Phi_upper)
            
            u_m = uniforms[:, :, m]
            Phi_upper = torch.special.ndtr(upper_m)
            Phi_upper_clamped = torch.clamp(Phi_upper, min=1e-10, max=1.0 - 1e-10)
            arg = torch.clamp(u_m * Phi_upper_clamped, min=1e-10, max=1.0 - 1e-10)
            z_m = torch.special.ndtri(arg)
            z_m = torch.clamp(z_m, min=Z_CLAMP_MIN, max=Z_CLAMP_MAX)
            
            z_samples_list.append(z_m)
        
        log_prob_draws = torch.stack(log_prob_contributions, dim=2).sum(dim=2)
        log_prob = torch.logsumexp(log_prob_draws, dim=1) - np.log(R)
        
        return log_prob
    

class GumbelEmulatorNegativeLogLikelihood(EmulatorNegativeLogLikelihood):
    """
    Emulator NLL for correlated-Gumbel errors (dense covariance).

    Every parent code path (optimizer, covariance-scale and batched SE computations) calls
    _compute_emulator_log_probs with the probit-style S = blockdiag(Omega, 0). This override
    turns it into blockdiag(Omega, 1), moves the outside option to index 0 (the emulator's
    base position), and returns log-probabilities in the original alternative order.
    """
    
    def __init__(
        self, 
        X: torch.Tensor,
        y: torch.Tensor,
        model: nn.Module,
        param_spec: AnyParameterSpec,
        log_L_diag_lower_bound: float = DEFAULT_LOG_L_DIAG_LOWER_BOUND,
        barrier_strength: float = DEFAULT_BARRIER_STRENGTH,
        cov_prior_scale: Optional[torch.Tensor] = None,
        cov_prior_df: Optional[float] = None,
        cov_prior_strength: float = DEFAULT_COV_PRIOR_STRENGTH,
        log_psi_lower_bound: float = DEFAULT_LOG_PSI_LOWER_BOUND,
        factor_barrier_strength: float = DEFAULT_FACTOR_BARRIER_STRENGTH,
    ):
        super().__init__(
            X, y, model, param_spec, log_L_diag_lower_bound, barrier_strength,
            cov_prior_scale, cov_prior_df, cov_prior_strength,
            log_psi_lower_bound, factor_barrier_strength
        )
        if self._is_factor_model:
            raise NotImplementedError("Correlated-Gumbel estimation supports the dense covariance structure only")
        if not getattr(model, 'include_c_star', False):
            raise ValueError("Gumbel estimation requires a Gumbel emulator (include_c_star=True)")
        self.perm, self.inv_perm = outside_first_permutation(self.k, self.device)
    
    def _compute_emulator_log_probs(
        self, 
        V: torch.Tensor, 
        S: torch.Tensor, 
        n_obs: int
    ) -> torch.Tensor:
        S_emu = gumbel_full_scale_matrix(S).index_select(-2, self.perm).index_select(-1, self.perm)
        V_emu = V.index_select(-1, self.perm)
        log_probs_emu = super()._compute_emulator_log_probs(V_emu, S_emu, n_obs)
        return log_probs_emu.index_select(-1, self.inv_perm)


class SmoothedSimulatorNegativeLogLikelihood(NegativeLogLikelihood):
    """
    Softmax-smoothed sample-average NLL for correlated-Gumbel errors (dense covariance).

    For observation i with choice y_i:

        log P~_i = logsumexp_r [ log_softmax(tau * U_ir)_{y_i} ] - log R,
        U_ir = V_i + Sigma^{1/2} z_ir,   Sigma = blockdiag(Omega, 1),

    with the symmetric root and z_ir ~ iid Gumbel(-gamma, 1), following
    training._compute_gradients. tau acts on the level scale, where the outside option's
    error has unit Gumbel scale.

    Common random numbers: the draws are pregenerated once per fit (seeded, with the global
    RNG state restored afterwards, as in GHKNegativeLogLikelihood) and observation i always
    uses draw set i mod pool_rows. With the default 8 GB budget every observation in the
    study gets its own draws. Pregeneration is also needed because torch.func.jacfwd (standard
    errors) runs under vmap, which does not allow random operations.

    Memory: every evaluation loops over row chunks. On the optimizer path, chunks are
    checkpointed (recomputed during backward) when the memory plan requires it. The batched
    path used for standard errors runs in forward mode, which keeps no graph, and uses
    smaller chunks to leave room for one tangent per parameter.
    """
    
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        param_spec: AnyParameterSpec,
        sim_config: SmoothedSimConfig,
        log_L_diag_lower_bound: float = DEFAULT_LOG_L_DIAG_LOWER_BOUND,
        barrier_strength: float = DEFAULT_BARRIER_STRENGTH,
        cov_prior_scale: Optional[torch.Tensor] = None,
        cov_prior_df: Optional[float] = None,
        cov_prior_strength: float = DEFAULT_COV_PRIOR_STRENGTH,
        log_psi_lower_bound: float = DEFAULT_LOG_PSI_LOWER_BOUND,
        factor_barrier_strength: float = DEFAULT_FACTOR_BARRIER_STRENGTH,
    ):
        super().__init__(
            X, y, param_spec, log_L_diag_lower_bound, barrier_strength,
            cov_prior_scale, cov_prior_df, cov_prior_strength,
            log_psi_lower_bound, factor_barrier_strength
        )
        if self._is_factor_model:
            raise NotImplementedError("The smoothed simulator supports the dense covariance structure only")
        if int(sim_config.num_samples) < 1:
            raise ValueError("num_samples must be at least 1")
        if not float(sim_config.tau) > 0:
            raise ValueError("tau must be positive")
        
        self.tau = float(sim_config.tau)
        self.num_samples = int(sim_config.num_samples)
        self.log_num_samples = math.log(self.num_samples)
        self.sim_seed = sim_config.seed
        
        self.memory_plan = plan_smoothed_memory(self.n, self.k, sim_config)
        self.pool_rows = self.memory_plan.pool_rows
        self.use_checkpoint = self.memory_plan.use_checkpoint
        
        per_row = self.num_samples * self.k
        chunk_elements = max(1, int(sim_config.chunk_elements))
        self.rows_per_chunk = max(1, chunk_elements // per_row)
        n_params = max(self.param_spec.num_params, self.cov_param_spec.num_params)
        self.rows_per_chunk_batch = max(1, chunk_elements // (per_row * (n_params + 1)))
        
        rng_state = torch.get_rng_state()
        try:
            if self.sim_seed is not None:
                torch.manual_seed(self.sim_seed)
            self.pool = generate_gumbel_pool(
                self.pool_rows, self.num_samples, self.k, self.device, dtype=X.dtype
            )
        finally:
            torch.set_rng_state(rng_state)
    
    @property
    def name(self) -> str:
        return smoothed_method_label(self.tau, self.num_samples)
    
    def _draws(self, start: int, end: int) -> torch.Tensor:
        """Draws for observations start..end-1: draw set i mod pool_rows for observation i."""
        pool_rows = self.pool_rows
        offset = start % pool_rows
        if offset + (end - start) <= pool_rows:
            return self.pool[offset:offset + (end - start)]
        idx = torch.arange(start, end, device=self.device) % pool_rows
        return self.pool[idx]
    
    def _chunk_log_probs(
        self,
        V_rows: torch.Tensor,
        root: torch.Tensor,
        start: int,
        end: int
    ) -> torch.Tensor:
        """Smoothed log-probability of the chosen alternative for observations start..end-1."""
        z = self._draws(start, end)  # (m, R, k)
        # training.simulate_data: errors = einsum('bij,bsj->bsi', sqrt_Sigma, z); U = v + errors
        U = torch.matmul(z, root.transpose(-1, -2))
        U = U + V_rows.unsqueeze(-2)
        # training._compute_gradients: log_softmax(tau * U), then logsumexp over draws - log(R)
        log_soft_choices = F.log_softmax(self.tau * U, dim=-1)
        chosen = self.y[start:end].view(-1, 1, 1).expand(-1, self.num_samples, 1)
        log_soft_chosen = log_soft_choices.gather(-1, chosen).squeeze(-1)  # (m, R)
        return torch.logsumexp(log_soft_chosen, dim=-1) - self.log_num_samples
    
    def _simulated_log_probs(
        self,
        beta: torch.Tensor,
        S: torch.Tensor,
        start: int,
        end: int,
        rows_per_chunk: int,
        use_checkpoint: bool
    ) -> torch.Tensor:
        root = gumbel_error_root(S)
        V = self.param_spec.compute_utilities(self.X[start:end], beta)
        checkpoint_chunks = (
            use_checkpoint and torch.is_grad_enabled()
            and (root.requires_grad or V.requires_grad)
        )
        
        pieces = []
        for chunk_start in range(start, end, rows_per_chunk):
            chunk_end = min(chunk_start + rows_per_chunk, end)
            V_rows = V[chunk_start - start:chunk_end - start]
            if checkpoint_chunks:
                piece = torch_checkpoint(
                    self._chunk_log_probs, V_rows, root, chunk_start, chunk_end,
                    use_reentrant=False, preserve_rng_state=False
                )
            else:
                piece = self._chunk_log_probs(V_rows, root, chunk_start, chunk_end)
            pieces.append(piece)
        return torch.cat(pieces)
    
    def _compute_observation_nlls(self, params: torch.Tensor) -> torch.Tensor:
        beta, S = self._get_covariance_matrix_from_params(params, use_cov_param=False)
        return -self._simulated_log_probs(
            beta, S, 0, self.n, self.rows_per_chunk, self.use_checkpoint
        )
    
    def _compute_observation_nlls_cov_param(self, params: torch.Tensor) -> torch.Tensor:
        beta, S = self._get_covariance_matrix_from_params(params, use_cov_param=True)
        return -self._simulated_log_probs(
            beta, S, 0, self.n, self.rows_per_chunk, self.use_checkpoint
        )
    
    def _compute_observation_nlls_transformed(self, params: torch.Tensor) -> torch.Tensor:
        raise ValueError("Transformed NLLs are only defined for factor models")
    
    def compute_observation_nlls_for_batch(
        self,
        params: torch.Tensor,
        start_idx: int,
        end_idx: int,
        use_cov_param: bool = False,
        use_transformed: bool = False
    ) -> torch.Tensor:
        if use_transformed:
            raise ValueError("Transformed NLLs are only defined for factor models")
        beta, S = self._get_covariance_matrix_from_params(params, use_cov_param, use_transformed)
        return -self._simulated_log_probs(
            beta, S, start_idx, end_idx, self.rows_per_chunk_batch, use_checkpoint=False
        )


class NegativeLogLikelihoodFactory:
    @staticmethod
    def create(
        method: str,
        X: torch.Tensor,
        y: torch.Tensor,
        param_spec: AnyParameterSpec,
        model: Optional[nn.Module] = None,
        ghk_config: Optional[GHKConfig] = None,
        log_L_diag_lower_bound: float = DEFAULT_LOG_L_DIAG_LOWER_BOUND,
        barrier_strength: float = DEFAULT_BARRIER_STRENGTH,
        cov_prior_scale: Optional[torch.Tensor] = None,
        cov_prior_df: Optional[float] = None,
        cov_prior_strength: float = DEFAULT_COV_PRIOR_STRENGTH,
        log_psi_lower_bound: float = DEFAULT_LOG_PSI_LOWER_BOUND,
        factor_barrier_strength: float = DEFAULT_FACTOR_BARRIER_STRENGTH,
        smoothed_config: Optional[SmoothedSimConfig] = None,
        error_distribution: str = "probit",
    ) -> NegativeLogLikelihood:
        if error_distribution not in ERROR_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown error distribution: {error_distribution}. Use one of {ERROR_DISTRIBUTIONS}."
            )
        if error_distribution == "gumbel":
            return NegativeLogLikelihoodFactory._create_gumbel(
                method=method,
                X=X,
                y=y,
                param_spec=param_spec,
                model=model,
                smoothed_config=smoothed_config,
                log_L_diag_lower_bound=log_L_diag_lower_bound,
                barrier_strength=barrier_strength,
                cov_prior_scale=cov_prior_scale,
                cov_prior_df=cov_prior_df,
                cov_prior_strength=cov_prior_strength,
                log_psi_lower_bound=log_psi_lower_bound,
                factor_barrier_strength=factor_barrier_strength,
            )
        
        if method.lower() == "emulator":
            if model is None:
                raise ValueError("Emulator method requires a model")
            return EmulatorNegativeLogLikelihood(
                X, y, model, param_spec, log_L_diag_lower_bound, barrier_strength,
                cov_prior_scale, cov_prior_df, cov_prior_strength,
                log_psi_lower_bound, factor_barrier_strength
            )
        elif method.lower() == "ghk":
            if ghk_config is None:
                ghk_config = GHKConfig()
            return GHKNegativeLogLikelihood(
                X, y, param_spec, ghk_config, log_L_diag_lower_bound, barrier_strength,
                cov_prior_scale, cov_prior_df, cov_prior_strength,
                log_psi_lower_bound, factor_barrier_strength
            )
        elif method.lower() == "smoothed":
            raise ValueError("The smoothed simulator is implemented for Gumbel errors only (error_distribution='gumbel').")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'emulator' or 'ghk'.")
    
    @staticmethod
    def _create_gumbel(
        method: str,
        X: torch.Tensor,
        y: torch.Tensor,
        param_spec: AnyParameterSpec,
        model: Optional[nn.Module],
        smoothed_config: Optional[SmoothedSimConfig],
        log_L_diag_lower_bound: float,
        barrier_strength: float,
        cov_prior_scale: Optional[torch.Tensor],
        cov_prior_df: Optional[float],
        cov_prior_strength: float,
        log_psi_lower_bound: float,
        factor_barrier_strength: float,
    ) -> NegativeLogLikelihood:
        """NLLs for correlated-Gumbel errors: emulator or softmax-smoothed simulator."""
        penalty_kwargs = dict(
            log_L_diag_lower_bound=log_L_diag_lower_bound,
            barrier_strength=barrier_strength,
            cov_prior_scale=cov_prior_scale,
            cov_prior_df=cov_prior_df,
            cov_prior_strength=cov_prior_strength,
            log_psi_lower_bound=log_psi_lower_bound,
            factor_barrier_strength=factor_barrier_strength,
        )
        if method.lower() == "emulator":
            if model is None:
                raise ValueError("Emulator method requires a model")
            return GumbelEmulatorNegativeLogLikelihood(X, y, model, param_spec, **penalty_kwargs)
        elif method.lower() == "smoothed":
            if smoothed_config is None:
                smoothed_config = SmoothedSimConfig()
            return SmoothedSimulatorNegativeLogLikelihood(X, y, param_spec, smoothed_config, **penalty_kwargs)
        elif method.lower() == "ghk":
            raise ValueError("GHK is only available for probit errors. Use 'smoothed' for Gumbel errors.")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'emulator' or 'smoothed' for Gumbel errors.")


# =============================================================================
# Optimization Classes
# =============================================================================

@dataclass
class OptimizationResult:
    params: torch.Tensor
    loss: float
    converged: bool
    grad_norm: float
    n_iter: int


class ManualLBFGS:
    def __init__(
        self,
        params: torch.Tensor,
        history_size: int = 20,
        min_step: float = 1e-6,
        max_step: float = 1.0,
        c1: float = 1e-4,
        max_ls_iter: int = 25,
    ):
        self.params = params.clone()
        self.history_size = history_size
        self.min_step = min_step
        self.max_step = max_step
        self.c1 = c1
        self.max_ls_iter = max_ls_iter
        
        self.s_history = []
        self.y_history = []
        self.rho_history = []
    
    def _two_loop_recursion(self, grad: torch.Tensor) -> torch.Tensor:
        q = grad.clone()
        n_hist = len(self.s_history)
        
        if n_hist == 0:
            return -grad
        
        alpha = []
        for i in range(n_hist - 1, -1, -1):
            a = self.rho_history[i] * torch.dot(self.s_history[i], q)
            alpha.append(a)
            q = q - a * self.y_history[i]
        alpha = alpha[::-1]
        
        s_last = self.s_history[-1]
        y_last = self.y_history[-1]
        gamma = torch.dot(s_last, y_last) / (torch.dot(y_last, y_last) + 1e-10)
        r = gamma * q
        
        for i in range(n_hist):
            beta = self.rho_history[i] * torch.dot(self.y_history[i], r)
            r = r + self.s_history[i] * (alpha[i] - beta)
        
        return -r
    
    def _line_search(
        self,
        loss_fn: Callable,
        params: torch.Tensor,
        direction: torch.Tensor,
        current_loss: float,
        current_grad: torch.Tensor,
    ) -> Tuple[float, torch.Tensor, float, bool, int]:
        step = self.max_step
        directional_deriv = torch.dot(current_grad, direction)
        
        if directional_deriv >= 0:
            direction = -current_grad
            directional_deriv = -torch.dot(current_grad, current_grad)
        
        best_step = self.min_step
        best_loss = float('inf')
        best_params = params + self.min_step * direction
        
        n_func_evals = 0
        
        for _ in range(self.max_ls_iter):
            new_params = params + step * direction
            new_params_eval = new_params.clone().detach().requires_grad_(True)
            new_loss_tensor = loss_fn(new_params_eval)
            n_func_evals += 1
            
            if not torch.isfinite(new_loss_tensor):
                step *= 0.5
                if step < self.min_step:
                    break
                continue
            
            new_loss = new_loss_tensor.item()
            
            if new_loss < best_loss:
                best_loss = new_loss
                best_step = step
                best_params = new_params.clone()
            
            armijo_loss = current_loss + self.c1 * step * directional_deriv
            if new_loss <= armijo_loss:
                return step, new_params, new_loss, True, 0
            
            step *= 0.5
            if step < self.min_step:
                break
        
        if best_loss < float('inf'):
            return best_step, best_params, best_loss, best_loss < current_loss * 1.1, 0
        return 0.0, params, current_loss, False, 0
    
    def _update_history(self, s: torch.Tensor, y: torch.Tensor):
        ys = torch.dot(y, s)
        if ys > 1e-10:
            if len(self.s_history) >= self.history_size:
                self.s_history.pop(0)
                self.y_history.pop(0)
                self.rho_history.pop(0)
            self.s_history.append(s.clone())
            self.y_history.append(y.clone())
            self.rho_history.append(1.0 / ys)
    
    def step(
        self,
        loss_fn: Callable,
        current_loss: float,
        current_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, float, torch.Tensor, bool, int]:
        direction = self._two_loop_recursion(current_grad.flatten())
        
        step, new_params, new_loss, success, ls_grad_evals = self._line_search(
            loss_fn, self.params, direction, current_loss, current_grad.flatten()
        )
        
        new_params_eval = new_params.clone().detach().requires_grad_(True)
        new_loss_tensor = loss_fn(new_params_eval)
        
        n_grad_evals = ls_grad_evals
        
        if torch.isfinite(new_loss_tensor) and new_loss_tensor.requires_grad:
            new_grad = torch.autograd.grad(new_loss_tensor, new_params_eval)[0]
            n_grad_evals += 1
        else:
            new_grad = current_grad.clone()
        
        if success and step > 0:
            s = new_params - self.params
            y = new_grad - current_grad.flatten()
            self._update_history(s, y)
        
        self.params = new_params.clone()
        
        return new_params, new_loss, new_grad, success, n_grad_evals


class RobustLBFGSOptimizer:
    def __init__(
        self,
        config: OptimizationConfig,
        min_step: float = 1e-6,
        max_step: float = 1.0,
        max_stalls: int = 100,
        verbose: bool = False
    ):
        self.config = config
        self.min_step = min_step
        self.max_step = max_step
        self.max_stalls = max_stalls
        self.verbose = verbose
    
    def optimize(
        self,
        initial_params: torch.Tensor, 
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        n_obs: int
    ) -> OptimizationResult:
        loss_tol = self.config.loss_tol_base / n_obs
        
        lbfgs = ManualLBFGS(
            params=initial_params.flatten(),
            history_size=self.config.history_size,
            min_step=self.min_step,
            max_step=self.max_step,
        )
        
        param_shape = initial_params.shape
        n_grad_evals = 0
        
        def loss_fn_flat(params_flat):
            return loss_fn(params_flat.reshape(param_shape))
        
        params = initial_params.clone().requires_grad_(True)
        loss = loss_fn(params)
        
        if not torch.isfinite(loss):
            return OptimizationResult(
                params=initial_params,
                loss=float('inf'),
                converged=False,
                grad_norm=float('inf'),
                n_iter=0
            )
        
        current_loss = loss.item()
        current_grad = torch.autograd.grad(loss, params)[0]
        n_grad_evals += 1
        
        best_loss = current_loss
        best_params = initial_params.clone()
        prev_loss = current_loss
        stall_count = 0
        converged = False
        
        lbfgs.params = initial_params.flatten().clone()
        
        for iteration in range(self.config.max_iter):
            new_params_flat, new_loss, new_grad, success, step_grad_evals = lbfgs.step(
                loss_fn_flat, current_loss, current_grad.flatten()
            )
            n_grad_evals += step_grad_evals
            
            new_params = new_params_flat.reshape(param_shape)
            new_grad = new_grad.reshape(param_shape)
            
            if new_loss < best_loss - 1e-12:
                best_loss = new_loss
                best_params = new_params.clone()
                stall_count = 0
            else:
                stall_count += 1
            
            loss_change = abs(new_loss - prev_loss)
            if loss_change < loss_tol and iteration > 5:
                converged = True
                break
            
            if stall_count >= self.max_stalls:
                break
            
            if self.verbose and iteration % 50 == 0:
                print(f"Iter {iteration}: loss={new_loss:.8f}, grad_norm={torch.norm(new_grad):.2e}")
            
            prev_loss = current_loss
            current_loss = new_loss
            current_grad = new_grad
        
        final_params = best_params.clone().detach().requires_grad_(True)
        final_loss_tensor = loss_fn(final_params)
        final_grad = torch.autograd.grad(final_loss_tensor, final_params)[0]
        final_grad_norm = torch.norm(final_grad).item()
        n_grad_evals += 1
        
        if final_grad_norm < self.config.grad_tol:
            converged = True
        
        return OptimizationResult(
            params=best_params,
            loss=best_loss,
            converged=converged,
            grad_norm=final_grad_norm,
            n_iter=n_grad_evals
        )


# =============================================================================
# Hessian and Standard Error Computation
# =============================================================================

def _compute_penalty_hessian(
    params_flat: torch.Tensor,
    loss_fn: NegativeLogLikelihood,
    shape: Tuple[int, ...],
    use_cov_param: bool,
    use_transformed: bool = False
) -> torch.Tensor:
    """Compute the Hessian of the total penalty (barrier + prior)."""
    def penalty_fn(p):
        p_shaped = p.reshape(shape)
        
        if loss_fn._is_factor_model:
            if use_transformed:
                params_dict = loss_fn.param_spec.unflatten_transformed(p_shaped)
                return loss_fn._compute_total_penalty_transformed(params_dict)
            else:
                params_dict = loss_fn.param_spec.unflatten(p_shaped)
                return loss_fn._compute_total_penalty(params_dict)
        elif use_cov_param:
            params_dict = loss_fn.cov_param_spec.unflatten(p_shaped)
            return loss_fn._compute_total_penalty_transformed(params_dict)
        else:
            params_dict = loss_fn.param_spec.unflatten(p_shaped)
            return loss_fn._compute_total_penalty(params_dict)
    
    hessian = torch.func.hessian(penalty_fn)(params_flat)
    return hessian


# =============================================================================
# Standard Error Computation
# =============================================================================

def compute_standard_errors(
    params: torch.Tensor,
    loss_fn: NegativeLogLikelihood,
    shape: Tuple[int, ...],
    use_cov_param: bool = False,
    use_transformed: bool = False,
    batch_size: int = 10000,
    eigenvalue_floor: float = DEFAULT_EIGENVALUE_FLOOR
) -> torch.Tensor:
    """
    Compute standard errors using the outer product of gradients (OPG) estimator.
    """
    n = loss_fn.n
    n_params = params.numel()
    device = params.device
    
    params_flat = params.detach().flatten().clone()
    
    I_likelihood = torch.zeros(n_params, n_params, device=device)
    
    def make_batch_nll_fn(s_idx, e_idx):
        def batch_nll_fn(p):
            return loss_fn.compute_observation_nlls_for_batch(
                p.reshape(shape), s_idx, e_idx, use_cov_param, use_transformed
            )
        return batch_nll_fn
    
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        
        batch_nll_fn = make_batch_nll_fn(start_idx, end_idx)
        scores_batch = torch.func.jacfwd(batch_nll_fn)(params_flat)
        
        I_likelihood += scores_batch.T @ scores_batch
        
        del scores_batch
    
    I_prior = _compute_penalty_hessian(params_flat, loss_fn, shape, use_cov_param, use_transformed)
    
    I_total = I_likelihood + I_prior
    
    covariance = robust_matrix_inverse(I_total, eigenvalue_floor)
    
    se = torch.sqrt(torch.clamp(torch.diag(covariance), min=0.0))
    
    return se.to(device)

# =============================================================================
# Simulation Results and Replication
# =============================================================================

@dataclass
class ReplicationResult:
    estimates: torch.Tensor
    standard_errors: torch.Tensor
    estimates_cov: Optional[torch.Tensor]
    standard_errors_cov: Optional[torch.Tensor]
    loss: float
    converged: bool
    grad_norm: float
    elapsed_time: float
    # Number of loss-gradient evaluations (not L-BFGS iterations); reported as N_Iter.
    n_iter: int


@dataclass
class SimulationResults:
    estimates: torch.Tensor  # (num_reps, n_params)
    standard_errors: torch.Tensor  # (num_reps, n_params)
    true_values: torch.Tensor  # (num_reps, n_params)
    estimates_cov: Optional[torch.Tensor]  # (num_reps, n_params_cov) or None for factor
    standard_errors_cov: Optional[torch.Tensor]  # (num_reps, n_params_cov) or None for factor
    true_values_cov: Optional[torch.Tensor]  # (num_reps, n_params_cov) or None for factor
    losses: torch.Tensor
    converged: torch.Tensor
    grad_norms: torch.Tensor
    elapsed_times: torch.Tensor
    n_iters: torch.Tensor
    config: Dict[str, Any]


def run_single_replication(
    X: torch.Tensor,
    param_spec: AnyParameterSpec,
    dgp: DataGeneratingProcess,
    estimation_method: str,
    device: torch.device,
    model: Optional[nn.Module] = None,
    ghk_config: Optional[GHKConfig] = None,
    opt_config: Optional[OptimizationConfig] = None,
    debug: bool = False,
    cov_prior_strength: float = DEFAULT_COV_PRIOR_STRENGTH,
    smoothed_config: Optional[SmoothedSimConfig] = None,
    error_distribution: str = "probit",
) -> ReplicationResult:
    """Run a single replication of the simulation."""
    n = X.shape[0]
    k = param_spec.k
    is_factor_model = isinstance(param_spec, FactorParameterSpec)
    if error_distribution == "gumbel" and is_factor_model:
        raise NotImplementedError("Correlated-Gumbel simulations support the dense covariance structure only")
    
    # Build covariance matrix for DGP
    if is_factor_model:
        log_psi_true = param_spec.log_psi_true
        gamma_true = param_spec.gamma_true
        
        if log_psi_true is not None:
            log_psi_true = log_psi_true.clone().detach()
        if gamma_true is not None:
            gamma_true = gamma_true.clone().detach()
        
        S = param_spec.build_full_covariance_matrix(log_psi_true, gamma_true)
        S_chol = param_spec.get_cholesky_factor_full(log_psi_true, gamma_true)
    else:
        log_L_diag_true = param_spec.log_L_diag_true
        L_lower_true = param_spec.L_lower_true
        
        if log_L_diag_true is not None:
            log_L_diag_true = log_L_diag_true.clone().detach()
        if L_lower_true is not None:
            L_lower_true = L_lower_true.clone().detach()
        
        S = param_spec.build_full_covariance_matrix(log_L_diag_true, L_lower_true)
        S_chol = param_spec.get_cholesky_factor_full(log_L_diag_true, L_lower_true)
    
    V_true = param_spec.compute_utilities(X, param_spec.beta_true)
    
    y = dgp.sample_choices(V_true, S, S_chol, model=model)
    start_time = time.time()
    
    if debug:
        print(f"\nChoice distribution:")
        for j in range(k):
            count = (y == j).sum().item()
            print(f"  Alternative {j}: {count} ({100*count/n:.1f}%)")
    
    loss_fn = NegativeLogLikelihoodFactory.create(
        method=estimation_method,
        X=X,
        y=y,
        param_spec=param_spec,
        model=model,
        ghk_config=ghk_config,
        cov_prior_strength=cov_prior_strength,
        smoothed_config=smoothed_config,
        error_distribution=error_distribution,
    )
    
    initial_params = param_spec.initialize(device)
    
    if debug:
        true_params = param_spec.get_true_values()
        loss_at_true = loss_fn(true_params)
        print(f"Loss at TRUE parameters: {loss_at_true.item():.6f}")
        print(f"Estimation method: {loss_fn.name}")

    if opt_config is None:
        opt_config = OptimizationConfig()
    optimizer = RobustLBFGSOptimizer(opt_config)
    opt_result = optimizer.optimize(initial_params, loss_fn, n_obs=n)
    
    if debug:
        true_params = param_spec.get_true_values()
        print(f"\nOptimization Results:")
        print(f"  Final loss: {opt_result.loss:.6f}")
        print(f"  Grad norm: {opt_result.grad_norm:.2e}")
        print(f"  Converged: {opt_result.converged}")
        print(f"  Iterations: {opt_result.n_iter}")
        param_names = param_spec.get_param_names()
        for i, name in enumerate(param_names):
            true_val = true_params[i].item()
            est_val = opt_result.params[i].item()
            print(f"  {name}: true={true_val:.4f}, est={est_val:.4f}, error={est_val-true_val:.4f}")
    
    if is_factor_model:
        # For factor models:
        # - Primary results (estimates, standard_errors, true_values) are on TRANSFORMED scale (psi, gamma)
        # - Secondary results (estimates_cov, etc.) are on NATIVE scale (log_psi, gamma)
        
        # Native scale results
        se_native = compute_standard_errors(
            opt_result.params, loss_fn, (param_spec.num_params,), 
            use_cov_param=False, use_transformed=False
        )
        
        # Transform estimates to reported scale
        params_dict = param_spec.unflatten(opt_result.params)
        beta_est = params_dict['beta']
        log_psi_est = params_dict['log_psi']
        gamma_est = params_dict['gamma']
        
        psi_est, gamma_est_transformed = param_spec.native_to_transformed(log_psi_est, gamma_est)
        
        # Build transformed estimates vector
        transformed_parts = [beta_est]
        if psi_est is not None:
            transformed_parts.append(psi_est)
        if gamma_est_transformed is not None:
            transformed_parts.append(gamma_est_transformed)
        estimates_transformed = torch.cat(transformed_parts)
        
        # Compute SEs on transformed scale
        se_transformed = compute_standard_errors(
            estimates_transformed, loss_fn, (param_spec.num_params_transformed,),
            use_cov_param=False, use_transformed=True
        )
        
        # Primary: transformed scale
        estimates = estimates_transformed
        standard_errors = se_transformed
        
        # Secondary: native scale
        estimates_cov = opt_result.params
        standard_errors_cov = se_native
        
    else:
        # For dense models:
        # - Primary results are on covariance parameterization
        # - Secondary results are on Cholesky parameterization
        
        se_chol = compute_standard_errors(
            opt_result.params, loss_fn, (param_spec.num_params,), use_cov_param=False
        )
        
        params_dict = param_spec.unflatten(opt_result.params)
        beta_est = params_dict['beta']
        log_L_diag_est = params_dict['log_L_diag']
        L_lower_est = params_dict['L_lower']
        
        variances_est, covariances_est = param_spec.to_covariance_params(log_L_diag_est, L_lower_est)
        
        cov_params_parts = [beta_est]
        if variances_est is not None:
            cov_params_parts.append(variances_est)
        if covariances_est is not None:
            cov_params_parts.append(covariances_est)
        estimates_cov_scale = torch.cat(cov_params_parts)
        
        se_cov = compute_standard_errors(
            estimates_cov_scale, loss_fn, (loss_fn.cov_param_spec.num_params,), use_cov_param=True
        )
        
        estimates = estimates_cov_scale
        standard_errors = se_cov
        estimates_cov = opt_result.params
        standard_errors_cov = se_chol
    
    elapsed_time = time.time() - start_time
    
    return ReplicationResult(
        estimates=estimates,
        standard_errors=standard_errors,
        estimates_cov=estimates_cov,
        standard_errors_cov=standard_errors_cov,
        loss=opt_result.loss,
        converged=opt_result.converged,
        grad_norm=opt_result.grad_norm,
        elapsed_time=elapsed_time,
        n_iter=opt_result.n_iter
    )


def run_simulation_study(
    num_reps: int,
    n: int,
    p: int,
    k: int,
    dgp: DataGeneratingProcess,
    estimation_method: str,
    device: torch.device,
    model: Optional[nn.Module] = None,
    ghk_config: Optional[GHKConfig] = None,
    opt_config: Optional[OptimizationConfig] = None,
    seed: Optional[int] = None,
    show_progress: bool = True,
    debug: bool = False,
    debug_first_n: int = 1,
    covariance_structure: str = "dense",
    cov_prior_strength: Optional[float] = None,
    smoothed_config: Optional[SmoothedSimConfig] = None,
    error_distribution: str = "probit",
) -> SimulationResults:
    """Run simulation study for a given estimation method and DGP."""
    base_seed = seed if seed is not None else 0
    is_factor_model = covariance_structure == "factor"
    if cov_prior_strength is None:
        cov_prior_strength = DEFAULT_COV_PRIOR_STRENGTH
    if error_distribution not in ERROR_DISTRIBUTIONS:
        raise ValueError(f"Unknown error distribution: {error_distribution}. Use one of {ERROR_DISTRIBUTIONS}.")
    if error_distribution == "gumbel" and is_factor_model:
        raise NotImplementedError("Correlated-Gumbel simulations support the dense covariance structure only")
    if estimation_method == "smoothed" and smoothed_config is None:
        raise ValueError("The smoothed simulator requires a SmoothedSimConfig")
    
    torch.manual_seed(base_seed)
    X = generate_covariates(n, k, p, device)
    
    beta_true = torch.linspace(0, 1, p, device=device)
    
    # Create template param spec for storage sizing
    if is_factor_model:
        log_psi_init, gamma_init = sample_factor_params(k, device)
        param_spec_template = FactorParameterSpec(
            p, k, beta_true, log_psi_init, gamma_init, device
        )
        # For factor models: primary is transformed (psi, gamma), secondary is native (log_psi, gamma)
        n_params = param_spec_template.num_params_transformed
        n_params_secondary = param_spec_template.num_params
        param_names = param_spec_template.get_param_names_transformed()
        param_names_secondary = param_spec_template.get_param_names()
        param_types = param_spec_template.get_param_types_transformed()
        param_types_secondary = param_spec_template.get_param_types()
    else:
        log_L_diag_init, L_lower_init = sample_cholesky_params(k, device)
        param_spec_template = CholeskyParameterSpec(p, k, beta_true, log_L_diag_init, L_lower_init, device)
        cov_param_spec_template = CovarianceParameterSpec.from_cholesky_spec(param_spec_template)
        # For dense models: primary is covariance, secondary is Cholesky
        n_params = cov_param_spec_template.num_params
        n_params_secondary = param_spec_template.num_params
        param_names = cov_param_spec_template.get_param_names()
        param_names_secondary = param_spec_template.get_param_names()
        param_types = cov_param_spec_template.get_param_types()
        param_types_secondary = param_spec_template.get_param_types()
    
    # Storage for primary parameterization
    estimates = torch.zeros(num_reps, n_params, device=device)
    standard_errors = torch.zeros(num_reps, n_params, device=device)
    true_values_all = torch.zeros(num_reps, n_params, device=device)
    
    # Storage for secondary parameterization
    estimates_cov = torch.zeros(num_reps, n_params_secondary, device=device)
    standard_errors_cov = torch.zeros(num_reps, n_params_secondary, device=device)
    true_values_cov_all = torch.zeros(num_reps, n_params_secondary, device=device)
    
    losses = torch.zeros(num_reps, device=device)
    converged = torch.zeros(num_reps, dtype=torch.bool, device=device)
    grad_norms = torch.zeros(num_reps, device=device)
    elapsed_times = torch.zeros(num_reps, device=device)
    n_iters = torch.zeros(num_reps, device=device)
    
    if estimation_method == "ghk":
        method_name = f"GHK({ghk_config.num_draws})" if ghk_config else "GHK"
        method_name_file = f"GHK{ghk_config.num_draws}" if ghk_config else "GHK"
    elif estimation_method == "smoothed":
        method_name = smoothed_method_label(smoothed_config.tau, smoothed_config.num_samples)
        method_name_file = smoothed_method_file_label(smoothed_config.tau, smoothed_config.num_samples)
    else:
        method_name = "Emulator"
        method_name_file = "Emulator"

    structure_suffix = "_factor" if is_factor_model else "_dense"
    if error_distribution == "gumbel":
        # Keeps Gumbel result checkpoints from overwriting the probit ones
        structure_suffix += "_gumbel"
    
    if estimation_method == "smoothed" and show_progress:
        memory_plan = plan_smoothed_memory(n, k, smoothed_config)
        if memory_plan.pool_rows >= n:
            pool_desc = "one draw set per observation"
        else:
            pool_desc = f"{memory_plan.pool_rows:,} draw sets, cycled"
        checkpoint_desc = "ON" if memory_plan.use_checkpoint else "off"
        print(
            f"    {method_name} memory plan: draw pool {memory_plan.pool_bytes / 2**30:.2f} GiB "
            f"({pool_desc}), estimated graph {memory_plan.graph_bytes / 2**30:.2f} GiB, "
            f"checkpointing {checkpoint_desc} "
            f"(budget after reserve {memory_plan.available_bytes / 2**30:.2f} GiB)"
        )
    desc = f"DGP={dgp.name}, Est={method_name}"
    iterator = tqdm(range(num_reps), desc=desc) if show_progress else range(num_reps)
    
    for rep in iterator:
        rep_debug = debug and (rep < debug_first_n)
        
        if rep_debug:
            print(f"\n{'#'*60}")
            print(f"# REPLICATION {rep + 1}")
            print(f"{'#'*60}")
        
        rep_seed = base_seed + rep * 1000
        torch.manual_seed(rep_seed)
        
        # Sample covariance parameters for this replication
        if is_factor_model:
            log_psi_true, gamma_true = sample_factor_params(k, device)
            param_spec = FactorParameterSpec(
                p, k, beta_true, log_psi_true, gamma_true, device
            )
            # Primary: transformed (psi, gamma), Secondary: native (log_psi, gamma)
            true_values_all[rep] = param_spec.get_true_values_transformed()
            true_values_cov_all[rep] = param_spec.get_true_values()
        else:
            log_L_diag_true, L_lower_true = sample_cholesky_params(k, device)
            param_spec = CholeskyParameterSpec(p, k, beta_true, log_L_diag_true, L_lower_true, device)
            cov_param_spec = CovarianceParameterSpec.from_cholesky_spec(param_spec)
            # Primary: covariance, Secondary: Cholesky
            true_values_all[rep] = cov_param_spec.get_true_values()
            true_values_cov_all[rep] = param_spec.get_true_values()
        
        if estimation_method == "ghk" and ghk_config is not None:
            rep_ghk_config = GHKConfig(
                num_draws=ghk_config.num_draws,
                seed=rep_seed + 500
            )
        else:
            rep_ghk_config = ghk_config
        
        rep_smoothed_config = None
        if estimation_method == "smoothed":
            rep_smoothed_config = replace(smoothed_config, seed=rep_seed + SIM_SEED_OFFSET)
        
        result = run_single_replication(
            X=X,
            param_spec=param_spec,
            dgp=dgp,
            estimation_method=estimation_method,
            device=device,
            model=model,
            ghk_config=rep_ghk_config,
            opt_config=opt_config,
            cov_prior_strength=cov_prior_strength,
            debug=rep_debug,
            smoothed_config=rep_smoothed_config,
            error_distribution=error_distribution,
        )
        
        estimates[rep] = result.estimates
        standard_errors[rep] = result.standard_errors
        estimates_cov[rep] = result.estimates_cov
        standard_errors_cov[rep] = result.standard_errors_cov
        
        losses[rep] = result.loss
        converged[rep] = result.converged
        grad_norms[rep] = result.grad_norm
        elapsed_times[rep] = result.elapsed_time
        n_iters[rep] = result.n_iter
    
    # Save checkpoint to CSV
    checkpoint_dir = "result-checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_filename = f"{method_name_file}_{dgp.name}_k{k}_n{n}{structure_suffix}.csv"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    checkpoint_data = {'rep': list(range(num_reps))}
    
    # Add primary parameter estimates, standard errors, and true values
    if is_factor_model:
        primary_prefix = "transformed"
        secondary_prefix = "native"
    else:
        primary_prefix = "cov"
        secondary_prefix = "chol"
    
    for j, pname in enumerate(param_names):
        col_name = pname.replace('[', '_').replace(']', '').replace(',', '_')
        checkpoint_data[f'{primary_prefix}_{col_name}_est'] = estimates[:, j].cpu().tolist()
        checkpoint_data[f'{primary_prefix}_{col_name}_se'] = standard_errors[:, j].cpu().tolist()
        checkpoint_data[f'{primary_prefix}_{col_name}_true'] = true_values_all[:, j].cpu().tolist()

    # Add secondary parameter estimates
    for j, pname in enumerate(param_names_secondary):
        col_name = pname.replace('[', '_').replace(']', '').replace(',', '_')
        checkpoint_data[f'{secondary_prefix}_{col_name}_est'] = estimates_cov[:, j].cpu().tolist()
        checkpoint_data[f'{secondary_prefix}_{col_name}_se'] = standard_errors_cov[:, j].cpu().tolist()
        checkpoint_data[f'{secondary_prefix}_{col_name}_true'] = true_values_cov_all[:, j].cpu().tolist()
    
    checkpoint_data['loss'] = losses.cpu().tolist()
    checkpoint_data['converged'] = converged.cpu().tolist()
    checkpoint_data['grad_norm'] = grad_norms.cpu().tolist()
    checkpoint_data['elapsed_time'] = elapsed_times.cpu().tolist()
    checkpoint_data['n_iter'] = n_iters.cpu().tolist()
    
    checkpoint_df = pd.DataFrame(checkpoint_data)
    checkpoint_df.to_csv(checkpoint_path, index=False)
    
    config_dict = {
        'n': n, 
        'p': p, 
        'k': k,
        'num_reps': num_reps,
        'dgp': dgp.name,
        'estimation_method': estimation_method,
        'covariance_structure': covariance_structure,
        'param_names': param_names,
        'param_types': param_types,
        'param_names_cov': param_names_secondary,
        'param_types_cov': param_types_secondary,
    }
        
    if estimation_method == "ghk" and ghk_config is not None:
        config_dict['ghk_num_draws'] = ghk_config.num_draws
    
    if estimation_method == "smoothed":
        config_dict['sim_tau'] = smoothed_config.tau
        config_dict['sim_num_samples'] = smoothed_config.num_samples
    
    if error_distribution != "probit":
        config_dict['error_distribution'] = error_distribution
    
    return SimulationResults(
        estimates=estimates,
        standard_errors=standard_errors,
        true_values=true_values_all,
        estimates_cov=estimates_cov,
        standard_errors_cov=standard_errors_cov,
        true_values_cov=true_values_cov_all,
        losses=losses,
        converged=converged,
        grad_norms=grad_norms,
        elapsed_times=elapsed_times,
        n_iters=n_iters,
        config=config_dict
    )


# =============================================================================
# Statistics and Reporting
# =============================================================================

def compute_simulation_statistics(
    results: SimulationResults, 
    n_bootstrap: int = 1000, 
    show_progress: bool = False
) -> SimulationStatistics:
    true_values = results.true_values
    estimates = results.estimates
    standard_errors = results.standard_errors
    elapsed_times = results.elapsed_times
    converged = results.converged
    grad_norms = results.grad_norms
    n_iters = results.n_iters
    config = results.config
    
    num_reps = estimates.shape[0]
    n_params = estimates.shape[1]
    device = estimates.device
    
    if true_values.dim() == 1:
        true_values = true_values.unsqueeze(0).expand(num_reps, -1)
    
    errors = estimates - true_values
    
    valid_estimates = ~torch.isnan(estimates)
    valid_standard_errors = ~torch.isnan(standard_errors)
    valid_errors = ~torch.isnan(errors)
    valid_both = valid_estimates & valid_standard_errors
    valid_elapsed_times = ~torch.isnan(elapsed_times)
    valid_grad_norms = ~torch.isnan(grad_norms)
    valid_n_iters = ~torch.isnan(n_iters.float())
    
    rmse = torch.sqrt((errors[valid_errors] ** 2).mean()).item()
    
    bias_per_param = torch.zeros(n_params, device=device)
    for j in range(n_params):
        valid_j = valid_errors[:, j]
        if valid_j.any():
            bias_per_param[j] = errors[valid_j, j].mean()
        else:
            bias_per_param[j] = torch.nan
    valid_bias = ~torch.isnan(bias_per_param)
    rms_bias = torch.sqrt((bias_per_param[valid_bias] ** 2).mean()).item()
    
    empirical_se_per_param = torch.zeros(n_params, device=device)
    for j in range(n_params):
        valid_j = valid_errors[:, j]
        if valid_j.sum() > 1:
            empirical_se_per_param[j] = errors[valid_j, j].std()
        else:
            empirical_se_per_param[j] = torch.nan
    valid_emp_se = ~torch.isnan(empirical_se_per_param)
    empirical_se_mean = empirical_se_per_param[valid_emp_se].mean().item()
    
    sqrt_avg_se_squared_per_param = torch.zeros(n_params, device=device)
    for j in range(n_params):
        valid_j = valid_standard_errors[:, j]
        if valid_j.any():
            avg_se_squared_j = (standard_errors[valid_j, j] ** 2).mean()
            sqrt_avg_se_squared_per_param[j] = torch.sqrt(avg_se_squared_j)
        else:
            sqrt_avg_se_squared_per_param[j] = torch.nan
    valid_sqrt_se = ~torch.isnan(sqrt_avg_se_squared_per_param)
    sqrt_avg_se_squared_mean = sqrt_avg_se_squared_per_param[valid_sqrt_se].mean().item()
    
    se_ratio_mean = sqrt_avg_se_squared_mean / (empirical_se_mean + 1e-10)
    
    lower = estimates - CONFIDENCE_Z_95 * standard_errors
    upper = estimates + CONFIDENCE_Z_95 * standard_errors
    covered = (lower <= true_values) & (true_values <= upper)

    # Compute coverage per replication (mean across parameters), then average across replications
    covered_float = covered.float()
    covered_float[~valid_both] = torch.nan
    coverage_per_rep = torch.nanmean(covered_float, dim=1)
    valid_coverage_reps = ~torch.isnan(coverage_per_rep)
    n_valid_coverage_reps = valid_coverage_reps.sum().item()
    coverage_mean = coverage_per_rep[valid_coverage_reps].mean().item()
    coverage_se = (coverage_per_rep[valid_coverage_reps].std() / np.sqrt(n_valid_coverage_reps)).item() if n_valid_coverage_reps > 1 else 0.0    

    conv_rate = converged.float().mean().item()
    grad_norm_mean = grad_norms[valid_grad_norms].mean().item() if valid_grad_norms.any() else float('nan')
    time_mean = elapsed_times[valid_elapsed_times].mean().item() if valid_elapsed_times.any() else float('nan')
    n_iter_mean = n_iters[valid_n_iters].float().mean().item() if valid_n_iters.any() else float('nan')
    
    bootstrap_indices = torch.randint(0, num_reps, (n_bootstrap, num_reps), device=device)
    
    rmse_boot = torch.zeros(n_bootstrap, device=device)
    rms_bias_boot = torch.zeros(n_bootstrap, device=device)
    empirical_se_boot = torch.zeros(n_bootstrap, device=device)
    sqrt_avg_se_sq_boot = torch.zeros(n_bootstrap, device=device)
    se_ratio_boot = torch.zeros(n_bootstrap, device=device)
    
    iterator = tqdm(range(n_bootstrap), desc="Bootstrap") if show_progress else range(n_bootstrap)
    
    for b in iterator:
        idx = bootstrap_indices[b]
        
        estimates_b = estimates[idx]
        standard_errors_b = standard_errors[idx]
        true_values_b = true_values[idx]
        errors_b = estimates_b - true_values_b
               
        valid_standard_errors_b = ~torch.isnan(standard_errors_b)
        valid_errors_b = ~torch.isnan(errors_b)
        valid_both_b = ~torch.isnan(estimates_b) & valid_standard_errors_b
        
        rmse_boot[b] = torch.sqrt((errors_b[valid_errors_b] ** 2).mean())
        
        bias_per_param_b = torch.zeros(n_params, device=device)
        for j in range(n_params):
            valid_j = valid_errors_b[:, j]
            if valid_j.any():
                bias_per_param_b[j] = errors_b[valid_j, j].mean()
            else:
                bias_per_param_b[j] = torch.nan
        valid_bias_b = ~torch.isnan(bias_per_param_b)
        rms_bias_boot[b] = torch.sqrt((bias_per_param_b[valid_bias_b] ** 2).mean())
        
        empirical_se_per_param_b = torch.zeros(n_params, device=device)
        for j in range(n_params):
            valid_j = valid_errors_b[:, j]
            if valid_j.sum() > 1:
                empirical_se_per_param_b[j] = errors_b[valid_j, j].std()
            else:
                empirical_se_per_param_b[j] = torch.nan
        valid_emp_se_b = ~torch.isnan(empirical_se_per_param_b)
        empirical_se_boot[b] = empirical_se_per_param_b[valid_emp_se_b].mean()
        
        sqrt_avg_se_squared_per_param_b = torch.zeros(n_params, device=device)
        for j in range(n_params):
            valid_j = valid_standard_errors_b[:, j]
            if valid_j.any():
                avg_se_squared_j = (standard_errors_b[valid_j, j] ** 2).mean()
                sqrt_avg_se_squared_per_param_b[j] = torch.sqrt(avg_se_squared_j)
            else:
                sqrt_avg_se_squared_per_param_b[j] = torch.nan
        valid_sqrt_se_b = ~torch.isnan(sqrt_avg_se_squared_per_param_b)
        sqrt_avg_se_sq_boot[b] = sqrt_avg_se_squared_per_param_b[valid_sqrt_se_b].mean()
        
        se_ratio_boot[b] = sqrt_avg_se_sq_boot[b] / (empirical_se_boot[b] + 1e-10)
            
    rmse_se = rmse_boot.std().item()
    rms_bias_se = rms_bias_boot.std().item()
    empirical_se_se = empirical_se_boot.std().item()
    sqrt_avg_se_squared_se = sqrt_avg_se_sq_boot.std().item()
    se_ratio_se = se_ratio_boot.std().item()
    
    conv_rate_se = np.sqrt(conv_rate * (1 - conv_rate) / num_reps) if 0 < conv_rate < 1 else 0.0
    n_valid_grad = valid_grad_norms.sum().item()
    grad_norm_se = grad_norms[valid_grad_norms].std().item() / np.sqrt(n_valid_grad) if n_valid_grad > 1 else 0.0
    n_valid_time = valid_elapsed_times.sum().item()
    time_se = elapsed_times[valid_elapsed_times].std().item() / np.sqrt(n_valid_time) if n_valid_time > 1 else 0.0
    n_valid_iter = valid_n_iters.sum().item()
    n_iter_se = n_iters[valid_n_iters].float().std().item() / np.sqrt(n_valid_iter) if n_valid_iter > 1 else 0.0
    
    method = config.get('estimation_method', 'unknown')
    dgp_name = config['dgp']
    
    if method == 'ghk':
        method_desc = f"GHK({config.get('ghk_num_draws', '?')})"
    elif method == 'smoothed':
        method_desc = smoothed_method_label(config['sim_tau'], config['sim_num_samples'])
    else:
        method_desc = "Emulator"

    return {
        'rms_bias': (rms_bias, rms_bias_se),
        'rmse': (rmse, rmse_se),
        'empirical_se': (empirical_se_mean, empirical_se_se),
        'sqrt_avg_se_sq': (sqrt_avg_se_squared_mean, sqrt_avg_se_squared_se),
        'se_ratio': (se_ratio_mean, se_ratio_se),
        'coverage': (coverage_mean, coverage_se),
        'conv_rate': (conv_rate, conv_rate_se),
        'grad_norm': (grad_norm_mean, grad_norm_se),
        'elapsed_time': (time_mean, time_se),
        'n_iter': (n_iter_mean, n_iter_se),
        'n': config['n'],
        'dgp': dgp_name,
        'estimation_method': method_desc,
    }


def print_summary_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 155)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 155)
    
    display_df = df.copy()
    display_df['n'] = display_df['n'].apply(lambda x: f"{int(x):,}")
    
    for col in ['RMSE', 'RMS_Bias', 'Emp_SE', 'Avg_SE']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    display_df['SE_Ratio'] = display_df['SE_Ratio'].apply(lambda x: f"{x:.3f}")
    display_df['Coverage'] = display_df['Coverage'].apply(lambda x: f"{x:.1%}")
    display_df['Conv_Rate'] = display_df['Conv_Rate'].apply(lambda x: f"{x:.1%}")
    display_df['Grad_Norm'] = display_df['Grad_Norm'].apply(lambda x: f"{x:.2e}")
    display_df['N_Iter'] = display_df['N_Iter'].apply(lambda x: f"{x:.1f}")
    display_df['Time'] = display_df['Time'].apply(lambda x: f"{x:.2f}s")
    
    cols_to_show = ['n', 'DGP', 'Method', 'RMSE', 'RMS_Bias', 'Emp_SE', 'Avg_SE', 
                    'SE_Ratio', 'Coverage', 'Conv_Rate', 'Grad_Norm', 'N_Iter', 'Time']
    
    print(display_df[cols_to_show].to_string(index=False))
    print("=" * 155)


def format_with_se(val: float, se: Optional[float], decimals: int = 3) -> str:
    if se is None:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} ({se:.{decimals}f})"


def generate_latex_table(
    df: pd.DataFrame, 
    caption: str = "Simulation Results", 
    label: str = "tab:simulation",
    k: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> str:
    """
    Generate a LaTeX table from simulation results DataFrame.
    
    Args:
        df: DataFrame with simulation results
        caption: Table caption (can include {k} placeholder)
        label: Table label (can include {k} placeholder)
        k: Number of alternatives (for caption/label substitution)
        columns: List of column names to include. If None, uses default set.
    """
    # Default columns if not specified
    if columns is None:
        columns = ['RMSE', 'RMS_Bias', 'SE_Ratio', 'Coverage', 'Time']
    
    # Define all available metrics with their SE column, display name, and decimal places
    all_metrics = {
        'RMSE': ('RMSE_SE', 'RMSE', 3),
        'RMS_Bias': ('RMS_Bias_SE', 'RMS Bias', 3),
        'Emp_SE': ('Emp_SE_SE', 'Emp.~SE', 3),
        'Avg_SE': ('Avg_SE_SE', r'$\sqrt{\overline{\text{SE}^2}}$', 3),
        'SE_Ratio': ('SE_Ratio_SE', 'SE Ratio', 2),
        'Coverage': ('Coverage_SE', 'Coverage', 2),
        'Time': ('Time_SE', 'Time (s)', 2),
        'Conv_Rate': ('Conv_Rate_SE', 'Conv Rate', 2),
        'Grad_Norm': ('Grad_Norm_SE', 'Grad Norm', 2),
        'N_Iter': ('N_Iter_SE', 'N Iter', 1),
    }
    
    # Build metrics list from requested columns
    metrics = []
    for col in columns:
        if col in all_metrics:
            se_col, display_name, decimals = all_metrics[col]
            metrics.append((col, se_col, display_name, decimals))
        else:
            raise ValueError(f"Unknown column: {col}")
    
    # Build header with proper alignment
    header_cols = [
        r'\multicolumn{1}{c}{$n$}',
        r'\multicolumn{1}{l}{DGP}',
        r'\multicolumn{1}{l}{Method}',
    ]
    for _, _, display_name, _ in metrics:
        if display_name == 'Time (s)':
            header_cols.append(r'\multicolumn{1}{r}{' + display_name + r'}')
        else:
            header_cols.append(r'\multicolumn{1}{c}{' + display_name + r'}')
    
    # Determine column alignment string
    n_metric_cols = len(metrics)
    col_align = '@{}rll' + 'r' * n_metric_cols + '@{}'
    
    # Sort helper functions
    def sort_key_method(row):
        """Sort key for methods: Emulator first, then GHK by draws, then Sim by (tau, R)."""
        method = row['Method']
        if method == 'Emulator':
            return (0, 0)
        elif method.startswith('GHK('):
            # Extract number from GHK(xxx)
            num_draws = int(method[4:-1])
            return (1, num_draws)
        elif method.startswith('Sim('):
            # Smoothed simulators Sim(tau,R): by tau, then by R
            try:
                tau_str, r_str = method[4:-1].split(',')
                return (2, float(tau_str), int(r_str))
            except ValueError:
                return (2, 0)
        else:
            return (2, 0)

    def sort_key_dgp(row):
        """Sort key for DGP: Emulator first, then Probit (or Gumbel)."""
        dgp = row['DGP']
        if dgp == 'Emulator':
            return 0
        elif dgp == 'Probit':
            return 1
        elif dgp == 'Gumbel':
            return 1
        else:
            return 2
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\footnotesize")
    latex_lines.append(r"\begin{tabular}{" + col_align + r"}")
    latex_lines.append(r"\toprule")
    
    header_str = " & ".join(header_cols) + r" \\"
    latex_lines.append(header_str)
    latex_lines.append(r"\midrule")
    
    unique_ns = sorted(df['n'].unique())
    
    for n_idx, n in enumerate(unique_ns):
        n_df = df[df['n'] == n].copy()
        n_df['_sort_dgp'] = n_df.apply(sort_key_dgp, axis=1)
        n_df['_sort_method'] = n_df.apply(sort_key_method, axis=1)
        n_df = n_df.sort_values(['_sort_dgp', '_sort_method'])
        n_row_count = len(n_df)
        
        first_n_row = True
        
        for row_idx, (_, row) in enumerate(n_df.iterrows()):
            if first_n_row:
                n_cell = r"\multirow{" + str(n_row_count) + r"}{*}{" + f"{int(n):,}" + r"}"
                first_n_row = False
            else:
                n_cell = ""
            
            dgp_cell = row['DGP']
            method_cell = row['Method']
            
            values = [n_cell, dgp_cell, method_cell]
            for col, se_col, _, decimals in metrics:
                val = row[col]
                se = row[se_col] if se_col and se_col in row else None
                values.append(format_with_se(val, se, decimals))
            
            row_str = " & ".join(values) + r" \\"
            latex_lines.append(row_str)
        
        if n_idx < len(unique_ns) - 1:
            latex_lines.append(r"\midrule")
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    
    # Format caption and label with k value if provided
    if k is not None:
        caption_formatted = caption.replace("{k}", str(k)).replace(r"\numalt", r"$\numalt=" + str(k) + r"$")
        if r"$\numalt" not in caption_formatted and "{k}" not in caption:
            # Add k info if not already present
            caption_formatted = caption + rf" with $\numalt={k}$"
        label_formatted = label.replace("{k}", str(k))
    else:
        caption_formatted = caption
        label_formatted = label
    
    latex_lines.append(r"\caption{" + caption_formatted + r"}")
    latex_lines.append(r"\label{" + label_formatted + r"}")
    latex_lines.append(r"\end{table}")
    
    return "\n".join(latex_lines)


def filter_results_by_param_type(
    results: SimulationResults, 
    param_type: str,
    use_secondary: bool = False
) -> Optional[SimulationResults]:
    """Filter SimulationResults to include only parameters of a specific type."""
    if use_secondary:
        param_types_list = results.config.get('param_types_cov', [])
        param_names_list = results.config.get('param_names_cov', [])
        estimates = results.estimates_cov
        standard_errors = results.standard_errors_cov
        true_values = results.true_values_cov
    else:
        param_types_list = results.config['param_types']
        param_names_list = results.config['param_names']
        estimates = results.estimates
        standard_errors = results.standard_errors
        true_values = results.true_values
    
    if estimates is None:
        return None
    
    mask = [i for i, pt in enumerate(param_types_list) if pt == param_type]
    
    if len(mask) == 0:
        return None
    
    filtered_estimates = estimates[:, mask]
    filtered_standard_errors = standard_errors[:, mask]
    
    if true_values.dim() == 1:
        filtered_true_values = true_values[mask]
    else:
        filtered_true_values = true_values[:, mask]
    
    filtered_param_names = [param_names_list[i] for i in mask]
    filtered_param_types = [param_types_list[i] for i in mask]
    
    new_config = results.config.copy()
    new_config['param_names'] = filtered_param_names
    new_config['param_types'] = filtered_param_types
    new_config['param_names_cov'] = filtered_param_names
    new_config['param_types_cov'] = filtered_param_types
    
    return SimulationResults(
        estimates=filtered_estimates,
        standard_errors=filtered_standard_errors,
        true_values=filtered_true_values,
        estimates_cov=filtered_estimates,
        standard_errors_cov=filtered_standard_errors,
        true_values_cov=filtered_true_values,
        losses=results.losses,
        converged=results.converged,
        grad_norms=results.grad_norms,
        elapsed_times=results.elapsed_times,
        n_iters=results.n_iters,
        config=new_config
    )


def compute_statistics_by_param_type(
    results: SimulationResults,
    n_bootstrap: int = 1000,
    show_progress: bool = False
) -> Dict[str, SimulationStatistics]:
    """
    Compute simulation statistics separately for each parameter type and overall.
    """
    stats_by_type: Dict[str, SimulationStatistics] = {}
    
    # Determine which parameter types to compute based on primary parameterization
    param_types_list = results.config['param_types']
    unique_param_types = list(set(param_types_list))
    
    for param_type in unique_param_types:
        filtered_results = filter_results_by_param_type(results, param_type)
        if filtered_results is not None:
            stats_by_type[param_type] = compute_simulation_statistics(
                filtered_results, n_bootstrap=n_bootstrap, show_progress=show_progress
            )
    
    # Also compute statistics for secondary parameterization types
    param_types_secondary = results.config.get('param_types_cov', [])
    unique_secondary_types = list(set(param_types_secondary))
    
    for param_type in unique_secondary_types:
        if param_type not in stats_by_type:
            filtered_results = filter_results_by_param_type(results, param_type, use_secondary=True)
            if filtered_results is not None:
                stats_by_type[param_type] = compute_simulation_statistics(
                    filtered_results, n_bootstrap=n_bootstrap, show_progress=show_progress
                )
    
    # Compute 'all' statistics using primary parameterization
    param_names_list = results.config['param_names']
    
    mask_all = list(range(len(param_types_list)))
    
    if len(mask_all) > 0:
        filtered_estimates = results.estimates[:, mask_all]
        filtered_standard_errors = results.standard_errors[:, mask_all]
        
        if results.true_values.dim() == 1:
            filtered_true_values = results.true_values[mask_all]
        else:
            filtered_true_values = results.true_values[:, mask_all]
        
        filtered_param_names = [param_names_list[i] for i in mask_all]
        filtered_param_types = [param_types_list[i] for i in mask_all]
        
        new_config = results.config.copy()
        new_config['param_names'] = filtered_param_names
        new_config['param_types'] = filtered_param_types
        new_config['param_names_cov'] = filtered_param_names
        new_config['param_types_cov'] = filtered_param_types
        
        all_results = SimulationResults(
            estimates=filtered_estimates,
            standard_errors=filtered_standard_errors,
            true_values=filtered_true_values,
            estimates_cov=filtered_estimates,
            standard_errors_cov=filtered_standard_errors,
            true_values_cov=filtered_true_values,
            losses=results.losses,
            converged=results.converged,
            grad_norms=results.grad_norms,
            elapsed_times=results.elapsed_times,
            n_iters=results.n_iters,
            config=new_config
        )
        
        stats_by_type['all'] = compute_simulation_statistics(
            all_results, n_bootstrap=n_bootstrap, show_progress=show_progress
        )
    
    return stats_by_type


def generate_latex_tables_by_param_type(
    dfs_by_type: Dict[str, pd.DataFrame],
    caption_prefix: str = "Comparison of Emulator and GHK estimation methods for multinomial probit models",
    label_prefix: str = "tab:emulator_vs_ghk",
    k: Optional[int] = None,
    columns: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Generate separate LaTeX tables for each parameter type."""
    type_labels = {
        'all': ('All Parameters', 'all'),
        'beta': ('Coefficients', 'beta'),
        'variance': ('Variance Parameters', 'variance'),
        'covariance': ('Covariance Parameters', 'covariance'),
        'L_diag': ('Log-Diagonal Cholesky Parameters', 'ldiag'),
        'L_lower': ('Off-Diagonal Cholesky Parameters', 'llower'),
        # Factor model types (anchored normalization)
        'psi': ('Idiosyncratic Variance Parameters', 'psi'),
        'gamma': ('Factor Loading Parameters', 'gamma'),
        'log_psi': ('Log Idiosyncratic Variance Parameters', 'logpsi'),
    }
    
    latex_tables = {}
    
    for param_type, df in dfs_by_type.items():
        if param_type in type_labels:
            desc, suffix = type_labels[param_type]
            caption = f"{caption_prefix}: {desc}"
            if k is not None:
                label = f"{label_prefix}:{k}:{suffix}"
            else:
                label = f"{label_prefix}:{suffix}"
        else:
            caption = f"{caption_prefix}: {param_type}"
            if k is not None:
                label = f"{label_prefix}:{k}:{param_type}"
            else:
                label = f"{label_prefix}:{param_type}"
        
        latex_tables[param_type] = generate_latex_table(
            df, caption=caption, label=label, k=k, columns=columns
        )
    
    return latex_tables


def create_results_dataframe_by_param_type(
    all_results_by_type: List[Dict[str, SimulationStatistics]]
) -> Dict[str, pd.DataFrame]:
    """
    Create separate DataFrames for each parameter type.
    """
    param_types = set()
    for stats_by_type in all_results_by_type:
        param_types.update(stats_by_type.keys())
    
    dfs_by_type = {}
    
    for param_type in param_types:
        rows = []
        for stats_by_type in all_results_by_type:
            if param_type in stats_by_type:
                stats = stats_by_type[param_type]
                row = {
                    'Method': stats.get('estimation_method', 'Unknown'),
                    'DGP': stats['dgp'],
                    'n': stats['n'],
                    'RMSE': stats['rmse'][0],
                    'RMSE_SE': stats['rmse'][1],
                    'RMS_Bias': stats['rms_bias'][0],
                    'RMS_Bias_SE': stats['rms_bias'][1],
                    'Emp_SE': stats['empirical_se'][0],
                    'Emp_SE_SE': stats['empirical_se'][1],
                    'Avg_SE': stats['sqrt_avg_se_sq'][0],
                    'Avg_SE_SE': stats['sqrt_avg_se_sq'][1],
                    'SE_Ratio': stats['se_ratio'][0],
                    'SE_Ratio_SE': stats['se_ratio'][1],
                    'Coverage': stats['coverage'][0],
                    'Coverage_SE': stats['coverage'][1],
                    'Conv_Rate': stats['conv_rate'][0],
                    'Conv_Rate_SE': stats['conv_rate'][1],
                    'Grad_Norm': stats['grad_norm'][0],
                    'Grad_Norm_SE': stats['grad_norm'][1],
                    'N_Iter': stats['n_iter'][0],
                    'N_Iter_SE': stats['n_iter'][1],
                    'Time': stats['elapsed_time'][0],
                    'Time_SE': stats['elapsed_time'][1],
                }
                rows.append(row)
        
        if rows:
            dfs_by_type[param_type] = pd.DataFrame(rows)
    
    return dfs_by_type