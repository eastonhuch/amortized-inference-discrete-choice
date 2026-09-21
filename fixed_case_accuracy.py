"""
fixed_case_accuracy.py
======================

Accuracy of the trained emulators against softmax-smoothed sample averages at fixed
(v, Sigma), with break-even simulation sizes and run times on CPU and MPS.

Cases
-----
family (probit, gumbel) x K (3, 5, 10) x factor structure (0, 1, 2 factors) x utility
profile (v = 0, v_2 = 2, v_2 = 4, v linear from 0 to -4). The scale matrix is

    Sigma = [[1, 0'], [0, Omega]],   Omega = G G' / f + I   (f = 0: Omega = I),

which is the factor construction in training.simulate_data with the uniquenesses fixed
at their chi-squared(1) mean of one. Column 1 of G rises linearly from 0 to 2 over
alternatives 2..K and column 2 falls linearly from 2 to 0. Utilities are centered, as in
training.simulate_data. With f = 0, the Gumbel model is the multinomial logit.

Reference ("truth")
-------------------
Hard-choice frequencies from n_reference = 1e8 draws of U = v + Sigma^{1/2} z, drawn as
in training.simulate_data (block Cholesky root for probit, symmetric root for Gumbel,
N(0, 1) or Gumbel(-gamma, 1) draws), in float32. The standard error of each
reference probability is sqrt(P (1 - P) / n_reference).

Smoothed simulator
------------------
P~_j(R) = (1/R) sum_r softmax(tau U_r)_j, with tau applied to the level-scale
utilities exactly as in training._compute_gradients. On the same 1e8 draws, for each
tau, the reference pass also records the smoothed limit P_tau = E[Y], the per-draw
variance Var(Y), and the smoothing bias P_tau - P. The bias is paired on the same draws,
so its noise is negligible. Because P~ is a mean of R iid draws,

    E[(P~_j(R) - P_j)^2] = (P_tau,j - P_j)^2 + Var(Y_j) / R          (exact)

The R -> infinity limit is therefore the squared smoothing bias. The simulator matches
the emulator's squared error e_j^2 at

    R*_j = Var(Y_j) / (e_j^2 - (P_tau,j - P_j)^2),   or never if e_j^2 <= bias^2.

Here e_j^2 is corrected for the reference's own noise: (P_emu - P_ref)^2 - SE^2.
Hard counts (tau = inf) are included as an unbiased reference simulator.

Nested replications check the formula on each device (n_reps simulators, checkpoints
R = 10, ..., 1e5). They also give log-scale errors as a secondary metric.

Timing
------
Seconds per likelihood pass over 1,000 rows (one shared Sigma, parameterized by the
Cholesky factor L of Omega), forward only (headline) and forward + backward (gradients
with respect to v and L). The simulator draws its errors on the fly for each chunk
of rows, at R = 10, ..., 1e4. Its time at the break-even R is extrapolated linearly
in R.

Usage
-----
PyTorch reads the MPS fallback flag when torch is first imported, so set it in the
first notebook cell (or import this module before torch)::

    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from fixed_case_accuracy import self_test, run_comparison, analyze, write_paper_outputs

    self_test()                     # about a minute: closed forms, quadrature, formula, emulators
    results = run_comparison()      # a few minutes on an Apple-silicon Mac (cached afterwards)
    report = analyze(results)       # diagnostic tables, figures and a text summary
    paper = write_paper_outputs(results)   # paper figure and LaTeX table (see below)

Paper outputs
-------------
``write_paper_outputs`` (also run by ``python fixed_case_accuracy.py``) writes

  figures/smoothed_error_vs_r.{pdf,png,eps}       via config_plt.save_my_fig
  <output_dir>/paper/smoothed_cost_table.tex     LaTeX table, ready to paste

and prints the numbers quoted in the paper's text, with their sources. All figures use
config_plt (usetex, serif fonts, the paper's palette), so config_plt.py must be importable.

Stages are cached in ``output_dir`` with their settings. A rerun with the same settings
loads them; ``rerun=True`` recomputes. Everything runs in float32. Chunk totals are
summed on the host in float64, which only prevents round-off when adding 1e8 values.
"""

from __future__ import annotations

import os
import sys
import warnings

# The flag is read when torch is first imported, so setting it here only helps if this
# module is imported before torch.
if "torch" in sys.modules and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
    warnings.warn(
        "torch was imported before PYTORCH_ENABLE_MPS_FALLBACK was set to '1', so the MPS "
        "CPU fallback is not active in this process. Set the variable before torch is "
        "first imported (first notebook cell, or export it in the shell) and restart the "
        "kernel.",
        RuntimeWarning,
        stacklevel=2,
    )
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import math
import time
import zlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import canonicalize, omega_to_cov
from training import _compute_sqrt_L, load_checkpoint
from config_plt import colors as PAPER_COLORS, config_plt, save_my_fig

try:
    from IPython.display import display
except ImportError:  # outside IPython
    def display(obj):
        print(obj)

__all__ = [
    "write_paper_outputs",
    "paper_figure",
    "paper_table",
    "Case",
    "build_cases",
    "case_inputs",
    "utility_vector",
    "factor_omega",
    "run_comparison",
    "analyze",
    "self_test",
]


# =============================================================================
# Settings
# =============================================================================

EULER_MASCHERONI = 0.57721566490153286060651209  # same constant as training.simulate_data
HARD = math.inf  # tau label for hard (unsmoothed) choice frequencies

FAMILIES = ("probit", "gumbel")
DEFAULT_EMULATOR_PATHS = {
    "probit": os.path.join("model-checkpoints", "probit12.pt"),
    "gumbel": os.path.join("model-checkpoints", "gumbel-large12.pt"),
}
DEFAULT_K_VALUES = (3, 5, 10)
DEFAULT_FACTOR_COUNTS = (0, 1, 2)
DEFAULT_TAUS = (5.0, 10.0, 20.0, 40.0)

DEFAULT_N_REFERENCE = 100_000_000
DEFAULT_N_REPS = 100
DEFAULT_REP_R_GRID = (10, 100, 1_000, 10_000, 100_000)

DEFAULT_TIMING_ROWS = 1_000
DEFAULT_TIMING_R_GRID = (10, 100, 1_000, 10_000)
DEFAULT_TIMING_REPEATS = 3
DEFAULT_TIMING_TAU = 5.0

DEFAULT_MAX_CHUNK_ELEMENTS = 2**23  # rows x draws x K per simulation chunk (memory control)
DEFAULT_OUTPUT_DIR = "fixed_case_accuracy"

# Utility profiles: name -> (label, function of K returning the raw utility vector).
UTILITY_PROFILES: Dict[str, Tuple[str, Callable[[int], torch.Tensor]]] = {
    "zero": ("v = 0", lambda k: torch.zeros(k)),
    "v2_2": ("v₂ = 2", lambda k: torch.zeros(k).index_fill_(0, torch.tensor([1]), 2.0)),
    "v2_4": ("v₂ = 4", lambda k: torch.zeros(k).index_fill_(0, torch.tensor([1]), 4.0)),
    "lin0_m4": ("v linear 0 → −4", lambda k: torch.linspace(0.0, -4.0, k)),
}

_STREAM_REFERENCE = 1
_STREAM_REPLICATIONS = 2
_STREAM_TIMING = 3
_CASE_COLUMNS = ["case_id", "family", "k", "n_factors", "utility"]


# =============================================================================
# Cases
# =============================================================================

@dataclass(frozen=True)
class Case:
    """One fixed choice situation: error family, number of alternatives, factor count, utility profile."""

    family: str
    k: int
    n_factors: int
    utility: str

    @property
    def case_id(self) -> str:
        return f"{self.family}_K{self.k}_f{self.n_factors}_{self.utility}"

    @property
    def gumbel(self) -> bool:
        return self.family == "gumbel"

    @property
    def key(self) -> int:
        """Stable integer key for seeding (does not depend on the order of the case grid)."""
        return zlib.crc32(self.case_id.encode())

    def columns(self) -> dict:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "k": self.k,
            "n_factors": self.n_factors,
            "utility": self.utility,
        }


def build_cases(
    families: Sequence[str] = FAMILIES,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    factor_counts: Sequence[int] = DEFAULT_FACTOR_COUNTS,
    utilities: Sequence[str] = tuple(UTILITY_PROFILES),
) -> List[Case]:
    for family in families:
        if family not in FAMILIES:
            raise ValueError(f"Unknown family {family!r}; use one of {FAMILIES}.")
    for f in factor_counts:
        if f not in (0, 1, 2):
            raise ValueError("factor_counts must be drawn from (0, 1, 2).")
    for name in utilities:
        if name not in UTILITY_PROFILES:
            raise ValueError(f"Unknown utility profile {name!r}; use one of {tuple(UTILITY_PROFILES)}.")
    return [
        Case(family, int(k), int(f), name)
        for family in families
        for k in k_values
        for f in factor_counts
        for name in utilities
    ]


def utility_vector(profile: str, k: int) -> torch.Tensor:
    """Centered utility vector (float32, CPU); centering mirrors training.simulate_data."""
    v = UTILITY_PROFILES[profile][1](k).to(torch.float32)
    return v - v.mean()


def factor_omega(n_factors: int, k: int, uniqueness: float = 1.0) -> torch.Tensor:
    """
    Omega = G G' / f + uniqueness * I, as in training.simulate_data's factor branch (which
    divides by the number of factors). Column 1 of G rises linearly from 0 to 2; column 2
    falls linearly from 2 to 0.
    """
    km1 = k - 1
    omega = uniqueness * torch.eye(km1)
    if n_factors > 0:
        columns = [torch.linspace(0.0, 2.0, km1), torch.linspace(2.0, 0.0, km1)][:n_factors]
        loadings = torch.stack(columns, dim=1)
        omega = omega + loadings @ loadings.T / n_factors
    return omega


def case_inputs(case: Case) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    (v, Sigma, L, root) on the CPU in float32: v centered utilities (K,), Sigma the scale
    matrix (K, K), L the Cholesky factor of Omega (K-1, K-1), and root the square root
    used for simulation (training._compute_sqrt_L), so that U = v + root @ z.
    """
    v = utility_vector(case.utility, case.k)
    omega = factor_omega(case.n_factors, case.k)
    L = torch.linalg.cholesky(omega)
    sigma = omega_to_cov(omega.unsqueeze(0))[0]
    root, is_valid = _compute_sqrt_L(L.unsqueeze(0), gumbel=case.gumbel)
    if not bool(is_valid.all()):
        raise RuntimeError(f"Invalid square root for {case.case_id}.")
    return v, sigma, L, root[0]


# =============================================================================
# Random numbers, devices, timing helpers
# =============================================================================

def _derive_seed(*keys: int) -> int:
    """Deterministic 63-bit seed derived from non-negative integer keys."""
    state = np.random.SeedSequence([int(key) for key in keys]).generate_state(1, dtype=np.uint64)
    return int(state[0] % np.uint64(2**63 - 1))


def _draw_errors(family: str, shape, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    """Standardized errors z, drawn exactly as in training.simulate_data."""
    torch.manual_seed(seed)
    shape = tuple(int(s) for s in shape)
    if family == "gumbel":
        gumbel = torch.distributions.Gumbel(
            loc=torch.tensor(-EULER_MASCHERONI, device=device, dtype=dtype),
            scale=torch.tensor(1.0, device=device, dtype=dtype),
        )
        return gumbel.sample(shape)
    return torch.randn(shape, device=device, dtype=dtype)


def _hard_choices(u: torch.Tensor) -> torch.Tensor:
    """One-hot hard choices along the last axis, as in training.simulate_data."""
    return (u == u.max(dim=-1, keepdim=True).values).to(u.dtype)


class _PreserveRNGState:
    """Restore the global CPU and device RNG states on exit (the draws reseed them)."""

    def __init__(self, device):
        self.device = torch.device(device)

    def __enter__(self):
        self._cpu_state = torch.get_rng_state()
        self._device_state = None
        try:
            if self.device.type == "mps":
                self._device_state = torch.mps.get_rng_state()
            elif self.device.type == "cuda":
                self._device_state = torch.cuda.get_rng_state(self.device)
        except Exception:
            self._device_state = None
        return self

    def __exit__(self, *exc_info):
        torch.set_rng_state(self._cpu_state)
        if self._device_state is not None:
            try:
                if self.device.type == "mps":
                    torch.mps.set_rng_state(self._device_state)
                elif self.device.type == "cuda":
                    torch.cuda.set_rng_state(self._device_state, self.device)
            except Exception:
                pass
        return False


def _synchronize(device) -> None:
    device = torch.device(device)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _empty_cache(device) -> None:
    device = torch.device(device)
    try:
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _resolve_devices(devices: Sequence[str]) -> List[str]:
    resolved = []
    for name in devices:
        device = torch.device(name)
        if device.type == "mps" and not torch.backends.mps.is_available():
            warnings.warn("MPS is not available; skipping it.")
            continue
        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA is not available; skipping it.")
            continue
        if str(device) not in resolved:
            resolved.append(str(device))
    if not resolved:
        raise ValueError("No usable device was given.")
    return resolved


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """float64 NumPy copy. MPS has no float64, so move to the CPU before casting."""
    return tensor.detach().cpu().numpy().astype(np.float64)


class _NullBar:
    def update(self, n=1):
        pass

    def set_postfix_str(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _time_call(fn: Callable[[], object], device, repeats: int) -> Tuple[float, float]:
    """Median and minimum wall-clock seconds of fn() after one warm-up call."""
    fn()
    _synchronize(device)
    times = []
    for _ in range(max(1, int(repeats))):
        _synchronize(device)
        start = time.perf_counter()
        fn()
        _synchronize(device)
        times.append(time.perf_counter() - start)
    return float(np.median(times)), float(np.min(times))


# =============================================================================
# Emulators
# =============================================================================

def load_emulator(path: str, device) -> Tuple[torch.nn.Module, dict]:
    """Load a checkpoint with training.load_checkpoint and put it in eval mode (no dropout)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Emulator checkpoint not found: {path}")
    model, epoch, loss = load_checkpoint(path, device=torch.device(device))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    if model.training or any(module.training for module in model.modules()):
        raise RuntimeError(f"{path}: the emulator is not in eval mode.")
    info = {
        "path": path,
        "epoch": epoch,
        "loss": float(loss) if loss is not None else math.nan,
        "include_k": bool(getattr(model, "include_k", False)),
        "include_c_star": bool(getattr(model, "include_c_star", False)),
    }
    return model, info


def emulator_probabilities(model: torch.nn.Module, v: torch.Tensor, sigma: torch.Tensor, device) -> np.ndarray:
    """Emulator choice probabilities for one (v, Sigma), via utils.canonicalize."""
    with torch.no_grad():
        v_star, cov_star, c_star = canonicalize(v.unsqueeze(0).to(device), sigma.unsqueeze(0).to(device))
        probs = torch.softmax(model(v_star, cov_star, c_star), dim=-1)
    return _to_numpy(probs[0])


# =============================================================================
# Reference pass: hard frequencies and smoothed moments from the same draws
# =============================================================================

def _reference_case(
    case: Case,
    taus: Sequence[float],
    n_draws: int,
    seed: int,
    device,
    max_chunk_elements: int,
    pbar=None,
) -> pd.DataFrame:
    """
    One pass over n_draws utility draws. Returns one row per (alternative, tau) with the
    reference probability (hard frequency), its standard error, and for each tau the
    smoothed limit P_tau, the bias P_tau - P, its standard error, and Var(Y) per draw.
    tau = inf rows describe the hard-count simulator (bias 0, Var = P (1 - P)).
    """
    pbar = pbar if pbar is not None else _NullBar()
    v, _, _, root = case_inputs(case)
    k = case.k
    finite_taus = [float(t) for t in taus if math.isfinite(t)]
    v_dev = v.to(device)
    root_t = root.T.contiguous().to(device)
    rows_per_chunk = max(1, int(max_chunk_elements) // k)
    totals = np.zeros((1 + 3 * len(finite_taus), k))

    with torch.no_grad():
        for chunk_idx, start in enumerate(range(0, n_draws, rows_per_chunk)):
            n = min(rows_per_chunk, n_draws - start)
            z = _draw_errors(case.family, (n, k), _derive_seed(seed, _STREAM_REFERENCE, case.key, chunk_idx), device)
            u = z @ root_t  # U = v + Sigma^{1/2} z, as einsum('bij,bsj->bsi') in training.py
            del z
            u += v_dev
            hard = _hard_choices(u)
            parts = [hard.sum(dim=0)]
            for tau in finite_taus:
                y = torch.softmax(u * tau, dim=1)
                d = y - hard
                parts += [y.sum(dim=0), (y * y).sum(dim=0), (d * d).sum(dim=0)]
                del y, d
            del u, hard
            totals += _to_numpy(torch.stack(parts))
            pbar.update(n)
    return _reference_rows(case, finite_taus, totals, n_draws)


def _reference_rows(case: Case, finite_taus: Sequence[float], totals: np.ndarray, n_draws: int) -> pd.DataFrame:
    """
    Reference table from the accumulated totals: row 0 holds the hard counts, then for
    each finite tau the sums of Y, Y^2 and (Y - H)^2 over draws (H = hard one-hot).
    """
    k = case.k
    n_total = float(n_draws)
    counts = totals[0]
    p_ref = counts / n_total
    p_ref_se = np.sqrt(p_ref * (1.0 - p_ref) / n_total)
    base = case.columns()
    rows = []
    for j in range(k):
        rows.append({
            **base, "alt": j, "p_ref": p_ref[j], "p_ref_se": p_ref_se[j], "count": counts[j],
            "tau": HARD, "p_tau": p_ref[j], "bias": 0.0, "bias_se": 0.0,
            "var_y": p_ref[j] * (1.0 - p_ref[j]),
        })
    for i, tau in enumerate(finite_taus):
        sum_y, sum_y2, sum_d2 = totals[1 + 3 * i: 4 + 3 * i]
        p_tau = sum_y / n_total
        var_y = np.maximum(sum_y2 / n_total - p_tau ** 2, 0.0)
        bias = p_tau - p_ref
        var_d = np.maximum(sum_d2 / n_total - bias ** 2, 0.0)
        bias_se = np.sqrt(var_d / n_total)
        for j in range(k):
            rows.append({
                **base, "alt": j, "p_ref": p_ref[j], "p_ref_se": p_ref_se[j], "count": counts[j],
                "tau": tau, "p_tau": p_tau[j], "bias": bias[j], "bias_se": bias_se[j], "var_y": var_y[j],
            })
    return pd.DataFrame(rows)


# =============================================================================
# Nested replications: empirical check of the MSE formula, and log-scale errors
# =============================================================================

def _replicate_case(
    case: Case,
    taus: Sequence[float],
    r_grid: Sequence[int],
    n_reps: int,
    seed: int,
    device,
    max_chunk_elements: int,
    p_ref: np.ndarray,
    pbar=None,
) -> pd.DataFrame:
    """
    n_reps independent simulators, each read off at every R in r_grid (nested: the
    estimate at R uses the first R draws). Errors are measured against p_ref.
    """
    pbar = pbar if pbar is not None else _NullBar()
    v, _, _, root = case_inputs(case)
    k = case.k
    finite_taus = [float(t) for t in taus if math.isfinite(t)]
    all_taus = [HARD] + finite_taus
    r_grid = sorted({int(r) for r in r_grid})
    r_set = set(r_grid)
    r_max = r_grid[-1]
    step = max(1, int(max_chunk_elements) // (n_reps * k))
    bounds = sorted(set(range(step, r_max, step)) | r_set)
    v_dev = v.to(device)
    root_t = root.T.contiguous().to(device)
    device_key = zlib.crc32(str(device).encode())
    sums = torch.zeros(len(all_taus), n_reps, k, device=device)
    rows: List[dict] = []
    start = 0
    with torch.no_grad():
        for chunk_idx, end in enumerate(bounds):
            n = end - start
            seed_chunk = _derive_seed(seed, _STREAM_REPLICATIONS, device_key, case.key, chunk_idx)
            z = _draw_errors(case.family, (n_reps, n, k), seed_chunk, device)
            u = torch.matmul(z, root_t)
            del z
            u += v_dev
            sums[0] += _hard_choices(u).sum(dim=1)
            for i, tau in enumerate(finite_taus, start=1):
                sums[i] += torch.softmax(u * tau, dim=-1).sum(dim=1)
            del u
            pbar.update(n_reps * n)
            start = end
            if end not in r_set:
                continue
            rows += _replication_rows(case, str(device), all_taus, _to_numpy(sums / end), end, p_ref)
    return pd.DataFrame(rows)


def _replication_rows(
    case: Case, device: str, all_taus: Sequence[float], estimates: np.ndarray, num_samples: int, p_ref: np.ndarray
) -> List[dict]:
    """Summary rows for replicate estimates of shape (n_taus, n_reps, K) at one R."""
    p_ref = np.asarray(p_ref, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_errors = np.log(estimates) - np.log(p_ref)
    base = case.columns()
    rows = []
    for i, tau in enumerate(all_taus):
        est = estimates[i]
        for j in range(case.k):
            rows.append({
                **base, "device": device, "tau": tau, "num_samples": int(num_samples), "alt": j,
                "n_reps": est.shape[0],
                "sim_mean": float(est[:, j].mean()),
                "sim_mse": float(np.mean((est[:, j] - p_ref[j]) ** 2)),
                "sim_msle": float(np.mean(log_errors[i][:, j] ** 2)),
                "frac_zero": float(np.mean(est[:, j] == 0.0)),
            })
    return rows


# =============================================================================
# Timing: one likelihood pass over n_rows rows sharing one Sigma
# =============================================================================

def _timing_inputs(k: int, n_rows: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generic inputs for timing (run times do not depend on the values): utilities linear
    from 0 to -4, a two-factor Omega with distinct uniquenesses (so the Gumbel root's
    eigendecomposition is non-degenerate for the backward pass), and fixed choices.
    """
    v = torch.linspace(0.0, -4.0, k)
    v = v - v.mean()
    km1 = k - 1
    loadings = torch.stack([torch.linspace(0.0, 2.0, km1), torch.linspace(2.0, 0.0, km1)], dim=1)
    omega = loadings @ loadings.T / 2.0 + torch.diag(torch.linspace(0.5, 1.5, km1))
    L = torch.linalg.cholesky(omega)
    y = torch.arange(n_rows) % k
    return v.expand(n_rows, k).contiguous(), L, y


def _root_from_L(L: torch.Tensor, gumbel: bool) -> torch.Tensor:
    """Differentiable Sigma^{1/2} from the Cholesky factor of Omega (training._compute_sqrt_L)."""
    root, _ = _compute_sqrt_L(L.unsqueeze(0), gumbel=gumbel)
    return root[0]


def _emulator_pass_fns(model, V: torch.Tensor, L: torch.Tensor, y: torch.Tensor, device):
    """Forward and forward+backward likelihood passes with the emulator."""
    V_dev, L_dev, y_dev = V.to(device), L.to(device), y.to(device)
    n, k = V_dev.shape
    rows = torch.arange(n, device=device)

    def sigma_rows(L_):
        return omega_to_cov((L_ @ L_.T).unsqueeze(0)).expand(n, k, k)

    def forward():
        with torch.no_grad():
            v_star, cov_star, c_star = canonicalize(V_dev, sigma_rows(L_dev))
            return F.log_softmax(model(v_star, cov_star, c_star), dim=-1)

    def forward_backward():
        V_leaf = V_dev.clone().requires_grad_(True)
        L_leaf = L_dev.clone().requires_grad_(True)
        v_star, cov_star, c_star = canonicalize(V_leaf, sigma_rows(L_leaf))
        log_probs = F.log_softmax(model(v_star, cov_star, c_star), dim=-1)
        loss = -log_probs[rows, y_dev].sum()
        loss.backward()
        return loss.detach()

    return forward, forward_backward


def _simulator_pass_fns(
    family: str,
    V: torch.Tensor,
    L: torch.Tensor,
    y: torch.Tensor,
    device,
    tau: float,
    num_samples: int,
    seed: int,
    max_chunk_elements: int,
):
    """
    Forward and forward+backward likelihood passes with the smoothed simulator: errors are
    drawn on the fly for each chunk of rows, then log P_j = logsumexp over draws of
    log_softmax(tau U)_j minus log R.
    """
    gumbel = family == "gumbel"
    V_dev, y_dev = V.to(device), y.to(device)
    n, k = V_dev.shape
    rows_per_chunk = max(1, int(max_chunk_elements) // (num_samples * k))
    chunks = [(s, min(s + rows_per_chunk, n)) for s in range(0, n, rows_per_chunk)]
    log_r = math.log(num_samples)

    def chunk_log_probs(root, V_rows, chunk_idx):
        z = _draw_errors(family, (V_rows.shape[0], num_samples, k), _derive_seed(seed, _STREAM_TIMING, chunk_idx), device)
        u = torch.matmul(z, root.T) + V_rows.unsqueeze(1)
        return torch.logsumexp(F.log_softmax(tau * u, dim=-1), dim=1) - log_r

    def forward():
        with torch.no_grad():
            root = _root_from_L(L, gumbel).to(device)  # linear algebra on the CPU (no MPS kernels)
            return torch.cat([chunk_log_probs(root, V_dev[s:e], c) for c, (s, e) in enumerate(chunks)])

    def forward_backward():
        V_leaf = V_dev.clone().requires_grad_(True)
        L_leaf = L.clone().requires_grad_(True)  # CPU leaf: the Gumbel root needs eigh
        total = torch.zeros((), device=device)
        for c, (s, e) in enumerate(chunks):
            root = _root_from_L(L_leaf, gumbel).to(device)
            log_probs = chunk_log_probs(root, V_leaf[s:e], c)
            loss = -log_probs.gather(1, y_dev[s:e].unsqueeze(1)).sum()
            loss.backward()  # per chunk, so memory stays bounded; gradients accumulate
            total = total + loss.detach()
        return total

    return forward, forward_backward


# =============================================================================
# Stage caching
# =============================================================================

def _normalize(settings: dict) -> dict:
    return json.loads(json.dumps(settings))


def _stage(name: str, settings: dict, output_dir: str, rerun: bool, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Load a stage's table if it was saved with the same settings; otherwise compute and save it."""
    csv_path = os.path.join(output_dir, f"{name}.csv")
    json_path = os.path.join(output_dir, f"{name}_settings.json")
    settings = _normalize(settings)
    if not rerun and os.path.exists(csv_path) and os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as handle:
            saved = json.load(handle)
        if saved == settings:
            print(f"[{name}] loaded from {csv_path}")
            return pd.read_csv(csv_path)
        print(f"[{name}] saved settings differ; recomputing.")
    table = compute()
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(settings, handle, indent=2)
    return table


# =============================================================================
# Main entry point
# =============================================================================

def run_comparison(
    emulator_paths: Mapping[str, str] = DEFAULT_EMULATOR_PATHS,
    families: Sequence[str] = FAMILIES,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    factor_counts: Sequence[int] = DEFAULT_FACTOR_COUNTS,
    utilities: Sequence[str] = tuple(UTILITY_PROFILES),
    taus: Sequence[float] = DEFAULT_TAUS,
    n_reference: int = DEFAULT_N_REFERENCE,
    reference_device: Optional[str] = None,
    devices: Sequence[str] = ("cpu", "mps"),
    n_reps: int = DEFAULT_N_REPS,
    rep_r_grid: Sequence[int] = DEFAULT_REP_R_GRID,
    timing_rows: int = DEFAULT_TIMING_ROWS,
    timing_r_grid: Sequence[int] = DEFAULT_TIMING_R_GRID,
    timing_repeats: int = DEFAULT_TIMING_REPEATS,
    timing_tau: float = DEFAULT_TIMING_TAU,
    run_timing: bool = True,
    seed: int = 0,
    max_chunk_elements: int = DEFAULT_MAX_CHUNK_ELEMENTS,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    rerun: bool = False,
) -> Dict[str, object]:
    """
    Run the four stages and return their tables:

      emulator      emulator probabilities per case, on every device (instant)
      reference     1e8-draw hard frequencies plus smoothed moments per tau, on reference_device
      replications  nested simulator replications per case, on every device
      timing        seconds per 1,000-row likelihood pass, per device, family and K

    The reference is computed once (on MPS if available) and shared by both devices.
    """
    devices = _resolve_devices(devices)
    reference_device = str(torch.device(reference_device)) if reference_device else _default_device()
    reference_device = _resolve_devices([reference_device])[0]
    taus = sorted({float(t) for t in taus if math.isfinite(t)})
    cases = build_cases(families, k_values, factor_counts, utilities)
    emulator_paths = {family: emulator_paths[family] for family in families}
    case_grid = {
        "families": list(families), "k_values": [int(k) for k in k_values],
        "factor_counts": [int(f) for f in factor_counts], "utilities": list(utilities),
    }
    print(
        f"{len(cases)} cases ({', '.join(families)}; K in {tuple(k_values)}; factors in "
        f"{tuple(factor_counts)}; {len(utilities)} utility profiles). Devices: {', '.join(devices)}; "
        f"reference on {reference_device}."
    )

    # --- Emulators (fast; also fails early if a checkpoint is missing) -------------------
    emulators: Dict[Tuple[str, str], torch.nn.Module] = {}
    info_rows = []
    for device in devices:
        for family in families:
            model, info = load_emulator(emulator_paths[family], device)
            emulators[(device, family)] = model
            info_rows.append({"device": device, "family": family, **info})
    emulator_info = pd.DataFrame(info_rows)
    for row in emulator_info.drop_duplicates("family").itertuples(index=False):
        if row.include_c_star != (row.family == "gumbel"):
            warnings.warn(
                f"{os.path.basename(row.path)}: include_c_star={row.include_c_star}, but Gumbel emulators "
                "normally use c_star and probit emulators do not. Check the family-to-path mapping."
            )
        print(
            f"Loaded {row.family} emulator {os.path.basename(row.path)} (epoch {row.epoch}, "
            f"loss {row.loss:.4g}, include_k={row.include_k}, include_c_star={row.include_c_star}); eval mode."
        )
    emu_rows = []
    for device in devices:
        for case in cases:
            v, sigma, _, _ = case_inputs(case)
            probs = emulator_probabilities(emulators[(device, case.family)], v, sigma, device)
            for j, p in enumerate(probs):
                emu_rows.append({**case.columns(), "device": device, "alt": j, "p_emu": float(p)})
    emulator_table = pd.DataFrame(emu_rows)

    # --- Reference and smoothed moments -------------------------------------------------
    reference_settings = {
        **case_grid, "taus": taus, "n_reference": int(n_reference), "seed": int(seed),
        "reference_device": reference_device, "max_chunk_elements": int(max_chunk_elements),
    }

    def compute_reference():
        tables = []
        bar = tqdm(total=len(cases) * int(n_reference), desc=f"Reference ({reference_device})", unit="draw", unit_scale=True)
        with _PreserveRNGState(reference_device):
            for case in cases:
                bar.set_postfix_str(case.case_id, refresh=False)
                tables.append(_reference_case(case, taus, int(n_reference), seed, reference_device, max_chunk_elements, bar))
                _empty_cache(reference_device)
        bar.close()
        return pd.concat(tables, ignore_index=True)

    reference = _stage("reference", reference_settings, output_dir, rerun, compute_reference)

    # --- Nested replications, per device --------------------------------------------------
    replication_settings = {
        "reference": reference_settings, "devices": devices, "n_reps": int(n_reps),
        "rep_r_grid": sorted(int(r) for r in rep_r_grid),
    }
    p_ref_by_case = {
        case_id: group.sort_values("alt")["p_ref"].to_numpy(float)
        for case_id, group in reference[reference["tau"] == HARD].groupby("case_id")
    }

    def compute_replications():
        tables = []
        r_max = max(int(r) for r in rep_r_grid)
        for device in devices:
            bar = tqdm(total=len(cases) * int(n_reps) * r_max, desc=f"Replications ({device})", unit="draw", unit_scale=True)
            with _PreserveRNGState(device):
                for case in cases:
                    bar.set_postfix_str(case.case_id, refresh=False)
                    tables.append(_replicate_case(
                        case, taus, rep_r_grid, int(n_reps), seed, device, max_chunk_elements,
                        p_ref_by_case[case.case_id], bar,
                    ))
                    _empty_cache(device)
            bar.close()
        return pd.concat(tables, ignore_index=True)

    replications = _stage("replications", replication_settings, output_dir, rerun, compute_replications)

    # --- Timing, per device -------------------------------------------------------------------
    timing = pd.DataFrame()
    if run_timing:
        timing_settings = {
            "families": list(families), "k_values": [int(k) for k in k_values], "devices": devices,
            "timing_rows": int(timing_rows), "timing_r_grid": sorted(int(r) for r in timing_r_grid),
            "timing_repeats": int(timing_repeats), "timing_tau": float(timing_tau), "seed": int(seed),
            "max_chunk_elements": int(max_chunk_elements), "emulator_paths": emulator_paths,
        }

        def compute_timing():
            rows = []
            n_jobs = len(devices) * len(families) * len(k_values) * (2 + 2 * len(timing_r_grid))
            bar = tqdm(total=n_jobs, desc="Timing", unit="job")
            for device in devices:
                with _PreserveRNGState(device):
                    for family in families:
                        model = emulators[(device, family)]
                        for k in k_values:
                            V, L, y = _timing_inputs(int(k), int(timing_rows))
                            common = {"device": device, "family": family, "k": int(k), "n_rows": int(timing_rows)}
                            fwd, fwd_bwd = _emulator_pass_fns(model, V, L, y, device)
                            for pass_name, fn in (("forward", fwd), ("forward_backward", fwd_bwd)):
                                bar.set_postfix_str(f"{device} {family} K={k} emulator {pass_name}", refresh=False)
                                median, fastest = _time_call(fn, device, timing_repeats)
                                rows.append({**common, "method": "emulator", "pass": pass_name,
                                             "num_samples": np.nan, "seconds": median, "seconds_min": fastest})
                                bar.update(1)
                            for r in sorted(int(r) for r in timing_r_grid):
                                fwd, fwd_bwd = _simulator_pass_fns(
                                    family, V, L, y, device, float(timing_tau), r, seed, max_chunk_elements
                                )
                                for pass_name, fn in (("forward", fwd), ("forward_backward", fwd_bwd)):
                                    bar.set_postfix_str(f"{device} {family} K={k} R={r:,} {pass_name}", refresh=False)
                                    median, fastest = _time_call(fn, device, timing_repeats)
                                    rows.append({**common, "method": "simulator", "pass": pass_name,
                                                 "num_samples": r, "seconds": median, "seconds_min": fastest})
                                    bar.update(1)
                                _empty_cache(device)
            bar.close()
            return pd.DataFrame(rows)

        timing = _stage("timing", timing_settings, output_dir, rerun, compute_timing)

    settings = {
        **reference_settings, "devices": devices, "n_reps": int(n_reps),
        "rep_r_grid": sorted(int(r) for r in rep_r_grid), "timing_rows": int(timing_rows),
        "timing_r_grid": sorted(int(r) for r in timing_r_grid), "timing_tau": float(timing_tau),
        "emulator_paths": emulator_paths, "output_dir": output_dir,
    }
    os.makedirs(output_dir, exist_ok=True)
    emulator_table.to_csv(os.path.join(output_dir, "emulator.csv"), index=False)
    emulator_info.to_csv(os.path.join(output_dir, "emulator_info.csv"), index=False)
    return {
        "reference": reference,
        "emulator": emulator_table,
        "emulator_info": emulator_info,
        "replications": replications,
        "timing": timing,
        "settings": settings,
    }


# =============================================================================
# Analysis
# =============================================================================

PROBABILITY_BINS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
PROBABILITY_BIN_LABELS = ("< 0.001", "0.001–0.01", "0.01–0.1", "≥ 0.1")


def _tau_label(tau: float) -> str:
    return "hard counts (τ = ∞)" if not math.isfinite(tau) else f"τ = {tau:g}"


def _tau_tex(tau: float) -> str:
    """LaTeX-safe tau label for figures (config_plt turns on usetex)."""
    return r"hard counts ($\tau = \infty$)" if not math.isfinite(tau) else rf"$\tau = {tau:g}$"


def _tau_colors(taus: Sequence[float]) -> Dict[float, object]:
    """Finite temperatures take the paper's palette in order; hard counts are gray."""
    finite = sorted(t for t in taus if math.isfinite(t))
    colors = {t: PAPER_COLORS[i % len(PAPER_COLORS)] for i, t in enumerate(finite)}
    colors[HARD] = "#666666"
    return colors


DEVICE_COLORS = {"cpu": PAPER_COLORS[0], "mps": PAPER_COLORS[1], "cuda": PAPER_COLORS[2]}
DEVICE_TEX = {"cpu": "CPU", "mps": "MPS", "cuda": "CUDA"}


def _break_even(var_y, error_sq, bias_sq) -> np.ndarray:
    """R* = Var(Y) / (e^2 - bias^2), or infinity when the simulator's floor bias^2 >= e^2."""
    var_y = np.asarray(var_y, dtype=float)
    gap = np.asarray(error_sq, dtype=float) - np.asarray(bias_sq, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(gap > 0, var_y / np.where(gap > 0, gap, 1.0), np.inf)


def _break_even_status(var_y, raw_sq, ref_var, bias_sq) -> Tuple[np.ndarray, np.ndarray]:
    """
    Break-even R* from the emulator's raw squared error against the reference (raw_sq) and
    the reference's variance (ref_var):

      finite       emulator error resolved (raw_sq > 4 ref_var): R* from the noise-corrected
                   squared error raw_sq - ref_var
      lower bound  emulator error within reference noise: R* computed from an upper bound on
                   the emulator's error, (|e| + 2 SE)^2, so the true R* is at least this
      never        the simulator's floor bias^2 is at least the emulator's squared error
                   (or its upper bound, when unresolved)
    """
    raw_sq = np.asarray(raw_sq, dtype=float)
    ref_var = np.asarray(ref_var, dtype=float)
    resolved = raw_sq > 4.0 * ref_var
    point = _break_even(var_y, np.maximum(raw_sq - ref_var, 0.0), bias_sq)
    bound = _break_even(var_y, (np.sqrt(raw_sq) + 2.0 * np.sqrt(ref_var)) ** 2, bias_sq)
    r_star = np.where(resolved, point, bound)
    status = np.where(np.isinf(r_star), "never", np.where(resolved, "finite", "lower bound"))
    return r_star, status


def _primary_device(results: Mapping[str, object]) -> str:
    devices = list(results["emulator"]["device"].unique())
    preferred = results.get("settings", {}).get("reference_device")
    return preferred if preferred in devices else devices[0]


def alternative_table(results: Mapping[str, object], device: Optional[str] = None) -> pd.DataFrame:
    """
    One row per (case, alternative, tau): reference, emulator error, smoothing bias,
    per-draw variance, and the per-alternative break-even R*.
    """
    device = device or _primary_device(results)
    emu = results["emulator"]
    emu = emu[emu["device"] == device][["case_id", "alt", "p_emu"]]
    table = results["reference"].merge(emu, on=["case_id", "alt"], how="inner")
    table["emu_error"] = table["p_emu"] - table["p_ref"]
    table["emu_sq_error"] = np.maximum(table["emu_error"] ** 2 - table["p_ref_se"] ** 2, 0.0)
    table["emu_resolved"] = table["emu_error"].abs() > 2.0 * table["p_ref_se"]
    table["emu_rel_error"] = table["emu_error"] / table["p_ref"]
    table["bias_sq"] = table["bias"] ** 2
    table["rel_bias"] = table["bias"] / table["p_ref"]
    table["r_star"], table["r_star_status"] = _break_even_status(
        table["var_y"], table["emu_error"] ** 2, table["p_ref_se"] ** 2, table["bias_sq"]
    )
    table["p_bin"] = pd.cut(
        table["p_ref"], bins=list(PROBABILITY_BINS), labels=list(PROBABILITY_BIN_LABELS), include_lowest=True
    )
    return table


def case_table(alt: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (case, tau). Squared errors are averaged over alternatives; R* is where the
    simulator's mean-over-alternatives MSE equals the emulator's.
    """
    work = alt.assign(
        ref_var=alt["p_ref_se"] ** 2,
        emu_raw_sq=alt["emu_error"] ** 2,
        emu_abs_error=alt["emu_error"].abs(),
        emu_abs_rel_error=alt["emu_rel_error"].abs(),
    )
    table = work.groupby(_CASE_COLUMNS + ["tau"], sort=True).agg(
        p_min=("p_ref", "min"),
        emu_mse=("emu_sq_error", "mean"),
        emu_raw_mse=("emu_raw_sq", "mean"),
        ref_var=("ref_var", "mean"),
        floor_mse=("bias_sq", "mean"),
        var_y=("var_y", "mean"),
        emu_max_abs_error=("emu_abs_error", "max"),
        emu_max_rel_error=("emu_abs_rel_error", "max"),
    ).reset_index()
    table["emu_rms"] = np.sqrt(table["emu_mse"])
    table["floor_rms"] = np.sqrt(table["floor_mse"])
    table["ref_se_rms"] = np.sqrt(table["ref_var"])
    table["emu_resolved"] = table["emu_raw_mse"] > 4.0 * table["ref_var"]
    table["r_star"], table["r_star_status"] = _break_even_status(
        table["var_y"], table["emu_raw_mse"], table["ref_var"], table["floor_mse"]
    )
    return table


def aggregate_table(cases: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (family, K, tau), averaging squared errors over the cases (i.e. RMS error
    over cases). R*_agg is where the averaged curves cross.
    """
    table = cases.groupby(["family", "k", "tau"], sort=True).agg(
        n_cases=("case_id", "nunique"),
        emu_mse=("emu_mse", "mean"),
        emu_raw_mse=("emu_raw_mse", "mean"),
        floor_mse=("floor_mse", "mean"),
        var_y=("var_y", "mean"),
        ref_var=("ref_var", "mean"),
        n_never=("r_star_status", lambda s: int((s == "never").sum())),
        n_lower_bound=("r_star_status", lambda s: int((s == "lower bound").sum())),
        median_case_r_star=("r_star", "median"),
        n_unresolved=("emu_resolved", lambda s: int((~s.astype(bool)).sum())),
    ).reset_index()
    table["emu_rms"] = np.sqrt(table["emu_mse"])
    table["floor_rms"] = np.sqrt(table["floor_mse"])
    table["ref_se_rms"] = np.sqrt(table["ref_var"])
    table["r_star"], table["r_star_status"] = _break_even_status(
        table["var_y"], table["emu_raw_mse"], table["ref_var"], table["floor_mse"]
    )
    return table


def case_wide(cases: pd.DataFrame) -> pd.DataFrame:
    """Readable per-case summary: emulator accuracy and R* for each tau and the best tau."""
    base = cases.drop_duplicates("case_id").set_index("case_id")[
        ["family", "k", "n_factors", "utility", "p_min", "emu_rms", "emu_max_abs_error", "emu_max_rel_error", "emu_resolved"]
    ]
    r_star = cases.pivot_table(index="case_id", columns="tau", values="r_star", aggfunc="first")
    r_star.columns = [f"R* {_tau_label(t)}" for t in r_star.columns]
    smoothed = cases[np.isfinite(cases["tau"])]
    best = smoothed.loc[smoothed.groupby("case_id")["r_star"].idxmin(), ["case_id", "tau", "r_star"]].set_index("case_id")
    best.columns = ["best_tau", "R* best smoothed τ"]
    wide = base.join(r_star).join(best).reset_index()
    return wide.sort_values(["family", "k", "n_factors", "utility"]).reset_index(drop=True)


def probability_bin_table(alt: pd.DataFrame) -> pd.DataFrame:
    """By family, probability bin and tau: relative errors and break-even R* per alternative."""
    work = alt.assign(
        emu_abs_rel_error=alt["emu_rel_error"].abs(),
        never=alt["r_star_status"] == "never",
        lower_bound=alt["r_star_status"] == "lower bound",
    )
    return work.groupby(["family", "p_bin", "tau"], observed=True, sort=True).agg(
        n_alternatives=("alt", "size"),
        emu_median_abs_rel_error=("emu_abs_rel_error", "median"),
        median_rel_bias=("rel_bias", "median"),
        median_r_star=("r_star", "median"),
        frac_never=("never", "mean"),
        frac_lower_bound=("lower_bound", "mean"),
    ).reset_index()


def validation_table(results: Mapping[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Empirical MSE from the replications against the exact formula bias^2 + Var(Y) / R.
    Returns (row-level table, summary by device and tau).
    """
    reps = results["replications"]
    ref = results["reference"][["case_id", "alt", "tau", "p_ref", "bias", "var_y"]]
    merged = reps.merge(ref, on=["case_id", "alt", "tau"], how="inner")
    merged["theory_mse"] = merged["bias"] ** 2 + merged["var_y"] / merged["num_samples"]
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["ratio"] = merged["sim_mse"] / merged["theory_mse"]
    # Case-level ratio (sum over alternatives) is far less noisy than per-alternative ratios.
    per_case = merged.groupby(["device", "case_id", "family", "k", "tau", "num_samples"], sort=True).agg(
        sim_mse=("sim_mse", "sum"), theory_mse=("theory_mse", "sum")
    ).reset_index()
    per_case["ratio"] = per_case["sim_mse"] / per_case["theory_mse"]
    summary = per_case.groupby(["device", "tau"], sort=True)["ratio"].describe(percentiles=[0.05, 0.5, 0.95])
    summary = summary[["count", "5%", "50%", "95%"]].rename(
        columns={"count": "n", "5%": "p05", "50%": "median", "95%": "p95"}
    ).reset_index()
    return merged, summary


def log_scale_table(results: Mapping[str, object], alt: pd.DataFrame) -> pd.DataFrame:
    """
    Secondary metric: probability-weighted RMS log error, sqrt(sum_j P_j (log Q_j - log P_j)^2),
    i.e. the RMS error of one observation's log-likelihood contribution, RMS over cases.
    Hard counts give inf whenever a replicate has a zero count.
    """
    reps = results["replications"].merge(
        results["reference"][results["reference"]["tau"] == HARD][["case_id", "alt", "p_ref"]],
        on=["case_id", "alt"], how="inner",
    )
    reps["weighted_msle"] = reps["p_ref"] * reps["sim_msle"]
    per_case = reps.groupby(["device", "case_id", "family", "k", "tau", "num_samples"], sort=True)["weighted_msle"].sum().reset_index()
    sim = per_case.groupby(["device", "family", "k", "tau", "num_samples"], sort=True)["weighted_msle"].mean().reset_index()
    sim["sim_rms_log_error"] = np.sqrt(sim["weighted_msle"])
    emu_alt = alt[alt["tau"] == HARD]
    emu_case = (emu_alt["p_ref"] * (np.log(emu_alt["p_emu"]) - np.log(emu_alt["p_ref"])) ** 2).groupby(emu_alt["case_id"]).sum()
    emu = emu_alt.drop_duplicates("case_id")[["case_id", "family", "k"]].assign(
        emu_weighted_msle=lambda d: d["case_id"].map(emu_case)
    )
    emu = emu.groupby(["family", "k"], sort=True)["emu_weighted_msle"].mean().reset_index()
    emu["emu_rms_log_error"] = np.sqrt(emu["emu_weighted_msle"])
    table = sim.merge(emu[["family", "k", "emu_rms_log_error"]], on=["family", "k"], how="left")
    return table.drop(columns=["weighted_msle"])


def device_agreement(results: Mapping[str, object]) -> pd.DataFrame:
    """Largest emulator probability difference between devices, by family and K."""
    emu = results["emulator"]
    wide = emu.pivot_table(index=["case_id", "family", "k", "alt"], columns="device", values="p_emu").reset_index()
    devices = [d for d in emu["device"].unique()]
    if len(devices) < 2:
        return pd.DataFrame()
    wide["max_abs_diff"] = wide[devices].max(axis=1) - wide[devices].min(axis=1)
    return wide.groupby(["family", "k"], sort=True)["max_abs_diff"].max().reset_index()


def _time_at(r_values, r_measured, t_measured) -> np.ndarray:
    """
    Seconds at simulation size R: log-log interpolation inside the measured range, the
    smallest measurement below it, and linear extrapolation in R above it (from the last
    two points), since the cost per draw is constant at large R.
    """
    r_values = np.asarray(r_values, dtype=float)
    order = np.argsort(r_measured)
    r_m = np.asarray(r_measured, dtype=float)[order]
    t_m = np.asarray(t_measured, dtype=float)[order]
    out = np.full(r_values.shape, np.nan)
    finite = np.isfinite(r_values)
    inside = finite & (r_values <= r_m[-1])
    out[inside] = np.exp(np.interp(np.log(np.maximum(r_values[inside], r_m[0])), np.log(r_m), np.log(t_m)))
    above = finite & (r_values > r_m[-1])
    if len(r_m) >= 2:
        slope = max((t_m[-1] - t_m[-2]) / (r_m[-1] - r_m[-2]), 0.0)
    else:
        slope = t_m[-1] / r_m[-1]
    out[above] = t_m[-1] + slope * (r_values[above] - r_m[-1])
    out[~finite] = np.inf
    return out


def break_even_summary(aggregate: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    """
    Per (family, K, tau): aggregated R*, cases that never break even, and per device and pass
    the simulator's seconds at R* and its ratio to the emulator's seconds (1,000 rows).
    """
    table = aggregate[[
        "family", "k", "tau", "n_cases", "emu_rms", "floor_rms", "r_star", "r_star_status",
        "median_case_r_star", "n_never", "n_lower_bound",
    ]].copy()
    if timing is None or timing.empty:
        return table
    for (device, pass_name), group in timing.groupby(["device", "pass"], sort=True):
        seconds, ratios, emus = [], [], []
        for row in table.itertuples(index=False):
            sub = group[(group["family"] == row.family) & (group["k"] == row.k)]
            sim = sub[sub["method"] == "simulator"]
            emu = sub[sub["method"] == "emulator"]
            if sim.empty or emu.empty:
                seconds.append(np.nan)
                ratios.append(np.nan)
                emus.append(np.nan)
                continue
            t_sim = float(_time_at([row.r_star], sim["num_samples"], sim["seconds"])[0])
            t_emu = float(emu["seconds"].iloc[0])
            seconds.append(t_sim)
            emus.append(t_emu)
            ratios.append(t_sim / t_emu if np.isfinite(t_sim) else np.inf)
        table[f"{device} {pass_name}: emulator s"] = emus
        table[f"{device} {pass_name}: simulator s at R*"] = seconds
        table[f"{device} {pass_name}: ratio"] = ratios
    return table


# ---- Figures ---------------------------------------------------------------------------------

def _finish(fig, name: str, output_dir: Optional[str], show: bool, figures: Dict[str, plt.Figure]) -> None:
    fig.tight_layout()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    figures[name] = fig
    if show:
        plt.show()
    else:
        plt.close(fig)


def _log_ticks(axis) -> None:
    axis.set_major_locator(LogLocator(base=10.0, numticks=12))
    axis.set_minor_formatter(NullFormatter())


def _grid(families, k_values, width=4.6, height=3.7, **kwargs):
    fig, axes = plt.subplots(len(families), len(k_values), figsize=(width * len(k_values), height * len(families)), squeeze=False, **kwargs)
    return fig, axes


def plot_error_vs_r(aggregate: pd.DataFrame, validation_rows: pd.DataFrame, device: str):
    """RMS probability error against R: exact curves per tau, replication points, emulator line."""
    families = [f for f in FAMILIES if f in set(aggregate["family"])]
    k_values = sorted(aggregate["k"].unique())
    taus = sorted(aggregate["tau"].unique())
    colors = _tau_colors(taus)
    r_dense = np.geomspace(1.0, 1e9, 300)
    reps = validation_rows[validation_rows["device"] == device]
    emp = reps.groupby(["family", "k", "tau", "num_samples"], sort=True)["sim_mse"].mean().reset_index()
    fig, axes = _grid(families, k_values)
    for i, family in enumerate(families):
        for j, k in enumerate(k_values):
            ax = axes[i][j]
            sub = aggregate[(aggregate["family"] == family) & (aggregate["k"] == k)]
            if sub.empty:
                ax.set_visible(False)
                continue
            emu_rms = float(sub["emu_rms"].iloc[0])
            for row in sub.itertuples(index=False):
                color = colors[row.tau]
                ls = ":" if not math.isfinite(row.tau) else "-"
                ax.plot(r_dense, np.sqrt(row.floor_mse + row.var_y / r_dense), ls=ls, lw=1.4, color=color, label=_tau_tex(row.tau))
                points = emp[(emp["family"] == family) & (emp["k"] == k) & (emp["tau"] == row.tau)]
                ax.plot(points["num_samples"], np.sqrt(points["sim_mse"]), "o", ms=3.5, color=color)
                if np.isfinite(row.r_star):
                    ax.plot([row.r_star], [emu_rms], marker="v", ms=8, color=color, ls="none")
            ax.axhline(emu_rms, color="black", lw=2.2, label="emulator")
            ax.axhline(float(sub["ref_se_rms"].iloc[0]), color="black", lw=0.8, ls="--", label="reference SE")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _log_ticks(ax.xaxis)
            ax.set_title(f"{family}, $K = {k}$ ({int(sub['n_cases'].iloc[0])} cases)", fontsize=10)
            ax.set_xlabel("simulation draws $R$")
            if j == 0:
                ax.set_ylabel("RMS probability error\n(RMS over alternatives and cases)")
            if i == 0 and j == len(k_values) - 1:
                ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(
        "Smoothed simulator vs emulator (probability scale). Lines: exact $\\sqrt{b^2 + \\sigma^2/R}$; "
        f"dots: replications ({device}); triangles: break-even $R$.",
        fontsize=10,
    )
    return fig


def plot_r_star_vs_probability(alt: pd.DataFrame):
    """Per-alternative break-even R* against the alternative's probability; 'never' at the top."""
    families = [f for f in FAMILIES if f in set(alt["family"])]
    taus = sorted(alt["tau"].unique())
    k_values = sorted(alt["k"].unique())
    markers = dict(zip(k_values, ["o", "s", "^", "D", "v", "P"]))
    finite = alt["r_star"][np.isfinite(alt["r_star"])]
    top = 10 ** math.ceil(math.log10(finite.max() * 10)) if len(finite) else 1e10
    fig, axes = plt.subplots(len(families), len(taus), figsize=(3.6 * len(taus), 3.4 * len(families)), squeeze=False, sharey=True)
    for i, family in enumerate(families):
        for j, tau in enumerate(taus):
            ax = axes[i][j]
            sub = alt[(alt["family"] == family) & (alt["tau"] == tau)]
            for n_k, k in enumerate(k_values):
                s = sub[sub["k"] == k]
                color = f"C{n_k}"
                y = np.where(np.isfinite(s["r_star"]), s["r_star"], top)
                bound = (s["r_star_status"] == "lower bound").to_numpy()
                ax.scatter(s["p_ref"][~bound], y[~bound], s=14, marker=markers[k], alpha=0.7, color=color, label=f"$K = {k}$")
                if bound.any():
                    ax.scatter(s["p_ref"][bound], y[bound], s=18, marker=markers[k], facecolors="none", edgecolors=color, alpha=0.8)
            ax.axhspan(top / 3.0, top * 3.0, color="red", alpha=0.08)
            ax.text(0.02, 0.97, "never", transform=ax.transAxes, fontsize=8, color="darkred", va="top")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(top=top * 3.0)
            ax.set_title(f"{family}, {_tau_tex(tau)}", fontsize=10)
            ax.set_xlabel(r"reference probability $P_j$")
            if j == 0:
                ax.set_ylabel(r"break-even $R^*_j$")
            if i == 0 and j == len(taus) - 1:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Draws needed to match the emulator, per alternative. Hollow markers are lower bounds "
        "(emulator error within the reference's noise).",
        fontsize=10,
    )
    return fig


def plot_frontier(aggregate: pd.DataFrame, timing: pd.DataFrame, pass_name: str = "forward"):
    """RMS probability error against seconds per 1,000-row pass, per device: tau = 5 and best tau."""
    families = [f for f in FAMILIES if f in set(aggregate["family"])]
    k_values = sorted(aggregate["k"].unique())
    r_dense = np.geomspace(10.0, 1e9, 300)
    reference_tau = min(t for t in aggregate["tau"].unique() if math.isfinite(t))
    fig, axes = _grid(families, k_values)
    timing = timing[timing["pass"] == pass_name]
    for i, family in enumerate(families):
        for j, k in enumerate(k_values):
            ax = axes[i][j]
            sub = aggregate[(aggregate["family"] == family) & (aggregate["k"] == k)]
            if sub.empty:
                ax.set_visible(False)
                continue
            curves = {
                row.tau: np.sqrt(row.floor_mse + row.var_y / r_dense)
                for row in sub.itertuples(index=False) if math.isfinite(row.tau)
            }
            best = np.min(np.vstack(list(curves.values())), axis=0)  # best smoothed tau at each R
            emu_rms = float(sub["emu_rms"].iloc[0])
            for device, group in timing[(timing["family"] == family) & (timing["k"] == k)].groupby("device"):
                sim = group[group["method"] == "simulator"]
                emu = group[group["method"] == "emulator"]
                if sim.empty or emu.empty:
                    continue
                seconds = _time_at(r_dense, sim["num_samples"], sim["seconds"])
                color = DEVICE_COLORS.get(device, None)
                ax.plot(seconds, curves[reference_tau], ls="--", lw=1.2, color=color, label=f"{DEVICE_TEX.get(device, device)}: simulator, {_tau_tex(reference_tau)}")
                ax.plot(seconds, best, ls="-", lw=1.6, color=color, label=f"{DEVICE_TEX.get(device, device)}: simulator, best smoothed $\\tau$")
                ax.plot([float(emu["seconds"].iloc[0])], [emu_rms], marker="*", ms=14, color=color, ls="none", markeredgecolor="black", label=f"{DEVICE_TEX.get(device, device)}: emulator")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _log_ticks(ax.xaxis)
            ax.set_title(f"{family}, $K = {k}$", fontsize=10)
            ax.set_xlabel(f"seconds per 1,000-row pass ({pass_name.replace('_', ' + ')})")
            if j == 0:
                ax.set_ylabel("RMS probability error")
            if i == 0 and j == len(k_values) - 1:
                ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Accuracy against run time (lower left is better). Times above the largest timed $R$ are extrapolated linearly.", fontsize=10)
    return fig


def plot_validation(validation_rows: pd.DataFrame):
    """Replication MSE against the exact formula, per device (case-level sums over alternatives)."""
    devices = list(validation_rows["device"].unique())
    taus = sorted(validation_rows["tau"].unique())
    colors = _tau_colors(taus)
    per_case = validation_rows.groupby(["device", "case_id", "tau", "num_samples"], sort=True).agg(
        sim_mse=("sim_mse", "sum"), theory_mse=("theory_mse", "sum")
    ).reset_index()
    fig, axes = plt.subplots(1, len(devices), figsize=(4.6 * len(devices), 4.2), squeeze=False)
    for j, device in enumerate(devices):
        ax = axes[0][j]
        sub = per_case[per_case["device"] == device]
        for tau in taus:
            s = sub[sub["tau"] == tau]
            ax.scatter(s["theory_mse"], s["sim_mse"], s=8, alpha=0.6, color=colors[tau], label=_tau_tex(tau))
        lo = np.nanmin(sub[["theory_mse", "sim_mse"]].to_numpy()[sub[["theory_mse", "sim_mse"]].to_numpy() > 0])
        hi = np.nanmax(sub[["theory_mse", "sim_mse"]].to_numpy())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"exact: $\sum_j (b_j^2 + \sigma_j^2/R)$")
        ax.set_ylabel(r"replications: $\sum_j$ mean $(\tilde{P}_j - P_j)^2$")
        ax.set_title(DEVICE_TEX.get(device, device), fontsize=10)
        if j == len(devices) - 1:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Check of the exact MSE formula against nested replications", fontsize=11)
    return fig


# ---- Text summary ------------------------------------------------------------------------------

def _fmt_r(r: float) -> str:
    return "never" if not np.isfinite(r) else f"{r:,.3g}"


def _takeaways(aggregate: pd.DataFrame, summary: pd.DataFrame, by_bin: pd.DataFrame, validation: pd.DataFrame, agreement: pd.DataFrame) -> List[str]:
    lines = []
    ratio_cols = [c for c in summary.columns if c.endswith("forward: ratio") and "backward" not in c]
    for (family, k), group in aggregate.groupby(["family", "k"], sort=True):
        first = group.iloc[0]
        lines.append(f"{family}, K = {k} ({int(first['n_cases'])} cases):")
        lines.append(
            f"  • Emulator RMS probability error {first['emu_rms']:.2e} (reference SE {first['ref_se_rms']:.1e}"
            + (f"; not resolved in {int(first['n_unresolved'])} cases)" if first["n_unresolved"] else ")")
        )
        for row in group.itertuples(index=False):
            srow = summary[(summary["family"] == family) & (summary["k"] == k) & (summary["tau"] == row.tau)]
            timing_text = []
            if np.isfinite(row.r_star) and not srow.empty:
                for col in ratio_cols:
                    ratio = float(srow[col].iloc[0])
                    if np.isfinite(ratio):
                        timing_text.append(f"{col.split(' ')[0]} {ratio:,.0f}× the emulator's forward time")
            bound = "≥ " if row.r_star_status == "lower bound" else "≈ "
            lower = f", lower bound only in {int(row.n_lower_bound)}" if row.n_lower_bound else ""
            lines.append(
                f"  • {_tau_label(row.tau)}: bias floor {row.floor_rms:.1e}; break-even R {bound}{_fmt_r(row.r_star)}"
                f" (per case: median {_fmt_r(row.median_case_r_star)}, never in {int(row.n_never)} of {int(row.n_cases)}{lower})"
                + (f"; at that R, {', '.join(timing_text)}" if timing_text else "")
                + "."
            )
    if not by_bin.empty:
        lines.append("Small probabilities (median over alternatives, by probability bin):")
        for family, group in by_bin.groupby("family", sort=True):
            emu = group.drop_duplicates("p_bin")
            emu_text = ", ".join(f"{b}: {e:.1%}" for b, e in zip(emu["p_bin"], emu["emu_median_abs_rel_error"]))
            lines.append(f"  • {family}: emulator |relative error| {emu_text}.")
            tau_min = min(t for t in group["tau"].unique() if math.isfinite(t))
            sub = group[group["tau"] == tau_min]
            bias_text = ", ".join(f"{b}: {e:+.1%}" for b, e in zip(sub["p_bin"], sub["median_rel_bias"]))
            lines.append(f"    {_tau_label(tau_min)} relative smoothing bias {bias_text}.")
    if not validation.empty:
        text = "; ".join(
            f"{row.device} {_tau_label(row.tau)}: median {row.median:.2f} (5–95%: {row.p05:.2f}–{row.p95:.2f})"
            for row in validation.itertuples(index=False)
        )
        lines.append(f"Formula check (replication MSE / exact MSE, per case; 1 is exact): {text}.")
    if agreement is not None and not agreement.empty:
        lines.append(f"Emulator CPU vs MPS: largest probability difference {agreement['max_abs_diff'].max():.1e}.")
    return lines


def analyze(
    results: Mapping[str, object],
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    show: bool = True,
) -> Dict[str, object]:
    """
    Tables, figures and a text summary. ``device`` selects whose emulator probabilities
    define the emulator's error (default: the reference device). Figures and tables are
    saved under ``<run output_dir>/analysis`` unless ``output_dir`` is given ("" to skip).
    """
    device = device or _primary_device(results)
    if output_dir is None:
        output_dir = os.path.join(results.get("settings", {}).get("output_dir", DEFAULT_OUTPUT_DIR), "analysis")
    alt = alternative_table(results, device)
    cases = case_table(alt)
    aggregate = aggregate_table(cases)
    wide = case_wide(cases)
    by_bin = probability_bin_table(alt)
    validation_rows, validation = validation_table(results)
    log_table = log_scale_table(results, alt)
    agreement = device_agreement(results)
    timing = results.get("timing")
    timing = timing if isinstance(timing, pd.DataFrame) else pd.DataFrame()
    summary = break_even_summary(aggregate, timing)

    config_plt()  # the paper's figure style (usetex, serif fonts, palette)
    figures: Dict[str, plt.Figure] = {}
    _finish(plot_error_vs_r(aggregate, validation_rows, device), "error_vs_r", output_dir, show, figures)
    _finish(plot_r_star_vs_probability(alt), "r_star_vs_probability", output_dir, show, figures)
    if not timing.empty:
        _finish(plot_frontier(aggregate, timing, "forward"), "frontier_forward", output_dir, show, figures)
        _finish(plot_frontier(aggregate, timing, "forward_backward"), "frontier_forward_backward", output_dir, show, figures)
    _finish(plot_validation(validation_rows), "formula_check", output_dir, show, figures)

    tables = {
        "alternatives": alt,
        "cases": cases,
        "case_summary": wide,
        "aggregate": aggregate,
        "break_even": summary,
        "probability_bins": by_bin,
        "validation": validation,
        "log_scale": log_table,
        "device_agreement": agreement,
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, table in tables.items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                table.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    lines = _takeaways(aggregate, summary, by_bin, validation, agreement)
    print("\n".join(lines))
    if show:
        with pd.option_context("display.max_columns", 30, "display.width", 200, "display.float_format", "{:.3g}".format):
            display(summary)
            display(wide)
    return {**tables, "figures": figures, "takeaways": lines}


# =============================================================================
# Paper outputs: one figure and one LaTeX table for the appendix
# =============================================================================

PAPER_FIGURE_NAME = "smoothed_error_vs_r"      # saved as figures/<name>.{pdf,png,eps}
PAPER_TABLE_FILE = "smoothed_cost_table.tex"   # saved under <output_dir>/paper/
PAPER_TAU = 40.0                               # temperature for the break-even column
PAPER_FAMILY_TITLES = {"probit": "Probit", "gumbel": "Correlated Gumbel"}
PAPER_FAMILY_ROWS = {"probit": "Probit", "gumbel": "Gumbel"}


def _round_sig(x: float, digits: int = 2) -> float:
    if not np.isfinite(x) or x == 0:
        return x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))


def _tex_int(x: float, digits: int = 2) -> str:
    """Integer rounded to `digits` significant figures, with LaTeX thousands separators."""
    return f"{int(_round_sig(x, digits)):,}".replace(",", "{,}")


def _tex_sci(x: float) -> str:
    """Two significant figures in scientific notation, e.g. $1.0 \times 10^{7}$."""
    if not np.isfinite(x):
        return r"$\infty$"
    exponent = int(math.floor(math.log10(x)))
    mantissa = _round_sig(x, 2) / 10 ** exponent
    if mantissa >= 10:
        mantissa, exponent = mantissa / 10, exponent + 1
    return rf"${mantissa:.1f} \times 10^{{{exponent}}}$"


def _draws_at_time(seconds: float, r_measured, t_measured) -> float:
    """Inverse of _time_at: the number of draws at which the simulator takes `seconds`."""
    order = np.argsort(np.asarray(r_measured, dtype=float))
    r_m = np.asarray(r_measured, dtype=float)[order]
    t_m = np.asarray(t_measured, dtype=float)[order]
    if seconds <= t_m[0]:
        return float(r_m[0])
    if seconds <= t_m[-1]:
        return float(np.exp(np.interp(np.log(seconds), np.log(t_m), np.log(r_m))))
    slope = (t_m[-1] - t_m[-2]) / (r_m[-1] - r_m[-2])
    return float(r_m[-1] + (seconds - t_m[-1]) / slope)


def paper_figure(aggregate: pd.DataFrame, replications: pd.DataFrame, device: str):
    """
    RMS probability error of Sim(tau, R) against R for each model and K (rows: probit,
    correlated Gumbel; columns: K). Curves are the exact sqrt(b^2 + sigma^2 / R) from the
    reference pass; dots are the 100 nested replications on `device`; the thick line is
    the emulator's RMS error and the dashed line the reference's RMS standard error.
    All errors are root mean squares over alternatives and configurations.
    """
    config_plt()
    families = [f for f in FAMILIES if f in set(aggregate["family"])]
    k_values = sorted(aggregate["k"].unique())
    taus = sorted(aggregate["tau"].unique())
    colors = _tau_colors(taus)
    r_dense = np.geomspace(1.0, 1e9, 300)
    reps = replications[replications["device"] == device]
    empirical = reps.groupby(["family", "k", "tau", "num_samples"])["sim_mse"].mean().reset_index()
    fig, axes = plt.subplots(len(families), len(k_values), figsize=(4.0 * len(k_values), 3.4 * len(families)),
                             sharex=True, sharey=True, squeeze=False)
    for i, family in enumerate(families):
        for j, k in enumerate(k_values):
            ax = axes[i][j]
            sub = aggregate[(aggregate["family"] == family) & (aggregate["k"] == k)]
            for tau in taus:
                row = sub[sub["tau"] == tau].iloc[0]
                hard = not math.isfinite(tau)
                ax.plot(r_dense, np.sqrt(row.floor_mse + row.var_y / r_dense), color=colors[tau],
                        ls=":" if hard else "-", lw=1.6 if hard else 2.0, label=_tau_tex(tau))
                if not hard:
                    pts = empirical[(empirical["family"] == family) & (empirical["k"] == k) & (empirical["tau"] == tau)]
                    ax.plot(pts["num_samples"], np.sqrt(pts["sim_mse"]), "o", ms=4, color=colors[tau])
            ax.axhline(float(sub["emu_rms"].iloc[0]), color="black", lw=3.0, label="emulator")
            ax.axhline(float(sub["ref_se_rms"].iloc[0]), color="black", lw=1.0, ls="--", label="reference SE")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1.0, 1e9)
            ax.set_ylim(5e-6, 0.6)
            ax.xaxis.set_major_locator(LogLocator(base=100.0, numticks=12))
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
            ax.minorticks_off()
            ax.set_title(f"{PAPER_FAMILY_TITLES[family]}, $K = {k}$")
            if i == len(families) - 1:
                ax.set_xlabel("Number of draws $R$")
            if j == 0:
                ax.set_ylabel("RMS error")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return fig


def paper_table(aggregate: pd.DataFrame, timing: pd.DataFrame, tau: float = PAPER_TAU,
                devices: Sequence[str] = ("cpu", "mps"), n_rows: int = DEFAULT_TIMING_ROWS) -> str:
    """
    LaTeX table (booktabs) with, per model and K: the break-even draws R* of Sim(tau, R)
    (from `aggregate`), and per device the emulator's forward-pass time in ms, the
    equal-cost number of draws R_eq, and the simulator's time at R* relative to the
    emulator's (from `timing`). Values are rounded to two significant figures, except
    times, which have one decimal.
    """
    timing = timing[timing["pass"] == "forward"]
    devices = [d for d in devices if d in set(timing["device"])]
    ncol = 3 + 3 * len(devices)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"{\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}ll" + "r" * (ncol - 2) + r"@{}}",
        r"\toprule",
        " & & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{DEVICE_TEX.get(d, d)}}}" for d in devices) + r" \\",
        " ".join(rf"\cmidrule(l){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(devices))),
        r"\multicolumn{1}{l}{Model} & \multicolumn{1}{c}{$\numalt$} & \multicolumn{1}{c}{$R^*$} & "
        + " & ".join([r"\multicolumn{1}{c}{Emulator (ms)} & \multicolumn{1}{c}{$R_{\mathrm{eq}}$} & \multicolumn{1}{c}{Ratio}"] * len(devices))
        + r" \\",
        r"\midrule",
    ]
    families = [f for f in FAMILIES if f in set(aggregate["family"])]
    for f_index, family in enumerate(families):
        k_values = sorted(aggregate[aggregate["family"] == family]["k"].unique())
        for k_index, k in enumerate(k_values):
            r_star = float(aggregate[(aggregate["family"] == family) & (aggregate["k"] == k) & (aggregate["tau"] == tau)]["r_star"].iloc[0])
            cells = []
            for device in devices:
                g = timing[(timing["device"] == device) & (timing["family"] == family) & (timing["k"] == k)]
                sim = g[g["method"] == "simulator"]
                t_emu = float(g[g["method"] == "emulator"]["seconds"].iloc[0])
                r_eq = _draws_at_time(t_emu, sim["num_samples"], sim["seconds"])
                ratio = float(_time_at([r_star], sim["num_samples"], sim["seconds"])[0]) / t_emu
                cells += [f"{1000 * t_emu:.1f}", _tex_int(r_eq), _tex_int(ratio) if np.isfinite(ratio) else r"$\infty$"]
            name = rf"\multirow{{{len(k_values)}}}{{*}}{{{PAPER_FAMILY_ROWS[family]}}}" if k_index == 0 else ""
            lines.append(f"{name} & {k} & {_tex_sci(r_star)} & " + " & ".join(cells) + r" \\")
        if f_index < len(families) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{Cost of matching the emulator's accuracy with Sim({tau:g}, $R$). $R^*$ is the number of draws at "
        r"which the simulator's RMS error equals the emulator's (Figure~\ref{fig:smoothed-error-vs-r}). For each device, "
        rf"Emulator (ms) is the time of one emulator evaluation of the choice probabilities for {n_rows:,} observations; "
        r"$R_{\mathrm{eq}}$ is the number of draws at which the simulator takes the same time; and Ratio is the "
        r"simulator's time at $R^*$ divided by the emulator's time. Simulator times beyond $R = 10^4$ are "
        r"extrapolated linearly in $R$.}".replace(f"{n_rows:,}", f"{n_rows:,}".replace(",", "{,}")),
        r"\label{tab:smoothed-cost}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def paper_numbers(results: Mapping[str, object], aggregate: pd.DataFrame, timing: pd.DataFrame,
                  tau: float = PAPER_TAU) -> List[str]:
    """The numbers quoted in the paper's text, each with the table and column it comes from."""
    ref = results["reference"]
    hard = ref[np.isinf(ref["tau"])]
    lines = [
        f"configurations: {hard['case_id'].nunique()}; alternatives: {len(hard)} "
        f"[reference.csv, tau = inf rows]",
        f"smallest / largest reference probability: {hard['p_ref'].min():.2e} / {hard['p_ref'].max():.3f} "
        f"[reference.csv, p_ref]",
        f"largest reference SE: {hard['p_ref_se'].max():.1e} [reference.csv, p_ref_se]",
    ]
    for k, g in aggregate[np.isinf(aggregate["tau"])].groupby("k"):
        lines.append(f"K = {k}: emulator RMS error {g['emu_rms'].min():.2e} to {g['emu_rms'].max():.2e} "
                     f"[aggregate.csv, emu_rms]")
    floors = aggregate[aggregate["tau"] == 5.0]["floor_rms"]
    lines.append(f"tau = 5 smoothing floor: {floors.min():.2e} to {floors.max():.2e} [aggregate.csv, floor_rms, tau = 5]")
    exceed = aggregate[aggregate["tau"] == 5.0]
    lines.append(f"tau = 5 floor exceeds emulator RMS error in {int((exceed['floor_rms'] > exceed['emu_rms']).sum())} "
                 f"of {len(exceed)} model-K groups [aggregate.csv]")
    for t in sorted(aggregate["tau"].unique()):
        if math.isfinite(t):
            g = aggregate[aggregate["tau"] == t]
            lines.append(f"tau = {t:g}: floor below emulator error in {int((g['floor_rms'] < g['emu_rms']).sum())} "
                         f"of {len(g)} groups [aggregate.csv]")
    for k, g in aggregate[aggregate["tau"] == tau].groupby("k"):
        lines.append(f"K = {k}: R* at tau = {tau:g} from {g['r_star'].min():.2e} to {g['r_star'].max():.2e} "
                     f"[aggregate.csv, r_star]")
    fwd = timing[timing["pass"] == "forward"]
    for device, g in fwd.groupby("device"):
        emu = g[g["method"] == "emulator"]["seconds"] * 1000
        lines.append(f"{device}: emulator forward time {emu.min():.1f} to {emu.max():.1f} ms [timing.csv]")
        r_eqs, err_ratios, ratios = [], [], []
        for (family, k), h in g.groupby(["family", "k"]):
            sim = h[h["method"] == "simulator"]
            t_emu = float(h[h["method"] == "emulator"]["seconds"].iloc[0])
            r_eq = _draws_at_time(t_emu, sim["num_samples"], sim["seconds"])
            agg = aggregate[(aggregate["family"] == family) & (aggregate["k"] == k)]
            smoothed = agg[np.isfinite(agg["tau"])]
            best = min(math.sqrt(r.floor_mse + r.var_y / r_eq) for r in smoothed.itertuples(index=False))
            r_star = float(agg[agg["tau"] == tau]["r_star"].iloc[0])
            r_eqs.append(r_eq)
            err_ratios.append(best / float(agg["emu_rms"].iloc[0]))
            ratios.append(float(_time_at([r_star], sim["num_samples"], sim["seconds"])[0]) / t_emu)
        lines.append(f"{device}: R_eq {min(r_eqs):.0f} to {max(r_eqs):.0f}; best smoothed simulator error at R_eq is "
                     f"{min(err_ratios):.1f} to {max(err_ratios):.0f} times the emulator's; time ratio at R* "
                     f"{min(ratios):.0f} to {max(ratios):,.0f} [timing.csv + aggregate.csv]")
    return lines


def write_paper_outputs(results: Mapping[str, object], device: Optional[str] = None,
                        output_dir: Optional[str] = None, tau: float = PAPER_TAU, show: bool = False) -> Dict[str, object]:
    """
    Write the paper's figure (figures/smoothed_error_vs_r.{pdf,png,eps}, via
    config_plt.save_my_fig) and table (<output_dir>/paper/smoothed_cost_table.tex), and
    print the table and the numbers quoted in the text.
    """
    device = device or _primary_device(results)
    output_dir = output_dir or results.get("settings", {}).get("output_dir", DEFAULT_OUTPUT_DIR)
    aggregate = aggregate_table(case_table(alternative_table(results, device)))
    timing = results["timing"]

    fig = paper_figure(aggregate, results["replications"], device)
    os.makedirs("figures", exist_ok=True)  # save_my_fig writes to ./figures/
    save_my_fig(fig, PAPER_FIGURE_NAME)
    if show:
        plt.show()
    else:
        plt.close(fig)

    table = paper_table(aggregate, timing, tau=tau,
                        n_rows=int(results.get("settings", {}).get("timing_rows", DEFAULT_TIMING_ROWS)))
    table_path = os.path.join(output_dir, "paper", PAPER_TABLE_FILE)
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    with open(table_path, "w", encoding="utf-8") as handle:
        handle.write(table + "\n")
    numbers = paper_numbers(results, aggregate, timing, tau)
    print(f"Wrote figures/{PAPER_FIGURE_NAME}.pdf and {table_path}\n")
    print(table)
    print("\nNumbers quoted in the text:")
    print("\n".join("  " + line for line in numbers))
    return {"figure": fig, "table": table, "table_path": table_path, "numbers": numbers, "aggregate": aggregate}


# =============================================================================
# Self-test
# =============================================================================

def _quadrature_probabilities(family: str, v: np.ndarray, scales: np.ndarray, n_grid: int = 200_001) -> np.ndarray:
    """
    Exact choice probabilities for independent errors eps_j = s_j z_j (diagonal Sigma):
    P_j = integral f(t) prod_{i != j} F((v_j + s_j t - v_i) / s_i) dt, by the trapezoid rule.
    """
    from scipy.special import ndtr

    if family == "gumbel":
        t = np.linspace(-8.0, 45.0, n_grid)

        def cdf(x):
            return np.exp(-np.exp(-(x + EULER_MASCHERONI)))

        pdf = np.exp(-(t + EULER_MASCHERONI)) * cdf(t)
    else:
        t = np.linspace(-12.0, 12.0, n_grid)
        cdf = ndtr
        pdf = np.exp(-0.5 * t ** 2) / math.sqrt(2.0 * math.pi)
    k = len(v)
    probs = np.zeros(k)
    for j in range(k):
        integrand = pdf.copy()
        for i in range(k):
            if i != j:
                integrand *= cdf((v[j] + scales[j] * t - v[i]) / scales[i])
        probs[j] = np.trapezoid(integrand, t) if hasattr(np, "trapezoid") else np.trapz(integrand, t)
    return probs


def self_test(
    device: Optional[str] = None,
    n_draws: int = 4_000_000,
    n_reps: int = 400,
    seed: int = 0,
    emulator_paths: Optional[Mapping[str, str]] = DEFAULT_EMULATOR_PATHS,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Quick checks on one device (about a minute):

    1. Gumbel with 0 factors (Sigma = I) reproduces the multinomial logit.
    2. Diagonal Sigma (K = 3, 1 factor => Sigma = diag(1, 1, 5)) matches 1-D quadrature for
       both families, which checks the square roots and the error draws.
    3. Smoothed probabilities sum to one.
    4. The exact MSE formula matches nested replications (K = 5, 2 factors, v_2 = 4, tau = 5).
    5. If the checkpoints exist: emulators load in eval mode, and their probabilities sum to
       one. Their largest errors against the quick reference are printed for information.

    Pass criteria are in standard-error units of the quick reference (|z| < 5), or a ratio band.
    """
    device = _resolve_devices([device or _default_device()])[0]
    results: Dict[str, float] = {}
    status: Dict[str, bool] = {}
    max_chunk = DEFAULT_MAX_CHUNK_ELEMENTS

    def reference(case, taus=(5.0,)):
        with _PreserveRNGState(device):
            return _reference_case(case, taus, n_draws, seed, device, max_chunk)

    def hard(table):
        return table[table["tau"] == HARD].sort_values("alt")

    # 1. Logit
    case = Case("gumbel", 5, 0, "lin0_m4")
    ref = hard(reference(case))
    logit = torch.softmax(utility_vector(case.utility, case.k).double(), dim=0).numpy()
    z = np.max(np.abs(ref["p_ref"].to_numpy() - logit) / ref["p_ref_se"].to_numpy())
    results["gumbel_iid_vs_logit_max_z"] = float(z)
    status["gumbel_iid_vs_logit_max_z"] = z < 5.0

    # 2. Quadrature for diagonal Sigma
    for family in FAMILIES:
        case = Case(family, 3, 1, "v2_2")
        v, sigma, _, _ = case_inputs(case)
        exact = _quadrature_probabilities(family, v.double().numpy(), np.sqrt(np.diag(sigma.double().numpy())))
        ref_table = reference(case)
        ref = hard(ref_table)
        z = np.max(np.abs(ref["p_ref"].to_numpy() - exact) / ref["p_ref_se"].to_numpy())
        results[f"{family}_diagonal_vs_quadrature_max_z"] = float(z)
        status[f"{family}_diagonal_vs_quadrature_max_z"] = z < 5.0
        smoothed_sum = ref_table[ref_table["tau"] == 5.0]["p_tau"].sum()
        results[f"{family}_smoothed_sum_minus_one"] = float(abs(smoothed_sum - 1.0))
        status[f"{family}_smoothed_sum_minus_one"] = abs(smoothed_sum - 1.0) < 1e-4

    # 4. Exact MSE formula vs replications
    for family in FAMILIES:
        case = Case(family, 5, 2, "v2_4")
        ref_table = reference(case)
        p_ref = hard(ref_table)["p_ref"].to_numpy()
        with _PreserveRNGState(device):
            reps = _replicate_case(case, (5.0,), (100,), n_reps, seed + 1, device, max_chunk, p_ref)
        reps = reps[reps["tau"] == 5.0].sort_values("alt")
        moments = ref_table[ref_table["tau"] == 5.0].sort_values("alt")
        theory = float(np.sum(moments["bias"].to_numpy() ** 2 + moments["var_y"].to_numpy() / 100.0))
        ratio = float(reps["sim_mse"].sum() / theory)
        results[f"{family}_formula_ratio_R100"] = ratio
        status[f"{family}_formula_ratio_R100"] = 0.75 < ratio < 1.33

    # 5. Emulators
    if emulator_paths:
        for family, path in emulator_paths.items():
            if not os.path.exists(path):
                if verbose:
                    print(f"  (skipping {family} emulator check: {path} not found)")
                continue
            model, _ = load_emulator(path, device)
            worst = 0.0
            sums = []
            for utility in ("zero", "v2_4"):
                case = Case(family, 5, 2, utility)
                v, sigma, _, _ = case_inputs(case)
                p_emu = emulator_probabilities(model, v, sigma, device)
                sums.append(abs(p_emu.sum() - 1.0))
                ref = hard(reference(case, taus=()))
                worst = max(worst, float(np.max(np.abs(p_emu - ref["p_ref"].to_numpy()))))
            results[f"{family}_emulator_sum_minus_one"] = float(max(sums))
            status[f"{family}_emulator_sum_minus_one"] = max(sums) < 1e-5
            results[f"{family}_emulator_max_abs_error_info"] = worst

    if verbose:
        print(f"Self-test on {device} ({n_draws:,} reference draws):")
        for name, value in results.items():
            flag = "PASS" if status.get(name, True) else "FAIL"
            if name.endswith("_info"):
                flag = "info"
            print(f"  {name:42s} {value:.4g}  {flag}")
    return results


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    self_test()
    results = run_comparison()  # loads the cached stages if the settings are unchanged
    analyze(results, show=False)
    write_paper_outputs(results)
