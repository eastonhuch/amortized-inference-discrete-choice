"""
Mixed logit/probit models with factor-structured random coefficients.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import trange
import json
import os
from typing import Tuple, Optional, Dict, Any
import gc

from utils import canonicalize
from simulation import OptimizationResult, robust_matrix_inverse

# =============================================================================
# Default Constants for Penalties
# =============================================================================

DEFAULT_LAMBDA_REG = 0.001
DEFAULT_LOG_PSI_PRIOR_STD = 1.
DEFAULT_GAMMA_PRIOR_STD = 1.
DEFAULT_LOG_PSI_LOWER_BOUND = -8.0
DEFAULT_LOG_L_DIAG_LOWER_BOUND = -8.0

# Gauss-Hermite quadrature nodes
HERMITE_NODES = 20


def ses_from_cov(cov):
    return torch.sqrt(torch.diag(cov))


# =============================================================================
# Flat Parameter Layout (single source of truth)
# =============================================================================

# Version of the flat parameter layout recorded in saved summaries
# (summary["config"]["param_version"]). Summaries with a different version
# are rejected when loaded.
PARAM_VERSION = 2


def _emulator_includes_c_star(emulator) -> bool:
    """
    Whether an emulator uses index features (include_c_star). Works for
    emulators wrapped in torch.nn.DataParallel, which does not forward
    plain attributes of the wrapped module.
    """
    return bool(getattr(getattr(emulator, "module", emulator), "include_c_star", False))


def _lower_tri_entries(n_rows: int, n_cols: int, skip_00: bool = False) -> list:
    """
    (row, col) positions of the lower-triangular part of an (n_rows x n_cols)
    matrix in column-major order, which is the order used in the flat
    parameter vector. The (0, 0) element is omitted when skip_00=True.
    """
    entries = []
    for j in range(n_cols):
        for i in range(j, n_rows):
            if skip_00 and i == 0 and j == 0:
                continue
            entries.append((i, j))
    return entries


def _check_param_version(summary: dict, source: str = "Model summary") -> None:
    """Raise if a saved summary was not written with the current parameter layout."""
    version = summary.get("config", {}).get("param_version")
    if version != PARAM_VERSION:
        raise ValueError(
            f"{source} has param_version={version!r}, but this code expects "
            f"param_version={PARAM_VERSION}. Summaries written before version 2 "
            "fixed L_rand[0,0]=1 for anchored random effects, so their parameter "
            "vectors do not match the current layout. Refit the model."
        )


class ParamLayout:
    """
    Layout of the flat (unconstrained) parameter vector of a MixedModel.

    This class is the single place that decides which parameters exist, in
    what order, and how they map to the constrained model quantities. The
    model, the warm-start/dense conversions, and the summary readers all use it.

    Each position in the vector is identified by a key:

        ("int", b)          Intercept of brand-form b + 1, b = 0..n_brand_forms-2
                            (the reference brand-form 0 has none). A fixed
                            intercept when n_factors_rand < 0, otherwise the mean
                            of the corresponding random intercept.
        ("beta", col)       Coefficient of column col >= 1 of X: a fixed
                            coefficient, or the mean of a random coefficient if
                            col is a random-feature column.
        ("L_rand", i, j)    Random-effects loading L_rand[i, j] (log scale if i == j).
        ("psi_rand", i)     Log-uniqueness of random effect i.
        ("L_time", i, j)    Time-specific loading L_time[i, j] (log scale if i == j).
        ("psi_time", i)     Log-uniqueness of time-specific error i.

    Random effects are indexed 0..k-1: the n_brand_forms - 1 intercepts, then the
    random feature coefficients in the order of random_feature_indices.

    Identification
    --------------
    Random effects: L_rand is lower triangular and all of its diagonal
    elements, including L_rand[0,0], are free (log scale). The error
    distribution already fixes the scale of utility, so no element of the
    random-effects covariance is normalized. With anchored_rand=True the first
    r_rand uniquenesses are fixed to 0; each added factor then absorbs one
    uniqueness, so models with 0, 1, 2, ..., k factors are nested. With
    anchored_rand=False all k uniquenesses are free.

    Time-specific errors: the reference alternative has unit variance and zero
    covariance with the others, and Ω[0,0] = 1 fixes the scale (L_time[0,0] = 1
    with psi_time[0] = 0 when n_factors_time > 0; psi_time[0] = 1 when
    n_factors_time = 0; Ω = I when n_factors_time < 0).

    Parameters
    ----------
    n_brand_forms, X_dim, random_feature_indices, n_factors_rand,
    include_uniquenesses_rand, anchored_rand, n_factors_time :
        As in MixedModel. n_factors_rand = 0 without uniquenesses is treated as
        a fixed-effects model, exactly as MixedModel does.
    has_time_params : bool
        Whether the time-specific block is present (MixedModel includes it
        only when an emulator is attached).
    """

    def __init__(
        self,
        n_brand_forms: int,
        X_dim: int,
        random_feature_indices=None,
        n_factors_rand: int = 0,
        include_uniquenesses_rand: bool = False,
        anchored_rand: bool = True,
        n_factors_time: int = -1,
        has_time_params: bool = False,
    ):
        rfi = list(random_feature_indices or [])
        r_rand = int(n_factors_rand)
        if r_rand == 0 and not include_uniquenesses_rand:
            r_rand = -1
        r_time = -1 if n_factors_time is None else int(n_factors_time)

        self.n_brand_forms = int(n_brand_forms)
        self.X_dim = int(X_dim)
        self.random_feature_indices = rfi
        self.r_rand = r_rand
        self.r_time = r_time
        self.include_uniquenesses_rand = bool(include_uniquenesses_rand)
        self.anchored_rand = bool(anchored_rand)
        self.has_time_params = bool(has_time_params)
        self.k = (self.n_brand_forms - 1) + len(rfi)
        self.k_time = self.n_brand_forms - 1

        if r_rand < 0:
            self.random_feature_cols = []
            self.fixed_feature_cols = list(range(1, self.X_dim))
        else:
            self.random_feature_cols = [idx + 1 for idx in rfi]
            self.fixed_feature_cols = [
                c for c in range(1, self.X_dim) if c not in self.random_feature_cols
            ]

        # Random-effects covariance block
        if r_rand < 0:
            L_rand_entries, psi_rand_free = [], []
        elif r_rand == 0:
            L_rand_entries, psi_rand_free = [], list(range(self.k))
        else:
            L_rand_entries = _lower_tri_entries(self.k, r_rand)
            if self.include_uniquenesses_rand:
                start = r_rand if self.anchored_rand else 0
                psi_rand_free = list(range(start, self.k))
            else:
                psi_rand_free = []

        # Time-specific covariance block (L_time[0,0] is always anchored)
        if not self.has_time_params or r_time < 0:
            L_time_entries, psi_time_free = [], []
        elif r_time > 0:
            L_time_entries = _lower_tri_entries(self.k_time, r_time, skip_00=True)
            psi_time_free = list(range(r_time, self.k_time))
        else:
            L_time_entries, psi_time_free = [], list(range(1, self.k_time))

        self.L_rand_entries = L_rand_entries
        self.psi_rand_free = psi_rand_free
        self.L_time_entries = L_time_entries
        self.psi_time_free = psi_time_free

        # Keys in parameter order
        keys = [("int", b) for b in range(self.n_brand_forms - 1)]
        keys += [("beta", c) for c in self.random_feature_cols]
        keys += [("beta", c) for c in self.fixed_feature_cols]
        keys += [("L_rand", i, j) for i, j in L_rand_entries]
        keys += [("psi_rand", i) for i in psi_rand_free]
        keys += [("L_time", i, j) for i, j in L_time_entries]
        keys += [("psi_time", i) for i in psi_time_free]
        self.keys = keys
        self._key_index = {key: pos for pos, key in enumerate(keys)}

        # Block sizes (same names as the MixedModel attributes)
        self.n_intercepts = self.n_brand_forms - 1 if r_rand < 0 else 0
        self.n_random_means = 0 if r_rand < 0 else self.k
        self.n_fixed_features = len(self.fixed_feature_cols)
        self.n_L_rand_params = len(L_rand_entries)
        self.n_psi_rand_params = len(psi_rand_free)
        self.n_L_time_params = len(L_time_entries)
        self.n_psi_time_params = len(psi_time_free)

        # Block offsets
        self.lead_start = 0
        self.lead_end = self.n_intercepts + self.n_random_means
        self.fixed_start = self.lead_end
        self.fixed_end = self.fixed_start + self.n_fixed_features
        self.L_rand_start = self.fixed_end
        self.L_rand_end = self.L_rand_start + self.n_L_rand_params
        self.psi_rand_start = self.L_rand_end
        self.psi_rand_end = self.psi_rand_start + self.n_psi_rand_params
        self.L_time_start = self.psi_rand_end
        self.L_time_end = self.L_time_start + self.n_L_time_params
        self.psi_time_start = self.L_time_end
        self.psi_time_end = self.psi_time_start + self.n_psi_time_params
        self.time_start = self.L_time_start
        self.n_params = self.psi_time_end
        assert self.n_params == len(keys)

        # Positions of diagonal / off-diagonal elements within the L blocks
        self.L_rand_diag_positions = [p for p, (i, j) in enumerate(L_rand_entries) if i == j]
        self.L_rand_offdiag_positions = [p for p, (i, j) in enumerate(L_rand_entries) if i != j]
        self.L_time_diag_positions = [p for p, (i, j) in enumerate(L_time_entries) if i == j]
        self.L_time_offdiag_positions = [p for p, (i, j) in enumerate(L_time_entries) if i != j]

        self._tensor_cache = {}

    # ------------------------------------------------------------------
    # Construction from saved configurations
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict, has_time_params: bool) -> "ParamLayout":
        """Build a layout from a summary/model config dictionary."""
        return cls(
            n_brand_forms=config["n_brand_forms"],
            X_dim=config["X_dim"],
            random_feature_indices=config.get("random_feature_indices"),
            n_factors_rand=config["n_factors_rand"],
            include_uniquenesses_rand=config.get("include_uniquenesses_rand", False),
            anchored_rand=config.get("anchored_rand", True),
            n_factors_time=config.get("n_factors_time", -1),
            has_time_params=has_time_params,
        )

    @classmethod
    def from_summary(cls, summary: dict) -> "ParamLayout":
        """
        Build the layout of a saved summary and check it against the saved
        parameter vector. Raises ValueError for summaries written with an older
        parameter layout or whose parameter count does not match the config.
        """
        _check_param_version(summary)
        config = summary["config"]
        n_saved = len(summary["param_values"])
        flag = config.get("has_time_params")
        candidates = [bool(flag)] if flag is not None else [True, False]
        for has_time in candidates:
            layout = cls.from_config(config, has_time_params=has_time)
            if layout.n_params == n_saved:
                return layout
        raise ValueError(
            f"Saved parameter vector has length {n_saved}, which does not match "
            f"the saved configuration ({[cls.from_config(config, h).n_params for h in candidates]} "
            "expected)."
        )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def index(self, key: tuple) -> int:
        """Position of `key` in the flat parameter vector (KeyError if absent)."""
        return self._key_index[key]

    def named_values(self, param) -> dict:
        """Dictionary {key: float} of the (unconstrained) parameter values."""
        if isinstance(param, torch.Tensor):
            values = param.detach().cpu().tolist()
        else:
            values = list(np.asarray(param, dtype=np.float64).ravel())
        if len(values) != self.n_params:
            raise ValueError(
                f"Parameter vector has length {len(values)}, expected {self.n_params}."
            )
        return dict(zip(self.keys, values))

    def _indices(self, device) -> dict:
        """Index tensors for vectorized matrix construction, cached per device."""
        cache_key = str(device)
        if cache_key not in self._tensor_cache:
            def as_long(values):
                return torch.tensor(values, dtype=torch.long, device=device)

            self._tensor_cache[cache_key] = {
                "L_rand_rows": as_long([i for i, _ in self.L_rand_entries]),
                "L_rand_cols": as_long([j for _, j in self.L_rand_entries]),
                "L_rand_is_diag": torch.tensor(
                    [i == j for i, j in self.L_rand_entries], dtype=torch.bool, device=device
                ),
                "L_time_rows": as_long([i for i, _ in self.L_time_entries]),
                "L_time_cols": as_long([j for _, j in self.L_time_entries]),
                "L_time_is_diag": torch.tensor(
                    [i == j for i, j in self.L_time_entries], dtype=torch.bool, device=device
                ),
            }
        return self._tensor_cache[cache_key]

    # ------------------------------------------------------------------
    # Unpacking into model quantities
    # ------------------------------------------------------------------

    def unpack(self, param):
        """
        Unpack a flat parameter vector (see MixedModel._unpack_params).

        Returns
        -------
        lead : intercepts (n_factors_rand < 0) or random-effect means (k,)
        fixed_coefs : (n_fixed_features,)
        L_rand : (k, r_rand) or None
        psi_rand : (k,) or None
        L_time : (k_time, r_time) or None
        psi_time : (k_time,) or None (None when there is no time block)
        """
        dtype, device = param.dtype, param.device

        lead = param[self.lead_start:self.lead_end]
        fixed_coefs = param[self.fixed_start:self.fixed_end]

        L_rand = None
        psi_rand = None
        if self.r_rand == 0:
            psi_rand = torch.exp(param[self.psi_rand_start:self.psi_rand_end])
        elif self.r_rand > 0:
            idx = self._indices(device)
            L_rand_raw = param[self.L_rand_start:self.L_rand_end]
            L_rand = torch.zeros(self.k, self.r_rand, dtype=dtype, device=device)
            # Diagonal elements (including L_rand[0,0]) are stored on the log scale
            L_rand_values = torch.where(idx["L_rand_is_diag"], torch.exp(L_rand_raw), L_rand_raw)
            L_rand[idx["L_rand_rows"], idx["L_rand_cols"]] = L_rand_values

            if self.include_uniquenesses_rand:
                log_psi_rand_free = param[self.psi_rand_start:self.psi_rand_end]
                if self.anchored_rand:
                    # First r_rand uniquenesses are anchored to 0
                    psi_rand = torch.zeros(self.k, dtype=dtype, device=device)
                    psi_rand[self.r_rand:] = torch.exp(log_psi_rand_free)
                else:
                    psi_rand = torch.exp(log_psi_rand_free)

        L_time = None
        psi_time = None
        if self.has_time_params:
            if self.r_time < 0:
                # Fixed identity covariance (no free parameters)
                psi_time = torch.ones(self.k_time, dtype=dtype, device=device)
            elif self.r_time > 0:
                idx = self._indices(device)
                L_time_raw = param[self.L_time_start:self.L_time_end]
                L_time = torch.zeros(self.k_time, self.r_time, dtype=dtype, device=device)
                L_time[0, 0] = 1.0  # Anchored for identification (scale)
                L_time_values = torch.where(idx["L_time_is_diag"], torch.exp(L_time_raw), L_time_raw)
                L_time[idx["L_time_rows"], idx["L_time_cols"]] = L_time_values

                # First r_time uniquenesses are anchored to 0
                psi_time = torch.zeros(self.k_time, dtype=dtype, device=device)
                psi_time[self.r_time:] = torch.exp(param[self.psi_time_start:self.psi_time_end])
            else:
                # Uniquenesses only, with psi_time[0] anchored to 1 (scale)
                psi_time = torch.zeros(self.k_time, dtype=dtype, device=device)
                psi_time[0] = 1.0
                psi_time[1:] = torch.exp(param[self.psi_time_start:self.psi_time_end])

        return lead, fixed_coefs, L_rand, psi_rand, L_time, psi_time

    def build_Sigma_time_full(self, L_time, psi_time, dtype, device):
        """
        Full (n_brand_forms x n_brand_forms) time-specific covariance: the
        reference brand-form has unit variance and zero covariance with the
        others; the lower-right block is L_time L_time' + diag(psi_time).
        """
        if L_time is not None:
            Sigma_time_partial = torch.matmul(L_time, L_time.T) + torch.diag(psi_time)
        else:
            Sigma_time_partial = torch.diag(psi_time)

        Sigma_time_full = torch.zeros(
            self.n_brand_forms, self.n_brand_forms, dtype=dtype, device=device
        )
        Sigma_time_full[0, 0] = 1.0
        Sigma_time_full[1:, 1:] = Sigma_time_partial
        return Sigma_time_full

    def Sigma_rand(self, param):
        """
        Random-effects covariance matrix (k x k) implied by `param`, or None for
        a fixed-effects model.
        """
        _, _, L_rand, psi_rand, _, _ = self.unpack(param)
        if self.r_rand < 0:
            return None
        if self.r_rand == 0:
            return torch.diag(psi_rand)
        Sigma = torch.matmul(L_rand, L_rand.T)
        if psi_rand is not None:
            Sigma = Sigma + torch.diag(psi_rand)
        return Sigma

    def Sigma_time(self, param):
        """
        Full time-specific covariance (n_brand_forms x n_brand_forms) implied by
        `param`. Without a time block this is the identity matrix.
        """
        if not self.has_time_params:
            return torch.eye(self.n_brand_forms, dtype=param.dtype, device=param.device)
        _, _, _, _, L_time, psi_time = self.unpack(param)
        return self.build_Sigma_time_full(L_time, psi_time, param.dtype, param.device)

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    def random_effect_label(self, idx, intercept_names=None, numeric_names=None) -> str:
        """Label of random effect idx (0..k-1)."""
        if idx < self.n_brand_forms - 1:
            if intercept_names is not None:
                return intercept_names[idx]
            return f"intercept_bf{idx + 1}"
        feat_idx = self.random_feature_indices[idx - (self.n_brand_forms - 1)]
        if numeric_names is not None:
            return numeric_names[feat_idx]
        return f"x{feat_idx + 1}"

    def brand_form_label(self, idx, intercept_names=None) -> str:
        """Label of non-reference brand-form idx (0..n_brand_forms-2)."""
        if intercept_names is not None:
            return intercept_names[idx]
        return f"bf{idx + 1}"

    def param_names(self, intercept_names=None, numeric_names=None) -> list:
        """Descriptive names for all parameters, in parameter order."""
        random_col_pos = {c: p for p, c in enumerate(self.random_feature_cols)}

        def re_label(i):
            return self.random_effect_label(i, intercept_names, numeric_names)

        def bf_label(i):
            return self.brand_form_label(i, intercept_names)

        names = []
        for key in self.keys:
            kind = key[0]
            if kind == "int":
                b = key[1]
                if self.r_rand < 0:
                    names.append(f"intercept[{bf_label(b)}]")
                else:
                    names.append(f"mean[{re_label(b)}]")
            elif kind == "beta":
                col = key[1]
                if col in random_col_pos:
                    names.append(f"mean[{re_label(self.n_brand_forms - 1 + random_col_pos[col])}]")
                elif numeric_names is not None:
                    names.append(f"beta[{numeric_names[col - 1]}]")
                else:
                    names.append(f"beta[x{col}]")
            elif kind == "L_rand":
                _, i, j = key
                if i == j:
                    names.append(f"log(L_rand[{re_label(i)},{j}])")
                else:
                    names.append(f"L_rand[{re_label(i)},{j}]")
            elif kind == "psi_rand":
                names.append(f"log(psi_rand[{re_label(key[1])}])")
            elif kind == "L_time":
                _, i, j = key
                if i == j:
                    names.append(f"log(L_time[{bf_label(i)},{j}])")
                else:
                    names.append(f"L_time[{bf_label(i)},{j}]")
            elif kind == "psi_time":
                names.append(f"log(psi_time[{bf_label(key[1])}])")
        return names


class MixedModel:
    """
    Mixed logit/probit model with factor-structured random coefficients.

    Random effects include brand-form intercepts and optionally specified
    numeric features. Covariance is parameterized as LL' where L is a
    lower-triangular k x r matrix, or as LL' + Ψ where Ψ is a diagonal
    matrix of uniquenesses when include_uniquenesses_rand=True.

    When an emulator is provided, the model uses a neural network to compute
    choice probabilities for correlated discrete choice models (e.g., probit).
    The time-specific error covariance is parameterized with a factor structure
    using L_time and psi_time. For the special case of 2 alternatives, closed-form
    probit probabilities are computed instead of using the emulator.

    When no emulator is provided, the model uses either logit or probit errors
    based on the model_family parameter. For probit with 2 alternatives, closed-form
    probabilities are used. For probit with more than 2 alternatives, Gauss-Hermite
    quadrature is used to approximate choice probabilities.

    When n_factors_time < 0, the time-specific covariance is fixed to an identity
    matrix with no free parameters.

    When n_factors_rand < 0, the model reduces to a fixed effects model with no
    household-level random effects.

    When n_factors_rand = 0 and include_uniquenesses_rand=True, the model has
    independent random effects (diagonal covariance matrix).

    When n_factors_rand = 0 and include_uniquenesses_rand=False, the model falls
    back to fixed effects (n_factors_rand is set to -1).

    All diagonal elements of L_rand, including L_rand[0,0], are free parameters
    stored on the log scale; the scale of utility is fixed by the error
    distribution, not by the random effects. When anchored_rand=True (default)
    and n_factors_rand > 0, the first r_rand uniquenesses are fixed to 0, so
    each added factor absorbs one uniqueness and models with 0, 1, 2, ... factors
    are nested. See ParamLayout for the full parameter layout.
    """

    def __init__(self, data_dict, n_brand_forms, X_dim, n_households, n_total_households,
                 random_feature_indices=None, n_factors_rand=0,
                 include_uniquenesses_rand=False, n_factors_time=0,
                 emulator=None, model_family="logit", n_draws=200, seed=42, 
                 device=None, parent_model=None, anchored_rand=True,
                 lambda_reg=DEFAULT_LAMBDA_REG,
                 log_psi_prior_std=DEFAULT_LOG_PSI_PRIOR_STD,
                 gamma_prior_std=DEFAULT_GAMMA_PRIOR_STD,
                 log_psi_lower_bound=DEFAULT_LOG_PSI_LOWER_BOUND,
                 log_L_diag_lower_bound=DEFAULT_LOG_L_DIAG_LOWER_BOUND):
        """
        Parameters
        ----------
        data_dict : dict
            Dictionary with choice set sizes as keys, containing alternative_ids,
            X, choice_idx, and household_id for each group.
        n_brand_forms : int
            Number of brand-form alternatives.
        X_dim : int
            Number of features including intercept column.
        n_households : int
            Number of unique households.
        n_total_households : int
            Number of unique households across all batches
        random_feature_indices : list, optional
            Indices of numeric features (0-indexed, where 0 = first numeric feature
            after intercept) with random effects. Default is empty list.
            Ignored if n_factors_rand < 0.
        n_factors_rand : int
            Number of factors for household-level random effects.
            If < 0, no household-level random effects are included (fixed effects).
            If = 0 and include_uniquenesses_rand=True, independent random effects.
            If = 0 and include_uniquenesses_rand=False, falls back to fixed effects.
        include_uniquenesses_rand : bool
            Whether to include uniquenesses (diagonal variance components) in the
            random effects covariance matrix. When True with n_factors_rand > 0,
            covariance is LL' + Ψ. When True with n_factors_rand = 0, covariance
            is diagonal (Ψ only). Default is False. Ignored if n_factors_rand < 0.
        n_factors_time : int
            Number of factors for time-specific error covariance.
            If 0 and emulator is provided, only uniquenesses are used.
            If < 0 and emulator is provided, covariance is fixed to identity
            with no free parameters.
            Ignored if emulator=None.
        emulator : callable, optional
            A PyTorch model that takes (v_star, s_star, c_star) and returns logits.
            When provided, replaces the logit choice probabilities with
            emulator-based probabilities. For 2-alternative choice sets,
            closed-form probit probabilities are used instead. Default is None.
        model_family : str
            Error distribution family when emulator is None. Either "logit" or
            "probit". Default is "logit". Ignored when emulator is provided.
        n_draws : int
            Number of Monte Carlo simulation draws. Ignored if n_factors_rand < 0.
        seed : int
            Random seed for generating draws.
        device : torch.device, optional
            Device for tensors. Default is torch.device('cpu').
        parent_model: MixedModel
            A parent model having the same structure but different data.
        anchored_rand : bool
            Whether to use anchored parameterization for random effects factor
            structure. When True and n_factors_rand > 0, the first r_rand
            uniquenesses are fixed to 0, which removes flat directions and makes
            models with increasing numbers of factors nested. L_rand[0,0] is
            always a free parameter (log scale). Has no effect when
            include_uniquenesses_rand=False. Default is True.
        lambda_reg : float
            Regularization strength for penalties. Default is DEFAULT_LAMBDA_REG (0.001).
        log_psi_prior_std : float
            Standard deviation for Gaussian prior on log-uniquenesses.
            Default is DEFAULT_LOG_PSI_PRIOR_STD (1.0).
        gamma_prior_std : float
            Standard deviation for Gaussian prior on factor loadings (off-diagonal
            elements of L matrices). Default is DEFAULT_GAMMA_PRIOR_STD (1.0).
        log_psi_lower_bound : float
            Lower bound for log-uniquenesses in barrier penalty. Default is -8.0.
        log_L_diag_lower_bound : float
            Lower bound for log-diagonal Cholesky elements in barrier penalty. 
            Default is -8.0.
        """
        self.data_dict = data_dict
        self.cs = [c for c in data_dict if isinstance(c, int)]
        self.n_brand_forms = n_brand_forms
        self.X_dim = X_dim
        self.n_households = n_households
        self.n_total_households = n_total_households
        self.n_factors_time = n_factors_time
        self.emulator = emulator
        self.n_draws = n_draws
        self.device = device if device is not None else torch.device('cpu')
        self.parent_model = parent_model

        # Model family (logit or probit) - only used when emulator is None
        if self.parent_model is None:
            if model_family not in ("logit", "probit"):
                raise ValueError(f"model_family must be 'logit' or 'probit', got '{model_family}'")
            self.model_family = model_family
            self.anchored_rand = anchored_rand
        else:
            self.model_family = self.parent_model.model_family
            self.anchored_rand = self.parent_model.anchored_rand

        # Warn if model_family is specified but emulator is provided
        if emulator is not None and model_family != "logit" and self.parent_model is None:
            warnings.warn(
                f"model_family='{model_family}' specified but emulator is provided. "
                "The emulator will be used for choice probabilities."
            )

        # Penalty hyperparameters
        if self.parent_model is None:
            self.lambda_reg = lambda_reg
            self.log_psi_prior_std = log_psi_prior_std
            self.gamma_prior_std = gamma_prior_std
            self.log_psi_lower_bound = log_psi_lower_bound
            self.log_L_diag_lower_bound = log_L_diag_lower_bound
        else:
            self.lambda_reg = self.parent_model.lambda_reg
            self.log_psi_prior_std = self.parent_model.log_psi_prior_std
            self.gamma_prior_std = self.parent_model.gamma_prior_std
            self.log_psi_lower_bound = self.parent_model.log_psi_lower_bound
            self.log_L_diag_lower_bound = self.parent_model.log_L_diag_lower_bound

        # Handle n_factors_rand = 0 with include_uniquenesses_rand = False
        if n_factors_rand == 0 and not include_uniquenesses_rand:
            warnings.warn(
                "n_factors_rand=0 and include_uniquenesses_rand=False results in "
                "no random effects. Setting n_factors_rand=-1 (fixed effects model)."
            )
            n_factors_rand = -1

        # Warn if include_uniquenesses_rand=True but n_factors_rand < 0
        if n_factors_rand < 0 and include_uniquenesses_rand:
            warnings.warn(
                "include_uniquenesses_rand=True but n_factors_rand < 0. "
                "Uniquenesses will be ignored (fixed effects model)."
            )

        self.n_factors_rand = n_factors_rand
        self.include_uniquenesses_rand = include_uniquenesses_rand

        if self.parent_model is None:
            if random_feature_indices is None:
                self.random_feature_indices = []
            else:
                self.random_feature_indices = list(random_feature_indices)
        else:
            self.random_feature_indices = self.parent_model.random_feature_indices

        # Validate emulator is callable if provided
        if emulator is not None and not callable(emulator):
            raise ValueError("emulator must be callable")

        # Warn if random_feature_indices specified but n_factors_rand < 0
        if n_factors_rand < 0 and len(self.random_feature_indices) > 0:
            warnings.warn(
                "random_feature_indices specified but n_factors_rand < 0. "
                "All coefficients will be treated as fixed effects."
            )

        # Warn if n_factors_time > 0 but emulator=None
        if n_factors_time > 0 and emulator is None:
            warnings.warn(
                "n_factors_time > 0 but emulator=None. "
                "Time-specific factors will be ignored."
            )

        # Error if n_factors_time > n_brand_forms - 1
        if n_factors_time > n_brand_forms - 1:
            raise ValueError(
                f"n_factors_time ({n_factors_time}) cannot exceed "
                f"n_brand_forms - 1 ({n_brand_forms - 1})"
            )

        if self.parent_model is None:
            # Number of random effects: (n_brand_forms - 1) intercepts + random features
            self.k = (n_brand_forms - 1) + len(self.random_feature_indices)
            self.r_rand = n_factors_rand
            self.r_time = n_factors_time
        else:
            self.k = self.parent_model.k
            self.r_rand = self.parent_model.r_rand
            self.r_time = self.parent_model.r_time

        if self.r_rand > self.k and self.r_rand > 0:
            raise ValueError(
                f"n_factors_rand ({self.r_rand}) cannot exceed "
                f"number of random effects ({self.k})"
            )

        # Warn about potential identification issues with factor models
        if include_uniquenesses_rand and n_factors_rand > 0 and self.r_rand >= self.k / 2:
            warnings.warn(
                f"n_factors_rand ({self.r_rand}) >= k/2 ({self.k/2}) where k is the "
                "number of random effects. This may lead to identification issues "
                "in the factor model."
            )

        if emulator is not None and n_factors_time >= 0 and n_factors_time >= (n_brand_forms - 1) / 2:
            warnings.warn(
                f"n_factors_time ({n_factors_time}) >= (n_brand_forms - 1)/2 "
                f"({(n_brand_forms - 1) / 2}). This may lead to identification "
                "issues in the factor model."
            )

        if self.parent_model is None:
            # The flat parameter layout (block sizes, ordering, anchoring) is
            # defined in one place: ParamLayout.
            self.layout = ParamLayout(
                n_brand_forms=n_brand_forms,
                X_dim=X_dim,
                random_feature_indices=self.random_feature_indices,
                n_factors_rand=n_factors_rand,
                include_uniquenesses_rand=include_uniquenesses_rand,
                anchored_rand=self.anchored_rand,
                n_factors_time=n_factors_time,
                has_time_params=emulator is not None,
            )
            self.random_feature_cols = list(self.layout.random_feature_cols)
            self.fixed_feature_cols = list(self.layout.fixed_feature_cols)
            self.n_fixed_features = self.layout.n_fixed_features

            # Parameter dimensions for random effects
            self.n_intercepts = self.layout.n_intercepts
            self.n_random_means = self.layout.n_random_means
            self.n_L_rand_params = self.layout.n_L_rand_params
            self.n_psi_rand_params = self.layout.n_psi_rand_params

            # Parameter dimensions for time-specific covariance
            # Dimension of time-specific covariance: n_brand_forms - 1 (excluding reference)
            self.k_time = self.layout.k_time
            self.n_L_time_params = self.layout.n_L_time_params
            self.n_psi_time_params = self.layout.n_psi_time_params

            # Total parameter count
            self.n_params = self.layout.n_params
        else:
            self.layout = self.parent_model.layout
            self.random_feature_cols = self.parent_model.random_feature_cols
            self.fixed_feature_cols = self.parent_model.fixed_feature_cols
            self.n_fixed_features = self.parent_model.n_fixed_features
            self.n_random_means = self.parent_model.n_random_means
            self.n_L_rand_params = self.parent_model.n_L_rand_params
            self.n_psi_rand_params = self.parent_model.n_psi_rand_params
            self.n_intercepts = self.parent_model.n_intercepts
            self.k_time = self.parent_model.k_time
            self.n_L_time_params = self.parent_model.n_L_time_params
            self.n_psi_time_params = self.parent_model.n_psi_time_params
            self.n_params = self.parent_model.n_params

        # Pre-generate standard normal draws (only if n_factors_rand >= 0)
        if n_factors_rand >= 0:
            # Check if pre-generated draws are provided in data_dict
            if "std_normal_draws" in data_dict:
                # Use pre-generated draws (slice to needed k dimensions)
                self.std_normal_draws = data_dict["std_normal_draws"][:, :, :self.k].clone()
                # Verify shape
                expected_shape = (n_households, n_draws, self.k)
                if self.std_normal_draws.shape[0] != n_households:
                    raise ValueError(
                        f"std_normal_draws has {self.std_normal_draws.shape[0]} households, "
                        f"expected {n_households}"
                    )
                if self.std_normal_draws.shape[1] < n_draws:
                    raise ValueError(
                        f"std_normal_draws has {self.std_normal_draws.shape[1]} draws, "
                        f"expected at least {n_draws}"
                    )
                # Slice to n_draws if more were provided
                self.std_normal_draws = self.std_normal_draws[:, :n_draws, :]
            else:
                # Generate draws
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                deterministic_old = torch.backends.cudnn.deterministic
                benchmark_old = torch.backends.cudnn.benchmark
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                self.std_normal_draws = torch.randn(
                    n_households, n_draws, self.k,
                    dtype=torch.float32, device=self.device
                )
                torch.backends.cudnn.deterministic = deterministic_old
                torch.backends.cudnn.benchmark = benchmark_old

            if self.parent_model is None and n_factors_rand > 0:
                self._precompute_L_rand_indices()
            elif self.parent_model is not None and n_factors_rand > 0:
                self.L_rand_row_indices = self.parent_model.L_rand_row_indices
                self.L_rand_col_indices = self.parent_model.L_rand_col_indices
                self.L_rand_is_diagonal = self.parent_model.L_rand_is_diagonal
        else:
            self.std_normal_draws = None

        # Precompute L_time indices if needed
        if emulator is not None and n_factors_time > 0:
            if self.parent_model is None:
                self._precompute_L_time_indices()
            else:
                self.L_time_row_indices = self.parent_model.L_time_row_indices
                self.L_time_col_indices = self.parent_model.L_time_col_indices
                self.L_time_is_diagonal = self.parent_model.L_time_is_diagonal

        # Convert feature column lists to tensors for vectorized assignment
        if self.parent_model is None:
            self.random_feature_cols_t = torch.tensor(
                self.random_feature_cols, dtype=torch.long, device=self.device
            ) if self.random_feature_cols else None
            self.fixed_feature_cols_t = torch.tensor(
                self.fixed_feature_cols, dtype=torch.long, device=self.device
            ) if self.fixed_feature_cols else None
        else:
            self.random_feature_cols_t = self.parent_model.random_feature_cols_t
            self.fixed_feature_cols_t = self.parent_model.fixed_feature_cols_t

        # Precompute parameter indices for penalty computation
        if self.parent_model is None:
            self._precompute_penalty_param_indices()
        else:
            self._log_psi_rand_start = self.parent_model._log_psi_rand_start
            self._log_psi_rand_end = self.parent_model._log_psi_rand_end
            self._L_rand_start = self.parent_model._L_rand_start
            self._L_rand_end = self.parent_model._L_rand_end
            self._L_rand_diag_indices = self.parent_model._L_rand_diag_indices
            self._L_rand_offdiag_indices = self.parent_model._L_rand_offdiag_indices
            self._log_psi_time_start = self.parent_model._log_psi_time_start
            self._log_psi_time_end = self.parent_model._log_psi_time_end
            self._L_time_start = self.parent_model._L_time_start
            self._L_time_end = self.parent_model._L_time_end
            self._L_time_diag_indices = self.parent_model._L_time_diag_indices
            self._L_time_offdiag_indices = self.parent_model._L_time_offdiag_indices

        # Initialize Gauss-Hermite quadrature nodes and weights for probit
        if self.parent_model is None:
            if self.model_family == "probit" and self.emulator is None:
                self._init_hermite_quadrature()
            else:
                self.hermite_nodes = None
                self.hermite_log_weights = None
        else:
            self.hermite_nodes = self.parent_model.hermite_nodes
            self.hermite_log_weights = self.parent_model.hermite_log_weights

    def _init_hermite_quadrature(self):
        """Initialize Gauss-Hermite quadrature nodes and weights."""
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(HERMITE_NODES)
        self.hermite_nodes = torch.tensor(
            nodes_np, dtype=torch.float32, device=self.device
        )
        # Include the 1/sqrt(pi) factor in the log weights
        self.hermite_log_weights = torch.tensor(
            np.log(weights_np) - 0.5 * np.log(np.pi),
            dtype=torch.float32, device=self.device
        )

    def _precompute_penalty_param_indices(self):
        """
        Precompute indices into the flat parameter vector for penalty computation.

        This identifies which parameters are:
        - log_psi_rand (log-uniquenesses for random effects)
        - L_rand diagonal elements (on log scale, including L_rand[0,0])
        - L_rand off-diagonal elements (factor loadings)
        - log_psi_time (log-uniquenesses for time-specific errors)
        - L_time diagonal elements (on log scale, excluding anchored [0,0])
        - L_time off-diagonal elements (factor loadings)
        """
        layout = self.layout

        def as_index(positions):
            return torch.tensor(positions, dtype=torch.long, device=self.device)

        # L_rand parameters
        self._L_rand_start = layout.L_rand_start
        self._L_rand_end = layout.L_rand_end
        if self.r_rand > 0 and self.n_L_rand_params > 0:
            self._L_rand_diag_indices = as_index(layout.L_rand_diag_positions)
            self._L_rand_offdiag_indices = as_index(layout.L_rand_offdiag_positions)
        else:
            self._L_rand_diag_indices = None
            self._L_rand_offdiag_indices = None

        # psi_rand parameters (log scale)
        self._log_psi_rand_start = layout.psi_rand_start
        self._log_psi_rand_end = layout.psi_rand_end

        # L_time parameters
        self._L_time_start = layout.L_time_start
        self._L_time_end = layout.L_time_end
        if self.emulator is not None and self.r_time > 0 and self.n_L_time_params > 0:
            self._L_time_diag_indices = as_index(layout.L_time_diag_positions)
            self._L_time_offdiag_indices = as_index(layout.L_time_offdiag_positions)
        else:
            self._L_time_diag_indices = None
            self._L_time_offdiag_indices = None

        # psi_time parameters (log scale)
        self._log_psi_time_start = layout.psi_time_start
        self._log_psi_time_end = layout.psi_time_end

    def _precompute_L_rand_indices(self):
        """Pre-compute indices for vectorized L_rand matrix construction.

        All lower-triangular elements of L_rand, including L_rand[0,0], are free
        parameters (see ParamLayout).
        """
        idx = self.layout._indices(self.device)
        self.L_rand_row_indices = idx["L_rand_rows"]
        self.L_rand_col_indices = idx["L_rand_cols"]
        self.L_rand_is_diagonal = idx["L_rand_is_diag"]

    def _precompute_L_time_indices(self):
        """Pre-compute indices for vectorized L_time matrix construction.

        The L_time[0,0] element is anchored to 1, so it is excluded from the
        parameter vector (see ParamLayout).
        """
        idx = self.layout._indices(self.device)
        self.L_time_row_indices = idx["L_time_rows"]
        self.L_time_col_indices = idx["L_time_cols"]
        self.L_time_is_diagonal = idx["L_time_is_diag"]

    def _get_parameter_lower_bounds(self) -> torch.Tensor:
        """
        Get lower bounds for each parameter position.
        
        Returns
        -------
        lower_bounds : (n_params,) tensor
            Lower bound for each parameter. -inf for unbounded parameters.
        """
        lower_bounds = torch.full(
            (self.n_params,), 
            float('-inf'), 
            dtype=torch.float32, 
            device=self.device
        )
        
        # L_rand diagonal elements (log scale) - only if r_rand > 0
        if self._L_rand_diag_indices is not None and len(self._L_rand_diag_indices) > 0:
            L_rand_param_indices = self._L_rand_start + self._L_rand_diag_indices
            lower_bounds[L_rand_param_indices] = self.log_L_diag_lower_bound
        
        # log_psi_rand elements
        if self.n_psi_rand_params > 0:
            lower_bounds[self._log_psi_rand_start:self._log_psi_rand_end] = self.log_psi_lower_bound
        
        # L_time diagonal elements (log scale) - only if r_time > 0
        if self._L_time_diag_indices is not None and len(self._L_time_diag_indices) > 0:
            L_time_param_indices = self._L_time_start + self._L_time_diag_indices
            lower_bounds[L_time_param_indices] = self.log_L_diag_lower_bound
        
        # log_psi_time elements
        if self.n_psi_time_params > 0:
            lower_bounds[self._log_psi_time_start:self._log_psi_time_end] = self.log_psi_lower_bound
        
        return lower_bounds

    def perturb_starting_values(
        self, 
        param: torch.Tensor, 
        noise_std: float = 1.0, 
        max_attempts: int = 100
    ) -> torch.Tensor:
        """
        Perturb starting values with rejection sampling to respect bounds.
        
        Parameters
        ----------
        param : (n_params,) tensor
            Initial parameter vector.
        noise_std : float
            Standard deviation of Gaussian noise to add.
        max_attempts : int
            Maximum number of sampling attempts per parameter before raising error.
        
        Returns
        -------
        perturbed : (n_params,) tensor
            Perturbed parameter vector satisfying all bounds.
        
        Raises
        ------
        RuntimeError
            If any parameter fails to find a valid value within max_attempts.
        """
        lower_bounds = self._get_parameter_lower_bounds()
        
        # Initialize output as copy of original
        perturbed = param.clone()
        
        # Track which parameters still need valid samples
        needs_sample = torch.ones(self.n_params, dtype=torch.bool, device=self.device)
        
        for attempt in range(max_attempts):
            # Sample noise for all parameters that still need it
            noise = torch.randn(self.n_params, dtype=param.dtype, device=self.device) * noise_std
            
            # Compute candidates
            candidates = param + noise
            
            # Check which candidates satisfy bounds
            valid = candidates >= lower_bounds
            
            # Update perturbed values where (needs_sample AND valid)
            update_mask = needs_sample & valid
            perturbed = torch.where(update_mask, candidates, perturbed)
            
            # Update needs_sample
            needs_sample = needs_sample & ~valid
            
            # Early exit if all parameters have valid values
            if not needs_sample.any():
                break
        
        # Check if any parameters still need samples
        if needs_sample.any():
            n_failed = needs_sample.sum().item()
            failed_indices = torch.where(needs_sample)[0].tolist()
            raise RuntimeError(
                f"Failed to find valid perturbed values for {n_failed} parameters "
                f"after {max_attempts} attempts. Failed parameter indices: {failed_indices[:10]}..."
            )
        
        return perturbed

    def _compute_barrier_penalty(self, param: torch.Tensor) -> torch.Tensor:
        """
        Compute barrier penalty to keep log-scale parameters above lower bounds.

        Uses log barrier: -mean(log(param - lower_bound))

        Applied to:
        - log_psi_rand (if present)
        - log-diagonal elements of L_rand (if present)
        - log_psi_time (if present)
        - log-diagonal elements of L_time (if present, excluding anchored)

        Parameters
        ----------
        param : torch.Tensor
            Flat parameter vector.

        Returns
        -------
        penalty : torch.Tensor
            Scalar barrier penalty (unscaled by lambda_reg or n_households).
        """
        device = param.device
        dtype = param.dtype
        penalty = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        log_psi_lower = torch.tensor(self.log_psi_lower_bound, device=device, dtype=dtype)
        log_L_diag_lower = torch.tensor(self.log_L_diag_lower_bound, device=device, dtype=dtype)

        # Barrier on log_psi_rand
        if self.n_psi_rand_params > 0:
            log_psi_rand = param[self._log_psi_rand_start:self._log_psi_rand_end]
            log_psi_rand_dist = log_psi_rand - log_psi_lower
            penalty = penalty - torch.log(log_psi_rand_dist).mean()

        # Barrier on log-diagonal elements of L_rand
        if self._L_rand_diag_indices is not None and len(self._L_rand_diag_indices) > 0:
            L_rand_params = param[self._L_rand_start:self._L_rand_end]
            log_L_rand_diag = L_rand_params[self._L_rand_diag_indices]
            log_L_rand_diag_dist = log_L_rand_diag - log_L_diag_lower
            penalty = penalty - torch.log(log_L_rand_diag_dist).mean()

        # Barrier on log_psi_time
        if self.n_psi_time_params > 0:
            log_psi_time = param[self._log_psi_time_start:self._log_psi_time_end]
            log_psi_time_dist = log_psi_time - log_psi_lower
            penalty = penalty - torch.log(log_psi_time_dist).mean()

        # Barrier on log-diagonal elements of L_time (excluding anchored [0,0])
        if self._L_time_diag_indices is not None and len(self._L_time_diag_indices) > 0:
            L_time_params = param[self._L_time_start:self._L_time_end]
            log_L_time_diag = L_time_params[self._L_time_diag_indices]
            log_L_time_diag_dist = log_L_time_diag - log_L_diag_lower
            penalty = penalty - torch.log(log_L_time_diag_dist).mean()

        return penalty

    def _compute_prior_penalty(self, param: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian pseudo-prior penalty.

        Applied to:
        - log_psi_rand: N(0, log_psi_prior_std^2) prior
        - Off-diagonal elements of L_rand: N(0, gamma_prior_std^2) prior
        - log_psi_time: N(0, log_psi_prior_std^2) prior
        - Off-diagonal elements of L_time: N(0, gamma_prior_std^2) prior

        Parameters
        ----------
        param : torch.Tensor
            Flat parameter vector.

        Returns
        -------
        penalty : torch.Tensor
            Scalar prior penalty (unscaled by lambda_reg or n_households).
        """
        device = param.device
        dtype = param.dtype
        penalty = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)

        log_psi_var = self.log_psi_prior_std ** 2
        gamma_var = self.gamma_prior_std ** 2

        # Prior on log_psi_rand
        if self.n_psi_rand_params > 0:
            log_psi_rand = param[self._log_psi_rand_start:self._log_psi_rand_end]
            penalty = penalty + 0.5 * (log_psi_rand ** 2).mean() / log_psi_var

        # Prior on off-diagonal elements of L_rand (factor loadings)
        if self._L_rand_offdiag_indices is not None and len(self._L_rand_offdiag_indices) > 0:
            L_rand_params = param[self._L_rand_start:self._L_rand_end]
            L_rand_offdiag = L_rand_params[self._L_rand_offdiag_indices]
            penalty = penalty + 0.5 * (L_rand_offdiag ** 2).mean() / gamma_var

        # Prior on log_psi_time
        if self.n_psi_time_params > 0:
            log_psi_time = param[self._log_psi_time_start:self._log_psi_time_end]
            penalty = penalty + 0.5 * (log_psi_time ** 2).mean() / log_psi_var

        # Prior on off-diagonal elements of L_time (factor loadings)
        if self._L_time_offdiag_indices is not None and len(self._L_time_offdiag_indices) > 0:
            L_time_params = param[self._L_time_start:self._L_time_end]
            L_time_offdiag = L_time_params[self._L_time_offdiag_indices]
            penalty = penalty + 0.5 * (L_time_offdiag ** 2).mean() / gamma_var

        return penalty

    def compute_penalty(self, param: torch.Tensor) -> torch.Tensor:
        """
        Compute the raw (unscaled) total penalty r(θ).

        This is the sum of barrier and prior penalties, without scaling by
        lambda_reg or n_households.

        Parameters
        ----------
        param : torch.Tensor
            Flat parameter vector.

        Returns
        -------
        penalty : torch.Tensor
            Scalar raw penalty r(θ).
        """
        barrier = self._compute_barrier_penalty(param)
        prior = self._compute_prior_penalty(param)
        zero = (param * 0.).mean()  # This ensures that param is always connected in the computation graph
        return zero + barrier + prior

    def _unpack_params(self, param):
        """
        Unpack flat parameter vector into components (vectorized).

        The layout is defined by ParamLayout (self.layout).

        Returns
        -------
        If n_factors_rand >= 0:
            random_means : (k,) tensor
            fixed_coefs : (n_fixed_features,) tensor
            L_rand : (k, r_rand) tensor or None (None if r_rand <= 0)
            psi_rand : (k,) tensor or None
            L_time : (k_time, r_time) tensor or None
            psi_time : (k_time,) tensor or None
        If n_factors_rand < 0:
            intercepts : (n_brand_forms - 1,) tensor
            fixed_coefs : (n_fixed_features,) tensor
            L_rand : None
            psi_rand : None
            L_time : (k_time, r_time) tensor or None
            psi_time : (k_time,) tensor or None
        """
        return self.layout.unpack(param)

    def _build_Sigma_time_full(self, L_time, psi_time, dtype, device):
        """
        Build the full (n_brand_forms, n_brand_forms) time-specific covariance matrix.

        Parameters
        ----------
        L_time : (k_time, r_time) tensor or None
            Factor loading matrix for time-specific errors.
        psi_time : (k_time,) tensor
            Uniquenesses for time-specific errors.
        dtype : torch.dtype
            Data type for the output tensor.
        device : torch.device
            Device for the output tensor.

        Returns
        -------
        Sigma_time_full : (n_brand_forms, n_brand_forms) tensor
            Full covariance matrix with zeros in first row/column.
        """
        return self.layout.build_Sigma_time_full(L_time, psi_time, dtype, device)

    def _compute_probit_log_probs_binary(self, utilities, Sigma_c):
        """
        Compute closed-form probit log probabilities for binary choice.

        For two alternatives, the probit choice probability has a closed-form
        solution using the normal CDF. Let u_0 and u_1 be the utilities of the
        two alternatives. The probability of choosing alternative 1 is:

            P(choice = 1) = Φ((u_1 - u_0) / σ)

        where σ² = Var(u_1 - u_0) = Σ[0,0] + Σ[1,1] - 2*Σ[0,1].

        Parameters
        ----------
        utilities : (..., 2) tensor
            Utilities for the two alternatives. Can have arbitrary leading dimensions.
        Sigma_c : (..., 2, 2) tensor
            Covariance matrix for the two alternatives. Must be broadcastable with
            utilities (excluding the last dimension).

        Returns
        -------
        log_probs : (..., 2) tensor
            Log probabilities for each alternative.
        """
        # Compute variance of utility difference: Var(u_1 - u_0)
        # Using differencing vector d = [-1, 1]: σ² = d' Σ d
        # Sigma_c has shape (..., 2, 2)
        var_diff = Sigma_c[..., 0, 0] + Sigma_c[..., 1, 1] - 2 * Sigma_c[..., 0, 1]
        std_diff = torch.sqrt(var_diff)

        # Compute z = (u_1 - u_0) / σ
        utility_diff = utilities[..., 1] - utilities[..., 0]
        z = utility_diff / std_diff

        # Compute log probabilities using log_ndtr (log of standard normal CDF)
        # log_ndtr computes log(Φ(x)) in a numerically stable way
        log_prob_1 = torch.special.log_ndtr(z)
        log_prob_0 = torch.special.log_ndtr(-z)

        # Stack into (..., 2) tensor
        log_probs = torch.stack([log_prob_0, log_prob_1], dim=-1)

        return log_probs

    def _compute_probit_log_prob_chosen_hermite(
        self, 
        utilities: torch.Tensor, 
        choice_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of chosen alternative using Gauss-Hermite quadrature.
        
        For probit models with c > 2 alternatives.
        
        Parameters
        ----------
        utilities : (n, c) or (n, n_draws, c) tensor
            Utilities for all alternatives.
        choice_idx : (n,) tensor
            Index of chosen alternative for each observation.
        
        Returns
        -------
        log_prob_chosen : (n,) or (n, n_draws) tensor
            Log probability of the chosen alternative.
        """
        # Determine if we have draws dimension
        has_draws = utilities.dim() == 3
        
        if has_draws:
            n, n_draws_dim, c = utilities.shape
        else:
            n, c = utilities.shape
            n_draws_dim = None
        
        # Get utility of chosen alternative
        if has_draws:
            # choice_idx: (n,) -> (n, n_draws, 1)
            choice_idx_expanded = choice_idx[:, None, None].expand(n, n_draws_dim, 1)
            u_chosen = torch.gather(utilities, 2, choice_idx_expanded).squeeze(2)  # (n, n_draws)
        else:
            # choice_idx: (n,) -> (n, 1)
            u_chosen = torch.gather(utilities, 1, choice_idx[:, None]).squeeze(1)  # (n,)
        
        # Hermite nodes and log weights
        # nodes: (K,), log_weights: (K,) where K = HERMITE_NODES
        nodes = self.hermite_nodes  # (K,)
        log_weights = self.hermite_log_weights  # (K,) includes -0.5*log(pi)
        K = nodes.shape[0]
        
        # Compute shifted utility of chosen alternative at each quadrature node
        # u_chosen + sqrt(2) * node_k for each node k
        # sigma_chosen = 1 for standard probit
        sqrt2 = torch.sqrt(torch.tensor(2.0, dtype=utilities.dtype, device=utilities.device))
        
        if has_draws:
            # u_chosen: (n, n_draws) -> (n, n_draws, K)
            u_chosen_shifted = u_chosen[:, :, None] + sqrt2 * nodes[None, None, :]  # (n, n_draws, K)
        else:
            # u_chosen: (n,) -> (n, K)
            u_chosen_shifted = u_chosen[:, None] + sqrt2 * nodes[None, :]  # (n, K)
        
        # For each non-chosen alternative i, compute log Phi((u_chosen_shifted - u_i) / sigma_i)
        # where sigma_i = 1
        # Sum over all i != chosen
        
        # Create mask for non-chosen alternatives
        # alternative_indices: (c,)
        alternative_indices = torch.arange(c, device=utilities.device)
        
        if has_draws:
            # choice_idx: (n,) -> (n, 1, 1, 1) for broadcasting
            # alternative_indices: (c,) -> (1, 1, c)
            # mask: (n, 1, c) - True for non-chosen alternatives
            mask = alternative_indices[None, None, :] != choice_idx[:, None, None]  # (n, 1, c)
            
            # utilities: (n, n_draws, c)
            # u_chosen_shifted: (n, n_draws, K)
            # Compute differences: (n, n_draws, K, c)
            # u_chosen_shifted[:, :, :, None] - utilities[:, :, None, :]
            diff = u_chosen_shifted[:, :, :, None] - utilities[:, :, None, :]  # (n, n_draws, K, c)
            
            # Compute log CDF for all alternatives
            log_cdf = torch.special.log_ndtr(diff)  # (n, n_draws, K, c)
            
            # Mask out chosen alternative and sum
            # mask: (n, 1, c) -> (n, 1, 1, c)
            mask_expanded = mask[:, :, None, :]  # (n, 1, 1, c)
            log_cdf_masked = torch.where(mask_expanded, log_cdf, torch.zeros_like(log_cdf))
            log_prod_cdf = log_cdf_masked.sum(dim=3)  # (n, n_draws, K)
            
            # Combine with log weights using logsumexp
            # log_weights: (K,) -> (1, 1, K)
            log_integrand = log_weights[None, None, :] + log_prod_cdf  # (n, n_draws, K)
            log_prob_chosen = torch.logsumexp(log_integrand, dim=2)  # (n, n_draws)
            
        else:
            # Similar logic but without n_draws dimension
            # mask: (n, c) - True for non-chosen alternatives
            mask = alternative_indices[None, :] != choice_idx[:, None]  # (n, c)
            
            # utilities: (n, c)
            # u_chosen_shifted: (n, K)
            # diff: (n, K, c)
            diff = u_chosen_shifted[:, :, None] - utilities[:, None, :]  # (n, K, c)
            
            # Compute log CDF
            log_cdf = torch.special.log_ndtr(diff)  # (n, K, c)
            
            # Mask out chosen alternative and sum
            # mask: (n, c) -> (n, 1, c)
            mask_expanded = mask[:, None, :]  # (n, 1, c)
            log_cdf_masked = torch.where(mask_expanded, log_cdf, torch.zeros_like(log_cdf))
            log_prod_cdf = log_cdf_masked.sum(dim=2)  # (n, K)
            
            # Combine with log weights using logsumexp
            # log_weights: (K,) -> (1, K)
            log_integrand = log_weights[None, :] + log_prod_cdf  # (n, K)
            log_prob_chosen = torch.logsumexp(log_integrand, dim=1)  # (n,)
        
        return log_prob_chosen

    def get_param_names(self, intercept_names=None, numeric_names=None):
        """
        Generate descriptive names for all parameters.

        Parameters
        ----------
        intercept_names : list of str, optional
            Names for brand-form intercepts (excluding reference brand-form).
            Should have length (n_brand_forms - 1).
        numeric_names : list of str, optional
            Names for numeric features (non-intercept columns of X).
            Should have length (X_dim - 1).

        Returns
        -------
        names : list of str
            Parameter names in the same order as the parameter vector.
        """
        return self.layout.param_names(intercept_names, numeric_names)

    def loglike_household(self, param):
        """
        Compute household-level log-likelihoods.

        Parameters
        ----------
        param : (n_params,) tensor
            Flat parameter vector (unconstrained).

        Returns
        -------
        loglike : (n_households,) tensor
            Log-likelihood for each household.
        """
        if self.r_rand < 0:
            ll = self._loglike_household_fixed(param)
        else:
            ll = self._loglike_household_mixed(param)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ll

    def _loglike_household_fixed(self, param):
        """Compute log-likelihoods for fixed effects only model (n_factors_rand < 0)."""
        intercepts, fixed_coefs, _, _, L_time, psi_time = self._unpack_params(param)

        # Build coefficient vector: (n_brand_forms, X_dim)
        coef_tensor = torch.zeros(
            self.n_brand_forms, self.X_dim,
            dtype=param.dtype, device=param.device
        )

        # Brand-form intercepts (first brand-form is reference)
        coef_tensor[1:, 0] = intercepts
        del intercepts

        # Fixed feature coefficients (same for all brand-forms)
        if self.fixed_feature_cols_t is not None and len(self.fixed_feature_cols) > 0:
            coef_tensor[:, self.fixed_feature_cols_t] = fixed_coefs
        del fixed_coefs

        # Build full time-specific covariance matrix if emulator is provided
        if self.emulator is not None:
            Sigma_time_full = self._build_Sigma_time_full(
                L_time, psi_time, param.dtype, param.device
            )
        del L_time, psi_time

        # Accumulate log-probabilities by household
        household_loglike = torch.zeros(
            self.n_households,
            dtype=param.dtype, device=param.device
        )

        for c in self.cs:
            data_dict_c = self.data_dict[c]
            alternative_ids_c = data_dict_c["alternative_ids"]
            X_c = data_dict_c["X"]
            choice_idx_c = data_dict_c["choice_idx"]
            household_id_c = data_dict_c["household_id"]

            n_c = alternative_ids_c.shape[0]

            # Gather coefficients for alternatives: (n_c, c, X_dim)
            coefs_for_alts = coef_tensor[alternative_ids_c]

            # Compute utilities: (n_c, c)
            utilities = (X_c * coefs_for_alts).sum(dim=2)
            del coefs_for_alts

            # Compute log choice probabilities
            if self.emulator is not None:
                row_idx = alternative_ids_c.unsqueeze(2).expand(-1, -1, c)  # (n_c, c, c)
                col_idx = alternative_ids_c.unsqueeze(1).expand(-1, c, -1)  # (n_c, c, c)
                Sigma_c = Sigma_time_full[row_idx, col_idx]  # (n_c, c, c)

                if (c == 2) and (self.emulator is not None) and (not _emulator_includes_c_star(self.emulator)):
                    # Use closed-form probit for binary choice (factorization-invariant models only)
                    log_probs = self._compute_probit_log_probs_binary(utilities, Sigma_c)
                    del utilities
                    chosen_log_probs = torch.gather(log_probs, 1, choice_idx_c[:, None]).squeeze(1)
                    del log_probs

                else:
                    # Use emulator for larger choice sets
                    self.emulator.eval()

                    v_star, s_star, c_star = canonicalize(utilities, Sigma_c)
                    del utilities, Sigma_c

                    logits = self.emulator(v_star, s_star, c_star)
                    del v_star, s_star, c_star

                    log_probs = F.log_softmax(logits, dim=1)
                    del logits
                    chosen_log_probs = torch.gather(log_probs, 1, choice_idx_c[:, None]).squeeze(1)
                    del log_probs

            elif self.model_family == "probit":
                # Probit without emulator
                if c == 2:
                    # Use closed-form for binary choice
                    Sigma_c = torch.eye(c, dtype=param.dtype, device=param.device)
                    Sigma_c = Sigma_c.unsqueeze(0).expand(n_c, -1, -1)  # (n_c, 2, 2)
                    log_probs = self._compute_probit_log_probs_binary(utilities, Sigma_c)
                    del utilities
                    chosen_log_probs = torch.gather(log_probs, 1, choice_idx_c[:, None]).squeeze(1)
                    del log_probs
                else:
                    # Use Gauss-Hermite quadrature for c > 2
                    chosen_log_probs = self._compute_probit_log_prob_chosen_hermite(
                        utilities, choice_idx_c
                    )
                    del utilities

            else:
                # Logit (default)
                log_probs = F.log_softmax(utilities, dim=1)
                del utilities
                chosen_log_probs = torch.gather(log_probs, 1, choice_idx_c[:, None]).squeeze(1)
                del log_probs

            # Accumulate into household-level log-likelihoods
            household_loglike.scatter_add_(0, household_id_c, chosen_log_probs)
            del chosen_log_probs

        return household_loglike

    def _loglike_household_mixed(self, param):
        """Compute log-likelihoods for mixed model (n_factors_rand >= 0)."""
        random_means, fixed_coefs, L_rand, psi_rand, L_time, psi_time = self._unpack_params(param)

        # Compute the square-root matrix for generating random effects
        if self.r_rand == 0:
            # Independent random effects: diagonal covariance
            # sqrt(diag(psi_rand)) is just diag(sqrt(psi_rand))
            sqrt_psi_rand = torch.sqrt(psi_rand)
            del psi_rand
            # We'll apply this directly as element-wise multiplication
            sqrt_cov_rand = None  # Signal to use diagonal approach
        elif self.include_uniquenesses_rand:
            # Compute covariance matrix: Σ = LL' + Ψ
            Sigma_rand = torch.matmul(L_rand, L_rand.T) + torch.diag(psi_rand)
            del L_rand, psi_rand

            # Compute Cholesky decomposition with jitter fallback
            jitter_values = [0, 1e-6, 1e-4, 1e-2, 1]
            chol_Sigma_rand = None

            for jitter in jitter_values:
                try:
                    if jitter > 0:
                        warnings.warn(
                            f"Cholesky decomposition failed. Adding jitter of {jitter} to diagonal."
                        )
                        Sigma_rand_jittered = Sigma_rand + jitter * torch.eye(
                            self.k, dtype=param.dtype, device=param.device
                        )
                        chol_Sigma_rand = torch.linalg.cholesky(Sigma_rand_jittered)
                        del Sigma_rand_jittered
                    else:
                        chol_Sigma_rand = torch.linalg.cholesky(Sigma_rand)
                    break
                except RuntimeError:
                    continue

            del Sigma_rand

            if chol_Sigma_rand is None:
                raise RuntimeError(
                    "Cholesky decomposition failed even with jitter of 1. "
                    "The covariance matrix may be severely ill-conditioned."
                )

            sqrt_cov_rand = chol_Sigma_rand
            del chol_Sigma_rand
        else:
            del psi_rand
            # Without uniquenesses, L_rand is the square-root matrix
            # Pad L_rand with zeros to match std_normal_draws dimension (k)
            sqrt_cov_rand = torch.zeros(self.k, self.k, dtype=param.dtype, device=param.device)
            sqrt_cov_rand[:, :self.r_rand] = L_rand
            del L_rand

        # Generate random effects: (n_households, n_draws, k)
        if self.r_rand == 0:
            # Diagonal case: element-wise multiplication
            random_effects = self.std_normal_draws * sqrt_psi_rand
            del sqrt_psi_rand
        else:
            random_effects = torch.matmul(self.std_normal_draws, sqrt_cov_rand.T)
            del sqrt_cov_rand

        # Add means: (n_households, n_draws, k)
        random_coefs = random_means + random_effects
        del random_means, random_effects

        # Build full time-specific covariance matrix if emulator is provided
        if self.emulator is not None:
            Sigma_time_full = self._build_Sigma_time_full(
                L_time, psi_time, param.dtype, param.device
            )
        del L_time, psi_time

        # Accumulate log-probabilities by household and draw
        household_draw_loglike = torch.zeros(
            self.n_households, self.n_draws,
            dtype=param.dtype, device=param.device
        )

        for c in self.cs:
            data_dict_c = self.data_dict[c]
            alternative_ids_c = data_dict_c["alternative_ids"]
            X_c = data_dict_c["X"]
            choice_idx_c = data_dict_c["choice_idx"]
            household_id_c = data_dict_c["household_id"]

            n_c = alternative_ids_c.shape[0]

            # Get coefficients for households in this batch: (n_c, n_draws, k)
            random_coefs_c = random_coefs[household_id_c]

            # Build parameter tensor: (n_c, n_draws, n_brand_forms, X_dim)
            param_tensor = torch.zeros(
                n_c, self.n_draws, self.n_brand_forms, self.X_dim,
                dtype=param.dtype, device=param.device
            )

            # Brand-form intercepts (first brand-form is reference)
            param_tensor[:, :, 1:, 0] = random_coefs_c[:, :, :(self.n_brand_forms - 1)]

            # Random feature coefficients (vectorized)
            if self.random_feature_cols_t is not None and len(self.random_feature_cols) > 0:
                n_rand_feat = len(self.random_feature_cols)
                rand_coefs = random_coefs_c[:, :, (self.n_brand_forms - 1):]
                rand_coefs_expanded = rand_coefs[:, :, None, :]
                del rand_coefs
                rand_coefs_broadcast = rand_coefs_expanded.expand(n_c, self.n_draws, self.n_brand_forms, n_rand_feat)
                del rand_coefs_expanded
                param_tensor[:, :, :, self.random_feature_cols_t] = rand_coefs_broadcast
                del rand_coefs_broadcast

            del random_coefs_c

            # Fixed feature coefficients (vectorized)
            if self.fixed_feature_cols_t is not None and len(self.fixed_feature_cols) > 0:
                param_tensor[:, :, :, self.fixed_feature_cols_t] = fixed_coefs

            # Gather parameters for alternatives in each choice set
            alt_ids_expanded = alternative_ids_c[:, None, :, None].expand(
                n_c, self.n_draws, c, self.X_dim
            )
            params_for_alts = torch.gather(param_tensor, 2, alt_ids_expanded)
            del param_tensor, alt_ids_expanded

            # Expand X: (n_c, n_draws, c, X_dim)
            X_c_expanded = X_c[:, None, :, :].expand(n_c, self.n_draws, c, self.X_dim)

            # Compute utilities: (n_c, n_draws, c)
            utilities = (X_c_expanded * params_for_alts).sum(dim=3)
            del X_c_expanded, params_for_alts

            # Compute log choice probabilities
            if self.emulator is not None:
                # Select covariance submatrix for alternatives in this choice set
                row_idx = alternative_ids_c.unsqueeze(2).expand(-1, -1, c)  # (n_c, c, c)
                col_idx = alternative_ids_c.unsqueeze(1).expand(-1, c, -1)  # (n_c, c, c)
                Sigma_c = Sigma_time_full[row_idx, col_idx]  # (n_c, c, c)

                if (c == 2) and (self.emulator is not None) and (not _emulator_includes_c_star(self.emulator)):
                    # Use closed-form probit for binary choice (factorization-invariant models only)
                    # utilities shape: (n_c, n_draws, 2); Sigma_c is shared across draws,
                    # so add a draws axis to make it (n_c, 1, 2, 2).
                    log_probs = self._compute_probit_log_probs_binary(
                        utilities, Sigma_c.unsqueeze(1)
                    )
                    del utilities
                    choice_idx_expanded = choice_idx_c[:, None, None].expand(n_c, self.n_draws, 1)
                    chosen_log_probs = torch.gather(log_probs, 2, choice_idx_expanded).squeeze(2)
                    del log_probs, choice_idx_expanded

                else:
                    # Use emulator for larger choice sets
                    self.emulator.eval()

                    # Sigma_c is (n_c, c, c), need (n_c, n_draws, c, c)
                    Sigma_c_expanded = Sigma_c.unsqueeze(1).expand(
                        -1, self.n_draws, -1, -1
                    )
                    del Sigma_c

                    # Reshape for emulator: (n_c * n_draws, c) and (n_c * n_draws, c, c)
                    utilities_flat = utilities.reshape(n_c * self.n_draws, c)
                    del utilities
                    Sigma_c_flat = Sigma_c_expanded.reshape(n_c * self.n_draws, c, c)
                    del Sigma_c_expanded

                    # Canonicalize and get emulator logits
                    v_star, s_star, c_star = canonicalize(utilities_flat, Sigma_c_flat)
                    del utilities_flat, Sigma_c_flat
                    logits_flat = self.emulator(v_star, s_star, c_star)
                    del v_star, s_star, c_star

                    # Reshape back: (n_c, n_draws, c)
                    logits = logits_flat.reshape(n_c, self.n_draws, c)
                    del logits_flat

                    # Log choice probabilities
                    log_probs = F.log_softmax(logits, dim=2)
                    del logits

                    # Extract log prob of chosen alternative: (n_c, n_draws)
                    choice_idx_expanded = choice_idx_c[:, None, None].expand(n_c, self.n_draws, 1)
                    chosen_log_probs = torch.gather(log_probs, 2, choice_idx_expanded).squeeze(2)
                    del log_probs, choice_idx_expanded

            elif self.model_family == "probit":
                # Probit without emulator
                if c == 2:
                    # Use closed-form for binary choice
                    # utilities: (n_c, n_draws, 2)
                    Sigma_c = torch.eye(c, dtype=param.dtype, device=param.device)
                    Sigma_c = Sigma_c.unsqueeze(0).unsqueeze(0).expand(n_c, self.n_draws, -1, -1)  # (n_c, n_draws, 2, 2)
                    log_probs = self._compute_probit_log_probs_binary(utilities, Sigma_c)
                    # log_probs: (n_c, n_draws, 2)
                    del utilities
                    choice_idx_expanded = choice_idx_c[:, None, None].expand(n_c, self.n_draws, 1)
                    chosen_log_probs = torch.gather(log_probs, 2, choice_idx_expanded).squeeze(2)
                    del log_probs, choice_idx_expanded
                else:
                    # Use Gauss-Hermite quadrature for c > 2
                    # utilities: (n_c, n_draws, c)
                    chosen_log_probs = self._compute_probit_log_prob_chosen_hermite(
                        utilities, choice_idx_c
                    )
                    del utilities

            else:
                # Standard logit
                log_probs = F.log_softmax(utilities, dim=2)
                del utilities
                choice_idx_expanded = choice_idx_c[:, None, None].expand(n_c, self.n_draws, 1)
                chosen_log_probs = torch.gather(log_probs, 2, choice_idx_expanded).squeeze(2)
                del log_probs, choice_idx_expanded

            # Accumulate into household-level log-likelihoods
            household_draw_loglike.scatter_add_(
                0,
                household_id_c[:, None].expand(n_c, self.n_draws),
                chosen_log_probs
            )
            del chosen_log_probs

        del random_coefs, fixed_coefs

        # Compute household log-likelihoods using logsumexp
        household_loglike = torch.logsumexp(household_draw_loglike, dim=1) - torch.log(
            torch.tensor(self.n_draws, dtype=param.dtype, device=param.device)
        )
        del household_draw_loglike

        return household_loglike

    def loss(self, param):
        """
        Compute the unpenalized loss (negative mean log-likelihood).

        Parameters
        ----------
        param : torch.Tensor
            Flat parameter vector.

        Returns
        -------
        loss : torch.Tensor
            Scalar loss value.
        """
        l = -self.loglike_household(param).mean()
        return l


def _compute_penalty_hessian_mixed_model(
    params: torch.Tensor,
    model,
) -> torch.Tensor:
    """
    Compute the Hessian of the raw penalty r(θ) for a MixedModel.
    
    Parameters
    ----------
    params : torch.Tensor
        Flat parameter vector.
    model : MixedModel
        A MixedModel instance (used for penalty computation structure).
        
    Returns
    -------
    hessian : torch.Tensor
        Hessian matrix of shape (n_params, n_params).
    """
    params_flat = params.detach().flatten().clone().requires_grad_(True)
    
    def raw_penalty_fn(p):
        return model.compute_penalty(p)
    
    hessian = torch.func.hessian(raw_penalty_fn)(params_flat)
    # params_flat requires grad, so the Hessian is attached to the autograd
    # graph; detach so the returned covariance is a plain tensor.
    return hessian.detach()


def compute_opg_covariance(
    params: torch.Tensor,
    models: list,
    eigenvalue_floor: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Outer Product of Gradients (OPG) covariance estimator,
    adjusted for regularization penalties.
    
    The covariance is estimated as:
        [J'J + lambda_reg * ∇²r(θ)]^{-1}
    
    where J'J is the sum of outer products of household-level scores and
    ∇²r(θ) is the Hessian of the raw penalty r(θ) = MixedModel.compute_penalty.

    Sign convention: the optimizers minimize
        F(θ) = -(1/N) Σ_i ℓ_i(θ) + lambda_reg * r(θ) / N,
    i.e. r(θ) is *added to the loss* (negative log barrier plus negative log
    pseudo-prior). The Hessian of N·F is -Σ_i ∇²ℓ_i(θ) + lambda_reg * ∇²r(θ),
    and J'J approximates the first term. In the paper's notation the penalty is
    added to the log likelihood (r_paper = -r), giving the equivalent
    {n Ĵ - lambda_reg ∇² r_paper}^{-1}.
    
    Parameters
    ----------
    params : torch.Tensor
        Parameter vector at which to compute the covariance.
    models : list
        List of MixedModel instances, one per batch.
    eigenvalue_floor : float
        Minimum eigenvalue for matrix inversion. Default is 1e-8.
        
    Returns
    -------
    covariance : torch.Tensor
        Estimated covariance matrix of shape (n_params, n_params).
    """
    params_flat = params.detach().flatten()
    n_params = params_flat.shape[0]
    device = params_flat.device
    dtype = params_flat.dtype
    
    # Infer penalty parameters from models
    lambda_reg = models[0].lambda_reg
    
    # Verify consistency across models
    for model in models[1:]:
        if model.lambda_reg != lambda_reg:
            raise ValueError(
                "All models must have the same lambda_reg. "
                f"Found {lambda_reg} and {model.lambda_reg}."
            )
    
    def compute_batch_JtJ(model, p):
        """Compute J'J contribution for a single batch."""
        def batch_loglike_fn(p_inner):
            return model.loglike_household(p_inner)
        
        # Compute Jacobian: (batch_n_households, n_params)
        scores_batch = torch.func.jacrev(batch_loglike_fn)(p)
        
        # Compute outer product and detach immediately
        batch_JtJ = (scores_batch.T @ scores_batch).detach().clone()
        
        return batch_JtJ
    
    # Accumulate J'J across batches
    JtJ = torch.zeros(n_params, n_params, device=device, dtype=dtype)
    
    for model in models:
        # Compute batch contribution with no_grad context for safety
        with torch.no_grad():
            batch_JtJ = compute_batch_JtJ(model, params_flat)
        
        # Accumulate
        JtJ += batch_JtJ
        
        # Explicit cleanup
        del batch_JtJ
    
    # Compute penalty Hessian (use first model as representative)
    penalty_hessian = _compute_penalty_hessian_mixed_model(params_flat, models[0])
    
    # Compute adjusted information matrix (Hessian of the penalized objective,
    # with J'J standing in for the negative log-likelihood Hessian):
    # I_total = J'J + lambda_reg * ∇²r(θ)
    I_total = JtJ + lambda_reg * penalty_hessian
    
    # Compute covariance as I_total^{-1}
    covariance = robust_matrix_inverse(I_total, eigenvalue_floor)
    
    return covariance


def split_data_by_household(
    data_dict: dict, 
    std_normal_draws_full: torch.Tensor,
    n_batches: int = 10,
) -> list:
    """
    Split data_dict into batches by household and include random draws.
    
    Households are sorted by number of decision points and assigned to batches
    in round-robin fashion to balance workload across batches.
    
    Parameters
    ----------
    data_dict : dict
        Dictionary with choice set sizes as keys, containing alternative_ids,
        X, choice_idx, and household_id for each group.
    std_normal_draws_full : torch.Tensor
        Full random draws tensor of shape (n_households, n_draws, max_k).
    n_batches : int
        Number of batches to create. Default is 10.
        
    Returns
    -------
    batch_data_dicts : list of dict
        List of n_batches data dicts, each containing a subset of households.
        Each dict has:
        - "n_households": count of households in that batch
        - "std_normal_draws": random draws for households in this batch
        - "original_household_ids": mapping back to original household IDs
    """
    # Infer device from input data
    first_c = next(c for c in data_dict if isinstance(c, int))
    device = data_dict[first_c]["household_id"].device
    
    # Infer n_households from data_dict
    max_household_id = -1
    for c in data_dict:
        if not isinstance(c, int):
            continue
        household_ids = data_dict[c]["household_id"]
        max_in_batch = household_ids.max().item()
        if max_in_batch > max_household_id:
            max_household_id = max_in_batch
    n_households = max_household_id + 1
    
    # Count observations per household
    obs_counts = torch.zeros(n_households, dtype=torch.long, device=device)
    for c in data_dict:
        if not isinstance(c, int):
            continue
        household_ids = data_dict[c]["household_id"]
        counts_c = torch.bincount(household_ids, minlength=n_households)
        obs_counts += counts_c
        
    # Sort households by number of observations (descending for better balance)
    sorted_indices = torch.argsort(obs_counts, descending=True)
    
    # Assign households to batches in round-robin fashion
    batch_assignments = [[] for _ in range(n_batches)]
    for i, hh_id in enumerate(sorted_indices.tolist()):
        batch_idx = i % n_batches
        batch_assignments[batch_idx].append(hh_id)
    
    # Create mapping tensors for vectorized operations
    batch_idx_map = torch.zeros(n_households, dtype=torch.long, device=device)
    new_hh_id_map = torch.zeros(n_households, dtype=torch.long, device=device)
    
    for batch_idx, hh_ids in enumerate(batch_assignments):
        for new_id, orig_id in enumerate(hh_ids):
            batch_idx_map[orig_id] = batch_idx
            new_hh_id_map[orig_id] = new_id
    
    # Distribute observations to batches using tensor indexing
    batch_data_dicts = [{} for _ in range(n_batches)]
    
    for c in data_dict:
        if not isinstance(c, int):
            continue
        data_c = data_dict[c]
        alternative_ids = data_c["alternative_ids"]
        X = data_c["X"]
        choice_idx = data_c["choice_idx"]
        household_id = data_c["household_id"]
        
        # Get batch assignment for each observation
        obs_batch_idx = batch_idx_map[household_id]
        obs_new_hh_id = new_hh_id_map[household_id]
        
        # For each batch, select observations belonging to that batch
        for batch_idx in range(n_batches):
            mask = obs_batch_idx == batch_idx
            
            if not mask.any():
                # No observations for this choice set size in this batch
                continue
            
            batch_data_dicts[batch_idx][c] = {
                "alternative_ids": alternative_ids[mask],
                "X": X[mask],
                "choice_idx": choice_idx[mask],
                "household_id": obs_new_hh_id[mask],
            }
    
    # Add n_households, random draws, and original IDs to each batch dict
    for batch_idx in range(n_batches):
        orig_hh_ids = batch_assignments[batch_idx]
        batch_data_dicts[batch_idx]["n_households"] = len(orig_hh_ids)
        batch_data_dicts[batch_idx]["original_household_ids"] = orig_hh_ids
        
        # Extract random draws for these households
        # Shape: (batch_n_households, n_draws, max_k)
        batch_data_dicts[batch_idx]["std_normal_draws"] = std_normal_draws_full[orig_hh_ids]
    
    return batch_data_dicts


def empty_cache(device):
    if device == 'mps' or (hasattr(device, 'type') and device.type == 'mps'):
        torch.mps.empty_cache()
    elif device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
        torch.cuda.empty_cache()
    # No-op for CPU


@dataclass
class OptimizationConfig:
    max_iter: int = 2000
    loss_tol: float = 1e-4
    history_size: int = 50
    convergence_window: int = 50


class BatchedManualLBFGS:
    """
    L-BFGS optimizer that computes gradients in batches over household groups.
    Uses finite penalty checks instead of Armijo line search.
    """
    
    def __init__(
        self,
        params: torch.Tensor,
        history_size: int = 50,
        min_step: float = 1e-8,
        max_step: float = 1.0,
        max_ls_iter: int = 50,
        verbose: bool = False,
    ):
        self.params = params.clone()
        self.device = params.device
        self.history_size = history_size
        self.min_step = min_step
        self.max_step = max_step
        self.max_ls_iter = max_ls_iter
        self.verbose = verbose
        
        self.lr = 1.0
        
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
    
    def _compute_batched_loss_and_grad(
        self,
        params: torch.Tensor,
        models: list,
        total_n_households: int,
    ) -> Tuple[float, torch.Tensor, bool, float]:
        """
        Compute loss and gradient by accumulating over batches.
        
        Parameters
        ----------
        params : torch.Tensor
            Current parameter vector (requires_grad should be False).
        models : list
            List of MixedModel instances, one per batch.
        total_n_households : int
            Total number of households across all batches.
            
        Returns
        -------
        total_loss : float
            Total loss (negative mean log-likelihood).
        total_grad : torch.Tensor
            Total gradient.
        is_finite : bool
            Whether the loss is finite.
        penalty_scaled : float
            Scaled penalty value.
        """
        
        def compute_batch(model, params):
            """Compute weighted loss and gradient for a single batch."""
            param_batch = params.clone().detach().requires_grad_(True)
            batch_loss = model.loss(param_batch)
            
            if not torch.isfinite(batch_loss):
                return None, None, False
            
            batch_grad = torch.autograd.grad(batch_loss, param_batch)[0]
            
            batch_n_households = model.n_households
            batch_weight = batch_n_households / total_n_households
            
            weighted_loss = batch_loss.item() * batch_weight
            weighted_grad = batch_grad.detach() * batch_weight
            
            return weighted_loss, weighted_grad, True
        
        total_loss = 0.0
        total_grad = torch.zeros_like(params)
        
        for i, model in enumerate(models):
            empty_cache(self.device)
            weighted_loss, weighted_grad, is_finite = compute_batch(model, params)
            
            if not is_finite:
                return total_loss, total_grad, False, 0.0
            
            total_loss += weighted_loss
            total_grad += weighted_grad
            del weighted_loss, weighted_grad, is_finite

        # Compute penalty
        param_penalty = params.clone().detach().requires_grad_(True)
        penalty = model.compute_penalty(param_penalty)
        if not torch.isfinite(penalty):
            return total_loss, total_grad, False, 0.0

        penalty_scaled = penalty * model.lambda_reg / total_n_households
        penalty_grad = torch.autograd.grad(penalty_scaled, param_penalty)[0]
        total_loss += penalty_scaled.item()
        total_grad += penalty_grad.detach()
        
        return total_loss, total_grad, True, penalty_scaled.item()
    
    def _find_finite_step(
        self,
        models: list,
        params: torch.Tensor,
        direction: torch.Tensor,
    ) -> Tuple[float, torch.Tensor, bool]:
        """
        Find a step size that results in finite penalty.
        
        Parameters
        ----------
        models : list
            List of MixedModel instances.
        params : torch.Tensor
            Current parameter vector.
        direction : torch.Tensor
            Search direction.
            
        Returns
        -------
        step : float
            Step size taken.
        new_params : torch.Tensor
            New parameter values.
        success : bool
            Whether a finite step was found.
        """
        initial_step = self.max_step * self.lr
        step = initial_step
        
        for _ in range(self.max_ls_iter):
            new_params = params + step * direction
            
            with torch.no_grad():
                penalty = models[-1].compute_penalty(new_params)
            
            if torch.isfinite(penalty):
                return step, new_params, True
            
            step *= 0.5
            if step < self.min_step:
                break
        
        # All halvings exhausted
        warnings.warn("Step halving failed to find finite penalty; accepting step anyway")
        new_params = params + self.min_step * direction
        return self.min_step, new_params, False
    
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
        models: list,
        total_n_households: int,
        current_loss: float,
        current_grad: torch.Tensor
    ) -> Tuple[torch.Tensor, float, torch.Tensor, bool, float]:
        """
        Perform one L-BFGS optimization step.
        
        Parameters
        ----------
        models : list
            List of MixedModel instances, one per batch.
        total_n_households : int
            Total number of households across all batches.
        current_loss : float
            Current loss value.
        current_grad : torch.Tensor
            Current gradient.
            
        Returns
        -------
        new_params : torch.Tensor
            Updated parameters.
        new_loss : float
            New loss value.
        new_grad : torch.Tensor
            New gradient.
        loss_improved : bool
            Whether the loss improved compared to current_loss.
        penalty_scaled : float
            Scaled penalty value.
        """
        direction = self._two_loop_recursion(current_grad.flatten())
        
        step, new_params, ls_success = self._find_finite_step(
            models, self.params, direction
        )
        
        new_loss, new_grad, is_finite, penalty_scaled = self._compute_batched_loss_and_grad(
            new_params, models, total_n_households
        )
        
        # Revert if loss is non-finite or worse by 1% or more
        if not is_finite or new_loss > current_loss * 1.01:
            new_params = self.params.clone()
            new_loss = current_loss
            new_grad = current_grad.clone()
            loss_improved = False
        else:
            loss_improved = (new_loss < current_loss)
        
        # Update history only if loss improved
        if loss_improved:
            s = new_params - self.params
            y = new_grad - current_grad
            self._update_history(s, y)
        
        self.params = new_params.clone()
        
        return new_params, new_loss, new_grad, loss_improved, penalty_scaled
    

class BatchedRobustLBFGSOptimizer:
    """
    Robust L-BFGS optimizer that operates on batches of households.
    Uses finite penalty checks and adaptive learning rate.
    """
    
    def __init__(
        self,
        config: OptimizationConfig,
        min_step: float = 1e-8,
        max_step: float = 1.0,
        min_lr: float = 1e-6,
        verbose: bool = False
    ):
        self.config = config
        self.min_step = min_step
        self.max_step = max_step
        self.min_lr = min_lr
        self.verbose = verbose
    
    def optimize(
        self,
        initial_params: torch.Tensor,
        models: list,
        best_loss_previous_fits: float = None,
        early_cutoff: int = None,
    ) -> OptimizationResult:
        """
        Optimize parameters using batched L-BFGS.
        
        Parameters
        ----------
        initial_params : torch.Tensor
            Initial parameter vector.
        models : list
            List of MixedModel instances, one per batch.
        best_loss_previous_fits : float, optional
            Best loss achieved by previous fits
        early_cutoff : int, optional
            If specified, function checks for improvement over previous starts.
            If the model has not improved, the model-fitting terminates early.
            
        Returns
        -------
        OptimizationResult
            Optimization result containing final parameters, loss, convergence
            status, gradient norm, and number of iterations.
        """
        device = initial_params.device

        # Compute total number of households
        total_n_households = sum(model.n_households for model in models)

        if self.verbose:
            print("Creating BatchedManualLBFGS")
        lbfgs = BatchedManualLBFGS(
            params=initial_params,
            history_size=self.config.history_size,
            min_step=self.min_step,
            max_step=self.max_step,
            verbose=self.verbose
        )
        
        # Compute initial loss and gradient
        if self.verbose:
            print("Compute batched loss and grad")
        current_loss, current_grad, is_finite, current_penalty_scaled = lbfgs._compute_batched_loss_and_grad(
            initial_params, models, total_n_households
        )
        
        if not is_finite:
            return OptimizationResult(
                params=initial_params,
                loss=float('inf'),
                converged=False,
                grad_norm=float('inf'),
                n_iter=0
            )
        
        # Initialize tracking variables
        best_loss = current_loss
        best_params = initial_params.clone()
        no_improvement_count = 0
        consecutive_worse_count = 0
        current_lr = 1.0
        converged = False

        if self.verbose:
            print("Starting optimization loop")
        pbar = trange(self.config.max_iter)
        for iteration in pbar:
            pbar.set_postfix_str(f"loss={current_loss:.4f}, penalty={current_penalty_scaled:.2e}, lr={current_lr:.2e}")
            
            # Set learning rate
            lbfgs.lr = current_lr
            
            # Take step
            new_params, new_loss, new_grad, loss_improved, new_penalty_scaled = lbfgs.step(
                models, total_n_households, current_loss, current_grad
            )
            empty_cache(device)
            
            if loss_improved:
                consecutive_worse_count = 0
                current_lr = min(current_lr * 1.5, 1.0)
            else:
                consecutive_worse_count += 1
                current_lr *= 0.5
                if current_lr < self.min_lr:
                    current_lr = self.min_lr
                    lbfgs.s_history.clear()
                    lbfgs.y_history.clear()
                    lbfgs.rho_history.clear()                    
            
            # Update best tracking
            if new_loss < best_loss - self.config.loss_tol:
                best_loss = new_loss
                best_params = new_params.clone()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Check convergence
            if no_improvement_count >= self.config.convergence_window:
                converged = True
                break

            if (early_cutoff is not None) and (best_loss_previous_fits is not None):
                if best_loss >= best_loss_previous_fits and iteration >= early_cutoff:
                    break
            
            # Update for next iteration
            current_loss = new_loss
            current_grad = new_grad
            current_penalty_scaled = new_penalty_scaled
        
        # Compute final gradient norm at best params
        _, final_grad, is_finite, _ = lbfgs._compute_batched_loss_and_grad(
            best_params, models, total_n_households
        )
        final_grad_norm = torch.norm(final_grad).item() if is_finite else float('inf')
        empty_cache(device)
        
        return OptimizationResult(
            params=best_params,
            loss=best_loss,
            converged=converged,
            grad_norm=final_grad_norm,
            n_iter=iteration + 1
        )
    

@dataclass
class AdamOptimizationConfig:
    max_iter: int = 2000
    loss_tol: float = 1e-4
    lr: float = 0.1
    lr_mult_start: float = 0.1
    lr_mult_step: float = 0.01
    lr_mult_max: float = 1.0
    betas: Tuple[float, float] = (0.9, 0.995)
    eps: float = 1e-8
    convergence_count: int = 50


class BatchedAdamOptimizer:
    """
    Adam optimizer that accumulates gradients across all batches, then takes one step.
    Learning rate multiplier increases each epoch until reaching lr_mult_max.
    Stops after convergence_count iterations without meaningful improvement over best.
    """
    
    def __init__(
        self,
        config: AdamOptimizationConfig,
        verbose: bool = False
    ):
        self.config = config
        self.verbose = verbose
    
    def _get_lr(self, epoch: int) -> float:
        """Calculate learning rate for given epoch (0-indexed)."""
        multiplier = self.config.lr_mult_start + epoch * self.config.lr_mult_step
        multiplier = min(multiplier, self.config.lr_mult_max)
        return self.config.lr * multiplier
    
    def optimize(
        self,
        initial_params: torch.Tensor,
        models: list,
        best_loss_previous_fits: float = None,
        early_cutoff: int = None,
        max_halvings: int = 20,
    ) -> OptimizationResult:
        device = initial_params.device
        
        params = initial_params.clone().detach().requires_grad_(True)
        
        # Check starting parameters
        initial_penalty = models[0].compute_penalty(params)
        if not torch.isfinite(initial_penalty):
            warnings.warn(
                f"Initial parameters produce non-finite penalty: {initial_penalty.item()}"
            )
        
        optimizer = torch.optim.Adam(
            [params],
            lr=self._get_lr(0),
            betas=self.config.betas,
            eps=self.config.eps,
            weight_decay=0.0
        )
        
        total_n_households = sum(model.n_households for model in models)
        
        best_loss = float('inf')
        best_params = initial_params.clone()
        no_improvement_count = 0
        converged = False
        
        if self.verbose:
            print("Starting Adam optimization")
        
        pbar = trange(self.config.max_iter, desc="Adam")
        for epoch in pbar:
            # Update learning rate
            lr = self._get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Zero gradients once at start of epoch
            optimizer.zero_grad()
            
            epoch_loss = 0.0
            
            for model in models:
                empty_cache(device)
                
                batch_loss = model.loss(params)
                
                if not torch.isfinite(batch_loss):
                    continue
                
                # Weight the loss for proper gradient scaling
                batch_weight = model.n_households / total_n_households
                weighted_loss = batch_loss * batch_weight
                
                # Accumulate gradients (no step yet)
                weighted_loss.backward()
                
                epoch_loss += batch_loss.item() * batch_weight
                
                del batch_loss, weighted_loss
    
            # Compute penalty
            penalty = model.compute_penalty(params)
            penalty_scaled = penalty * model.lambda_reg / total_n_households
            penalty_scaled.backward()
            epoch_loss += penalty_scaled.item()
            
            # Save position before stepping
            params_before = params.data.clone()
            
            # Take the full Adam step
            optimizer.step()
            
            # Compute the full update vector
            full_update = params.data - params_before
            
            # Step halving if penalty is non-finite
            scale = 1.0
            for _ in range(max_halvings):
                penalty_check = model.compute_penalty(params)
                if torch.isfinite(penalty_check):
                    break
                scale *= 0.5
                params.data = params_before + full_update * scale
            else:
                # All halvings exhausted; revert to original position
                warnings.warn(f"Step halving failed after {max_halvings} attempts; reverting step")
                params.data = params_before
            
            pbar.set_postfix_str(f"loss={epoch_loss:.4f}, penalty={penalty_scaled.item():.2e}")
            del penalty, penalty_scaled
            
            # Check for meaningful improvement over best
            if epoch_loss < best_loss - self.config.loss_tol:
                best_loss = epoch_loss
                best_params = params.clone().detach()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
                if no_improvement_count >= self.config.convergence_count:
                    converged = True
                    break

            # Break after many iterations and no improvement over previous fits
            if (early_cutoff is not None) and (best_loss_previous_fits is not None):
                if best_loss >= best_loss_previous_fits and epoch >= early_cutoff:
                    break

        
        final_grad_norm = self._compute_gradient_norm(
            best_params, models, total_n_households, device
        )
        
        return OptimizationResult(
            params=best_params,
            loss=best_loss,
            converged=converged,
            grad_norm=final_grad_norm,
            n_iter=epoch + 1
        )
    
    def _compute_gradient_norm(
        self,
        params: torch.Tensor,
        models: list,
        total_n_households: int,
        device: torch.device
    ) -> float:
        params_eval = params.clone().detach().requires_grad_(True)
        total_grad = torch.zeros_like(params_eval)
        
        for model in models:
            empty_cache(device)
            
            batch_loss = model.loss(params_eval)
            if torch.isfinite(batch_loss):
                batch_grad = torch.autograd.grad(
                    batch_loss, params_eval, retain_graph=False
                )[0]
                batch_weight = model.n_households / total_n_households
                total_grad += batch_grad.detach() * batch_weight
            
            del batch_loss

        # Compute penalty
        penalty = model.compute_penalty(params_eval)
        penalty_scaled = penalty * model.lambda_reg / total_n_households
        if torch.isfinite(penalty_scaled):
            penalty_grad = torch.autograd.grad(penalty_scaled, params_eval, retain_graph=False)[0]
            total_grad += penalty_grad.detach()
        
        return torch.norm(total_grad).item()
    

def create_model_summary(
    model: MixedModel,
    result: OptimizationResult,
    param_names: list,
    model_name: str,
    covariance=None,
) -> dict:
    """
    Create a comprehensive model summary for JSON serialization.
    
    Parameters
    ----------
    model : MixedModel
        The fitted model.
    result : OptimizationResult
        The optimization result.
    param_names : list
        List of parameter names.
    model_name : str
        Descriptive name for the model.
    covariance : torch.Tensor or np.ndarray, optional
        Joint covariance matrix of the parameters, shape (n_params, n_params),
        e.g. from compute_opg_covariance. Saved under "covariance" if provided.
        
    Returns
    -------
    summary : dict
        JSON-serializable dictionary containing model summary.
    """
    n = sum(
        model.data_dict[c]["X"].shape[0] 
        for c in model.cs
    )
    k = model.n_params
    log_likelihood = -result.loss * model.n_households
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    
    params_np = result.params.detach().cpu().numpy()
    
    summary = {
        "model_name": model_name,
        "n_observations": int(n),
        "n_households": int(model.n_households),
        "n_total_households": int(model.n_total_households),
        "n_params": int(k),
        "log_likelihood": float(log_likelihood),
        "aic": float(aic),
        "bic": float(bic),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "grad_norm": float(result.grad_norm),
        "loss": float(result.loss),
        # Model configuration
        "config": {
            "n_brand_forms": model.n_brand_forms,
            "X_dim": model.X_dim,
            "n_factors_rand": model.n_factors_rand,
            "n_factors_time": model.n_factors_time,
            "include_uniquenesses_rand": model.include_uniquenesses_rand,
            "random_feature_indices": model.random_feature_indices,
            "n_draws": model.n_draws,
            "emulator_name": None,  # To be set by caller if applicable
            "model_family": model.model_family,
            "anchored_rand": model.anchored_rand,
            "has_time_params": model.emulator is not None,
            "param_version": PARAM_VERSION,
        },
        # Parameters
        "param_names": param_names,
        "param_values": params_np.tolist(),
    }
    if covariance is not None:
        add_covariance_to_summary(summary, covariance)
    return summary


def save_model_summary(summary: dict, path: str) -> None:
    """
    Save model summary to JSON file.
    
    Parameters
    ----------
    summary : dict
        Model summary dictionary from create_model_summary.
    path : str
        Path to save the JSON file.

    The file is written atomically (to a temporary file that then replaces
    `path`), so an interrupted run never leaves a truncated summary behind.
    Missing parent directories are created.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, 'w') as f:
        json.dump(summary, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_model_summary(path: str) -> dict:
    """
    Load model summary from JSON file.
    
    Parameters
    ----------
    path : str
        Path to the JSON file.
        
    Returns
    -------
    summary : dict
        Model summary dictionary.
    """
    with open(path, 'r') as f:
        return json.load(f)


def get_params_from_summary(
    summary: dict,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Extract parameter tensor from model summary.
    
    Parameters
    ----------
    summary : dict
        Model summary dictionary.
    device : torch.device, optional
        Device for the tensor. Default is CPU.
        
    Returns
    -------
    params : torch.Tensor
        Parameter tensor.
    """
    _check_param_version(summary)
    if device is None:
        device = torch.device('cpu')
    return torch.tensor(
        summary["param_values"],
        dtype=torch.float32,
        device=device,
    )


def _covariance_to_list(covariance) -> list:
    """Convert a covariance matrix (tensor or array) to a nested list of floats."""
    if isinstance(covariance, torch.Tensor):
        covariance = covariance.detach().cpu().double().numpy()
    return np.asarray(covariance, dtype=np.float64).tolist()


def add_covariance_to_summary(summary: dict, covariance) -> dict:
    """
    Store the joint covariance matrix of the parameters in a model summary.

    Parameters
    ----------
    summary : dict
        Model summary dictionary (modified in place).
    covariance : torch.Tensor or np.ndarray
        Covariance matrix of shape (n_params, n_params), ordered like
        summary["param_values"] (unconstrained scale).

    Returns
    -------
    summary : dict
        The same dictionary, with a "covariance" entry (nested lists).
    """
    n_params = len(summary["param_values"])
    cov_list = _covariance_to_list(covariance)
    shape = tuple(np.shape(cov_list))
    if shape != (n_params, n_params):
        raise ValueError(
            f"Covariance has shape {shape}, expected ({n_params}, {n_params})."
        )
    summary["covariance"] = cov_list
    return summary


def get_covariance_from_summary(
    summary: dict,
    device: Optional[torch.device] = None,
    dtype=torch.float32,
) -> Optional[torch.Tensor]:
    """
    Extract the saved joint covariance matrix of the parameters.

    Parameters
    ----------
    summary : dict
        Model summary dictionary.
    device : torch.device, optional
        Device for the tensor. Default is CPU.
    dtype : torch.dtype, optional
        Data type for the tensor. Default is torch.float32, matching
        get_params_from_summary.

    Returns
    -------
    covariance : torch.Tensor or None
        (n_params, n_params) covariance on the unconstrained parameter scale,
        or None if the summary does not contain one.
    """
    _check_param_version(summary)
    if summary.get("covariance") is None:
        return None
    if device is None:
        device = torch.device('cpu')
    return torch.tensor(summary["covariance"], dtype=dtype, device=device)


def create_model_from_summary(
    summary: dict,
    data_dict: dict,
    emulator_dict: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    parent_model: Optional[MixedModel] = None,
) -> MixedModel:
    """
    Reinstantiate a MixedModel from a saved summary.
    
    Parameters
    ----------
    summary : dict
        Model summary dictionary from load_model_summary.
    data_dict : dict
        Data dictionary (must be provided separately as data is not saved).
    emulator_dict : dict, optional
        Dictionary mapping emulator names to emulator objects.
        Required if the model used an emulator.
    device : torch.device, optional
        Device for the model. Default is CPU.
    parent_model : MixedModel, optional
        Parent model for sharing precomputed values.
        
    Returns
    -------
    model : MixedModel
        Reinstantiated model.
    """
    _check_param_version(summary)
    if device is None:
        device = torch.device('cpu')
    
    config = summary["config"]
    
    # Resolve emulator
    emulator = None
    emulator_name = config.get("emulator_name")
    if emulator_name is not None:
        if emulator_dict is None:
            raise ValueError(
                f"Model requires emulator '{emulator_name}' but emulator_dict not provided."
            )
        if emulator_name not in emulator_dict:
            raise ValueError(
                f"Emulator '{emulator_name}' not found in emulator_dict. "
                f"Available: {list(emulator_dict.keys())}"
            )
        emulator = emulator_dict[emulator_name]
    
    model = MixedModel(
        data_dict=data_dict,
        n_brand_forms=config["n_brand_forms"],
        X_dim=config["X_dim"],
        n_households=summary["n_households"],
        n_total_households=summary["n_total_households"],
        random_feature_indices=config["random_feature_indices"],
        n_factors_rand=config["n_factors_rand"],
        include_uniquenesses_rand=config["include_uniquenesses_rand"],
        n_factors_time=config["n_factors_time"],
        emulator=emulator,
        model_family=config["model_family"],
        n_draws=config["n_draws"],
        device=device,
        parent_model=parent_model,
        anchored_rand=config["anchored_rand"],
    )
    
    return model


def create_optimization_result_from_summary(
    summary: dict,
    device: Optional[torch.device] = None,
) -> OptimizationResult:
    """
    Reinstantiate an OptimizationResult from a saved summary.
    
    Parameters
    ----------
    summary : dict
        Model summary dictionary from load_model_summary.
    device : torch.device, optional
        Device for the parameter tensor. Default is CPU.
        
    Returns
    -------
    result : OptimizationResult
        Reinstantiated optimization result.
    """
    return OptimizationResult(
        params=get_params_from_summary(summary, device),
        loss=summary["loss"],
        converged=summary["converged"],
        grad_norm=summary["grad_norm"],
        n_iter=summary["n_iter"],
    )


def create_model_list(
    batch_data_dicts: list,
    n_factors_rand: int,
    model_kwargs: dict,
    n_factors_time: int = None,
    base_seed: int = 42,
    use_parent_model: bool = False,
) -> list:
    """
    Create a list of MixedModel instances for batched optimization.
    
    Parameters
    ----------
    batch_data_dicts : list
        List of data dictionaries, one per batch. Each dictionary should contain
        choice set sizes as keys with alternative_ids, X, choice_idx, and 
        household_id tensors, plus an "n_households" key.
    n_factors_rand : int
        Number of factors for household-level random effects (n_factors_rand).
    model_kwargs : dict
        Additional keyword arguments passed to MixedModel constructor.
        Should not include data_dict, n_households, seed, n_factors_rand,
        n_factors_time, or parent_model.
        Can include model_family ("logit" or "probit"), anchored_rand (bool),
        emulator, include_uniquenesses_rand, etc.
    n_factors_time : int, optional
        Number of factors for time-specific error covariance.
        If None, not passed to MixedModel (uses default).
    base_seed : int, optional
        Base random seed. Each batch model uses base_seed + batch_index.
        Default is 42.
    use_parent_model : bool, optional
        If True, the first model is set as parent_model for subsequent models
        to share precomputed indices. Default is False.
        
    Returns
    -------
    models : list
        List of MixedModel instances.
    """
    # Create a copy of model_kwargs to avoid modifying the original
    # and remove any stale parent_model from previous calls
    local_model_kwargs = {k: v for k, v in model_kwargs.items() if k != "parent_model"}
    
    models = []
    for i, batch_data_dict in enumerate(batch_data_dicts):
        batch_n_households = batch_data_dict["n_households"]
        
        # Build kwargs for this model
        kwargs = {
            "data_dict": batch_data_dict,
            "n_households": batch_n_households,
            "seed": base_seed + i,
            "n_factors_rand": n_factors_rand,
            **local_model_kwargs
        }
        
        # Add n_factors_time if specified
        if n_factors_time is not None:
            kwargs["n_factors_time"] = n_factors_time
        
        model = MixedModel(**kwargs)
        models.append(model)
        
        # Set first model as parent for subsequent models
        if use_parent_model and i == 0:
            local_model_kwargs["parent_model"] = model
    
    return models


def get_param_names_from_kwargs(
    model_kwargs: dict,
    n_factors_rand: int,
    n_factors_time: int = -1,
    intercept_names: list = None,
    numeric_names: list = None,
) -> list:
    """
    Parameter names for a model configuration, without building a MixedModel.

    Gives the same names as MixedModel(..., **model_kwargs).get_param_names(...)
    for the same n_factors_rand / n_factors_time.

    Parameters
    ----------
    model_kwargs : dict
        Keyword arguments as passed to MixedModel / fit_single_model. Uses
        n_brand_forms, X_dim, random_feature_indices, include_uniquenesses_rand,
        anchored_rand, and emulator (the time-specific block exists only when
        an emulator is given).
    n_factors_rand : int
        Number of random-effects factors.
    n_factors_time : int, optional
        Number of time-specific factors. Default is -1.
    intercept_names, numeric_names : list of str, optional
        As in MixedModel.get_param_names.

    Returns
    -------
    names : list of str
    """
    layout = ParamLayout(
        n_brand_forms=model_kwargs["n_brand_forms"],
        X_dim=model_kwargs["X_dim"],
        random_feature_indices=model_kwargs.get("random_feature_indices") or [],
        n_factors_rand=n_factors_rand,
        include_uniquenesses_rand=model_kwargs.get("include_uniquenesses_rand", False),
        anchored_rand=model_kwargs.get("anchored_rand", True),
        n_factors_time=n_factors_time,
        has_time_params=model_kwargs.get("emulator") is not None,
    )
    return layout.param_names(intercept_names, numeric_names)


def compute_household_loglikes_batched(
    batch_data_dicts: list,
    params: torch.Tensor,
    model_kwargs: dict,
    n_factors_rand: int,
    n_factors_time: int = -1,
    emulator=None,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Household log-likelihoods at `params`, evaluated batch by batch.

    Parameters
    ----------
    batch_data_dicts : list
        Output of split_data_by_household (each batch needs
        "original_household_ids" and "std_normal_draws").
    params : torch.Tensor
        Parameter vector for the model described by the remaining arguments.
    model_kwargs : dict
        MixedModel keyword arguments (n_brand_forms, X_dim, n_total_households,
        random_feature_indices, include_uniquenesses_rand, n_draws,
        model_family, anchored_rand, ...). Any "emulator" or "device" entries
        are replaced by the explicit arguments below.
    n_factors_rand, n_factors_time : int
        Model structure.
    emulator : optional
        Emulator to evaluate with (None for the exact logit/probit likelihood).
    device : torch.device, optional
        Device for the models. Default: model_kwargs["device"] or CPU.

    Returns
    -------
    loglikes : (n_households,) ndarray
        Log-likelihood of every household, in original household order.
    """
    kwargs = {k: v for k, v in model_kwargs.items() if k not in ("emulator", "device")}
    if device is None:
        device = model_kwargs.get("device", torch.device("cpu"))

    all_loglikes = []
    all_orig_ids = []
    for batch_dict in batch_data_dicts:
        model = MixedModel(
            data_dict=batch_dict,
            n_households=batch_dict["n_households"],
            n_factors_rand=n_factors_rand,
            n_factors_time=n_factors_time,
            emulator=emulator,
            device=device,
            **kwargs,
        )
        with torch.no_grad():
            batch_ll = model.loglike_household(params).detach().cpu().numpy()
        all_loglikes.append(batch_ll)
        all_orig_ids.extend(batch_dict["original_household_ids"])
        del model, batch_ll
        gc.collect()
        empty_cache(device)

    all_loglikes = np.concatenate(all_loglikes)
    loglikes_ordered = np.zeros(len(all_orig_ids))
    loglikes_ordered[np.asarray(all_orig_ids, dtype=np.int64)] = all_loglikes
    return loglikes_ordered


def _layout_for_params(
    params: torch.Tensor,
    has_time_params: bool,
    **layout_kwargs,
) -> ParamLayout:
    """
    Layout describing an existing parameter vector. If a time block is
    expected but the vector has none (e.g. the source model was fit without an
    emulator), fall back to the layout without it.
    """
    n = params.shape[0]
    layout = ParamLayout(has_time_params=has_time_params, **layout_kwargs)
    if layout.n_params == n:
        return layout
    if has_time_params:
        fallback = ParamLayout(has_time_params=False, **layout_kwargs)
        if fallback.n_params == n:
            return fallback
    raise ValueError(
        f"Source parameter vector has length {n}, but its configuration implies "
        f"{layout.n_params} parameters "
        f"(n_factors_rand={layout_kwargs.get('n_factors_rand')}, "
        f"n_factors_time={layout_kwargs.get('n_factors_time')})."
    )


def map_params_to_larger_model(
    source_params: torch.Tensor,
    source_n_factors_rand: int,
    target_n_factors_rand: int,
    source_n_factors_time: int,
    target_n_factors_time: int,
    n_brand_forms: int,
    n_fixed_features: int,
    random_feature_indices: list,
    include_uniquenesses_rand: bool,
    has_emulator: bool,
    device: torch.device,
    new_factor_log_diag: float = -2.,
    anchored_rand: bool = True,
) -> torch.Tensor:
    """
    Map parameters from a smaller model to a larger model for warm-starting.

    Parameters are matched by meaning (see ParamLayout keys), not by position:
    every target parameter that also exists in the source model is copied.
    New parameters are initialized so that the larger model starts at the same
    covariance matrices as the smaller one whenever the models are nested:

    - Under the anchored parameterization, adding factor r makes the uniqueness
      ψ_r (free in the smaller model, anchored at 0 in the larger one) the
      diagonal loading: log(L[r,r]) = 0.5 * log(ψ_r), so L[r,r]² = ψ_r. This
      also applies to r = 0: a model with 0 factors maps into a model with 1
      factor with L[0,0]² = ψ_0 and zero loadings, reproducing Σ_rand exactly.
      Time-specific factors are always anchored and use the same rule.
    - Off-diagonal elements of new factors are set to 0.
    - New diagonal elements with no uniqueness to inherit are set to
      new_factor_log_diag, except L_rand[0,0] under the anchored
      parameterization, which starts at log(1) = 0.
    - Random-coefficient means inherit the corresponding fixed coefficient when
      the source is a fixed-effects model, and vice versa.
    - Remaining new parameters (e.g. log-uniquenesses) start at 0.

    Parameters
    ----------
    source_params : torch.Tensor
        Parameters from the smaller model.
    source_n_factors_rand : int
        Number of random-effects factors in source model.
    target_n_factors_rand : int
        Number of random-effects factors in target model.
    source_n_factors_time : int
        Number of time factors in source model.
    target_n_factors_time : int
        Number of time factors in target model.
    n_brand_forms : int
        Number of brand-form alternatives.
    n_fixed_features : int
        Number of numeric features without random coefficients, i.e.
        X_dim - 1 - len(random_feature_indices).
    random_feature_indices : list
        Indices of features with random coefficients.
    include_uniquenesses_rand : bool
        Whether the model includes uniquenesses for random effects.
    has_emulator : bool
        Whether the model uses an emulator.
    device : torch.device
        Device for tensors.
    new_factor_log_diag : float
        Fallback initial value for diagonal elements of new factors on log scale.
        Used when anchored_rand=False or when source has no uniquenesses.
        Default is -2.
    anchored_rand : bool
        Whether the model uses anchored parameterization for random effects
        (first r_rand uniquenesses fixed to 0). Default is True.

    Returns
    -------
    target_params : torch.Tensor
        Parameters for the larger model, with shared parameters copied
        and new parameters initialized to preserve the nested structure.
    """
    random_feature_indices = list(random_feature_indices or [])
    common = dict(
        n_brand_forms=n_brand_forms,
        X_dim=n_fixed_features + len(random_feature_indices) + 1,
        random_feature_indices=random_feature_indices,
        include_uniquenesses_rand=include_uniquenesses_rand,
        anchored_rand=anchored_rand,
    )
    src = _layout_for_params(
        source_params, has_time_params=has_emulator,
        n_factors_rand=source_n_factors_rand, n_factors_time=source_n_factors_time, **common,
    )
    tgt = ParamLayout(
        n_factors_rand=target_n_factors_rand, n_factors_time=target_n_factors_time,
        has_time_params=has_emulator, **common,
    )
    src_values = src.named_values(source_params)

    target_values = []
    for key in tgt.keys:
        if key in src_values:
            target_values.append(src_values[key])
            continue

        kind = key[0]
        if kind in ("L_rand", "L_time"):
            _, i, j = key
            if i != j:
                target_values.append(0.0)
                continue
            psi_key = ("psi_rand" if kind == "L_rand" else "psi_time", j)
            inherits_uniqueness = anchored_rand if kind == "L_rand" else True
            if inherits_uniqueness and psi_key in src_values:
                # log(L[j,j]) = 0.5 * log(ψ_j), preserving L[j,j]² = ψ_j
                target_values.append(0.5 * src_values[psi_key])
            elif kind == "L_rand" and anchored_rand and j == 0:
                target_values.append(0.0)
            else:
                target_values.append(new_factor_log_diag)
        else:
            target_values.append(0.0)

    return torch.tensor(target_values, dtype=source_params.dtype, device=device)


def convert_to_dense_params(
    source_params: torch.Tensor,
    source_n_factors_rand: int,
    n_brand_forms: int,
    n_fixed_features: int,
    random_feature_indices: list,
    include_uniquenesses_rand: bool,
    device: torch.device,
    anchored_rand: bool = True,
) -> torch.Tensor:
    """
    Convert parameters from any model type to dense model parameters.
    
    Handles three cases:
    - source_n_factors_rand < 0: Fixed effects model (no random effects)
    - source_n_factors_rand = 0: Diagonal covariance (psi_rand only)
    - source_n_factors_rand > 0: Factor model (L_rand + optionally psi_rand)
    
    Dense model: Σ = L_dense @ L_dense' (full k×k lower triangular Cholesky,
    n_factors_rand = k, no uniquenesses). All diagonal elements of L_dense,
    including L_dense[0,0], are free parameters on the log scale, so the dense
    model is an unrestricted covariance matrix and reproduces the source Σ_rand.
    
    Only the random-effects part of the parameter vector is produced (means,
    fixed coefficients, L_dense); any time-specific parameters in source_params
    are ignored.
    
    Parameters
    ----------
    source_params : torch.Tensor
        Parameters from the source model.
    source_n_factors_rand : int
        Number of random-effects factors in source model.
    n_brand_forms : int
        Number of brand-form alternatives.
    n_fixed_features : int
        Number of numeric features without random coefficients, i.e.
        X_dim - 1 - len(random_feature_indices).
    random_feature_indices : list
        Indices of features with random coefficients.
    include_uniquenesses_rand : bool
        Whether the source model includes uniquenesses for random effects.
    device : torch.device
        Device for tensors.
    anchored_rand : bool
        Whether the source model uses anchored parameterization (first r_rand
        uniquenesses fixed to 0). Default is True.
        
    Returns
    -------
    target_params : torch.Tensor
        Parameters for the dense model.
    """
    random_feature_indices = list(random_feature_indices or [])
    common = dict(
        n_brand_forms=n_brand_forms,
        X_dim=n_fixed_features + len(random_feature_indices) + 1,
        random_feature_indices=random_feature_indices,
        anchored_rand=anchored_rand,
        n_factors_time=-1,
        has_time_params=False,
    )
    src = ParamLayout(
        n_factors_rand=source_n_factors_rand,
        include_uniquenesses_rand=include_uniquenesses_rand,
        **common,
    )
    if source_params.shape[0] < src.n_params:
        raise ValueError(
            f"Source parameter vector has length {source_params.shape[0]}, but its "
            f"random-effects block alone requires {src.n_params} parameters."
        )
    k = src.k
    tgt = ParamLayout(n_factors_rand=k, include_uniquenesses_rand=False, **common)

    # Random-effects block of the source (a time block, if any, follows it)
    src_params = source_params[:src.n_params]
    src_values = src.named_values(src_params)

    # Cholesky factor of the source random-effects covariance, as
    # {(i, j): value} with diagonal elements on the log scale.
    L_values = {}
    if src.r_rand < 0:
        # Fixed effects: initialize L_dense = I
        for i, j in tgt.L_rand_entries:
            L_values[(i, j)] = 0.0
    elif src.r_rand == 0:
        # Σ = diag(ψ), so L_dense = diag(sqrt(ψ)) and log(L_ii) = log(ψ_i) / 2
        for i, j in tgt.L_rand_entries:
            L_values[(i, j)] = 0.5 * src_values[("psi_rand", i)] if i == j else 0.0
    else:
        Sigma = src.Sigma_rand(src_params)
        # Add small jitter for numerical stability
        jitter = 1e-6
        Sigma = Sigma + jitter * torch.eye(k, dtype=Sigma.dtype, device=Sigma.device)
        L_dense = torch.linalg.cholesky(Sigma)
        log_diag = torch.log(torch.diagonal(L_dense, dim1=0, dim2=1)).tolist()
        L_dense_list = L_dense.tolist()
        for i, j in tgt.L_rand_entries:
            L_values[(i, j)] = log_diag[i] if i == j else L_dense_list[i][j]

    target_values = []
    for key in tgt.keys:
        if key[0] == "L_rand":
            target_values.append(L_values[(key[1], key[2])])
        else:
            target_values.append(src_values.get(key, 0.0))

    return torch.tensor(target_values, dtype=source_params.dtype, device=device)


def fit_single_model(
    n_factors: int,
    batch_data_dicts: list,
    start_param: torch.Tensor,
    start_param_n_factors_rand: int,
    model_kwargs: dict,
    n_households: int,
    device: torch.device,
    base_seed: int = 42,
    n_factors_time: int = -1,
    start_param_n_factors_time: int = -1,
    compute_covariance: bool = False,
    use_adam: bool = False,
    n_random_restarts: int = 0,
    restart_noise_std: float = 1.0,
    early_cutoff: int = None,
) -> dict:
    """
    Fit a single mixed model with the specified number of factors.
    
    Parameters
    ----------
    n_factors : int
        Number of factors for household-level random effects (n_factors_rand).
    batch_data_dicts : list
        List of data dictionaries, one per batch.
    start_param : torch.Tensor or None
        Starting parameter values. If None, starts from zeros. Otherwise the
        values are mapped to the target model with map_params_to_larger_model.
    start_param_n_factors_rand : int
        n_factors_rand of the model that produced start_param.
    model_kwargs : dict
        Additional keyword arguments passed to MixedModel constructor.
    n_households : int
        Total number of households across all batches.
    device : torch.device
        PyTorch device for tensors.
    base_seed : int, optional
        Base random seed for model creation. Default is 42.
    n_factors_time : int, optional
        Number of factors for time-specific error covariance.
        Use -1 for fixed identity covariance (standard logit/probit).
        Use 0 for diagonal covariance with free variances.
        Use >0 for factor-structured covariance. Default is -1.
    start_param_n_factors_time : int, optional
        n_factors_time of the model that produced start_param. Default is -1.
    compute_covariance : bool, optional
        Whether to compute OPG covariance matrix. Default is False.
    use_adam : bool
        Whether to use the Adam optimizer vs. LBFGS. Default = False.
    n_random_restarts : int, optional
        Number of random restarts. The model is fit n_random_restarts + 1 times
        total (once with original starting values, then n_random_restarts times
        with perturbed starting values). The best result is returned.
        Default is 0 (no restarts).
    restart_noise_std : float, optional
        Standard deviation of Gaussian noise added to starting values for
        random restarts. Default is 1.0.
    early_cutoff : int, optional
        If specified, function checks for improvement over previous starts.
        If the model has not improved, the model-fitting terminates early.
        Default is None (no early termination).
        
    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - params: Optimized parameter tensor
        - loss: Final loss value
        - loglike: Log-likelihood
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion
        - caic: Consistent AIC
        - n_params: Number of parameters
        - n_observations: Total number of observations
        - n_households: Total number of households
        - converged: Whether optimization converged
        - n_iter: Number of iterations
        - grad_norm: Final gradient norm
        - covariance: OPG covariance matrix (if compute_covariance=True)
        - model_config: Configuration dict for recreating the model
    """
    # Clean up extra memory
    empty_cache(device)
    gc.collect()
    
    # Create list of models
    models = create_model_list(
        batch_data_dicts=batch_data_dicts,
        n_factors_rand=n_factors,
        model_kwargs=model_kwargs,
        n_factors_time=n_factors_time,
        base_seed=base_seed,
    )
    
    n_params = models[0].n_params
    
    # Count total observations
    n_observations = sum(
        sum(
            model.data_dict[c]["X"].shape[0]
            for c in model.cs
        )
        for model in models
    )
    
    # Initialize start parameters with proper mapping
    if start_param is None:
        optim_start_param = torch.zeros(n_params, device=device)
    else:
        optim_start_param = map_params_to_larger_model(
            source_params=start_param,
            source_n_factors_rand=start_param_n_factors_rand,
            target_n_factors_rand=n_factors,
            source_n_factors_time=start_param_n_factors_time,
            target_n_factors_time=n_factors_time if n_factors_time is not None else -1,
            n_brand_forms=model_kwargs["n_brand_forms"],
            n_fixed_features=len([i for i in range(1, model_kwargs["X_dim"]) 
                                  if i - 1 not in model_kwargs.get("random_feature_indices", [])]),
            random_feature_indices=model_kwargs.get("random_feature_indices", []),
            include_uniquenesses_rand=model_kwargs.get("include_uniquenesses_rand", False),
            has_emulator=model_kwargs.get("emulator") is not None,
            device=device,
            anchored_rand=model_kwargs.get("anchored_rand", True),
        )

    # Set up optimizer
    if use_adam:
        adam_config = AdamOptimizationConfig()
        batched_optimizer = BatchedAdamOptimizer(adam_config, verbose=False)
    else:
        optimizer_config = OptimizationConfig()
        batched_optimizer = BatchedRobustLBFGSOptimizer(optimizer_config, verbose=False)
    
    # Fit with random restarts
    best_result = None
    best_loss = float('inf')
    
    for restart_idx in range(n_random_restarts + 1):
        # Perturb starting values for restarts (not for first fit)
        if restart_idx == 0:
            current_start = optim_start_param
        else:
            # Use first model to perturb (all models have same structure)
            current_start = models[0].perturb_starting_values(
                optim_start_param, 
                noise_std=restart_noise_std
            )
        
        result = batched_optimizer.optimize(current_start, models, best_loss, early_cutoff)
        
        if result.loss < best_loss:
            best_loss = result.loss
            best_result = result
    
    result = best_result
    
    # Extract values we need BEFORE cleanup
    params = result.params.clone()
    loss = float(result.loss)
    converged = bool(result.converged)
    n_iter = int(result.n_iter)
    grad_norm = float(result.grad_norm)
    
    # Calculate metrics (using n_households for BIC)
    loglike = -loss * n_households
    aic = 2.0 * n_params - 2.0 * loglike
    bic = np.log(n_households) * n_params - 2.0 * loglike
    caic = (np.log(n_households) + 1) * n_params - 2.0 * loglike
    
    # Extract model configuration from first model
    first_model = models[0]    
    model_config = {
        "n_brand_forms": first_model.n_brand_forms,
        "n_total_households": first_model.n_total_households,
        "X_dim": first_model.X_dim,
        "n_factors_rand": first_model.n_factors_rand,
        "n_factors_time": first_model.n_factors_time,
        "include_uniquenesses_rand": first_model.include_uniquenesses_rand,
        "random_feature_indices": first_model.random_feature_indices,
        "n_draws": first_model.n_draws,
        "emulator_name": None,  # Caller should set this if applicable
        "model_family": first_model.model_family,
        "anchored_rand": first_model.anchored_rand,
        "has_time_params": first_model.emulator is not None,
        "param_version": PARAM_VERSION,
    }
    
    # Optionally compute covariance
    covariance = None
    if compute_covariance:
        covariance = compute_opg_covariance(params, models)
    
    # Clean up models to free memory
    del models, result, batched_optimizer, optim_start_param, first_model
    gc.collect()
    empty_cache(device)
    
    result_dict = {
        "params": params,
        "loss": loss,
        "loglike": loglike,
        "aic": aic,
        "bic": bic,
        "caic": caic,
        "n_params": n_params,
        "n_observations": n_observations,
        "n_households": n_households,
        "converged": converged,
        "n_iter": n_iter,
        "grad_norm": grad_norm,
        "model_config": model_config,
    }
    
    if compute_covariance:
        result_dict["covariance"] = covariance
    
    return result_dict


def fit_single_model_with_emulator(
    n_factors_rand: int,
    n_factors_time: int,
    batch_data_dicts: list,
    start_param: torch.Tensor,
    start_param_n_factors_rand: int,
    start_param_n_factors_time: int,
    model_kwargs: dict,
    n_households: int,
    device: torch.device,
    base_seed: int = 42,
    compute_covariance: bool = False,
    use_adam: bool = False,
    n_random_restarts: int = 0,
    restart_noise_std: float = 1.0,
    early_cutoff: int = 200,
) -> dict:
    """
    Fit a single mixed model with the specified number of random-effects and time factors.
    
    Parameters
    ----------
    n_factors_rand : int
        Number of factors for household-level random effects.
    n_factors_time : int
        Number of factors for time-specific error covariance.
        Use -1 for fixed identity covariance (standard logit/probit).
        Use 0 for diagonal covariance with free variances.
        Use >0 for factor-structured covariance.
    batch_data_dicts : list
        List of data dictionaries, one per batch.
    start_param : torch.Tensor or None
        Starting parameter values from a previous model.
    start_param_n_factors_rand : int or None
        Number of random-effects factors in the start_param model.
        Required if start_param is not None.
    start_param_n_factors_time : int or None
        Number of time factors in the start_param model.
        Required if start_param is not None.
    model_kwargs : dict
        Additional keyword arguments passed to MixedModel constructor.
        Should include 'emulator' key with the emulator to use.
    n_households : int
        Total number of households across all batches.
    device : torch.device
        PyTorch device for tensors.
    base_seed : int, optional
        Base random seed for model creation. Default is 42.
    compute_covariance : bool, optional
        Whether to compute OPG covariance matrix. Default is False.
    use_adam : bool
        Whether to use the Adam optimizer vs. LBFGS. Default = False.
    n_random_restarts : int, optional
        Number of random restarts. The model is fit n_random_restarts + 1 times
        total (once with original starting values, then n_random_restarts times
        with perturbed starting values). The best result is returned.
        Default is 0 (no restarts).
    restart_noise_std : float, optional
        Standard deviation of Gaussian noise added to starting values for
        random restarts. Default is 1.0.
    early_cutoff : int, optional
        If specified, function checks for improvement over previous starts.
        If the model has not improved, the model-fitting terminates early.
        Default is 200.
        
    Returns
    -------
    result_dict : dict
        Dictionary containing:
        - params: Optimized parameter tensor
        - loss: Final loss value
        - loglike: Log-likelihood
        - aic: Akaike Information Criterion
        - bic: Bayesian Information Criterion
        - caic: Consistent AIC
        - n_params: Number of parameters
        - n_observations: Total number of observations
        - n_households: Total number of households
        - converged: Whether optimization converged
        - n_iter: Number of iterations
        - grad_norm: Final gradient norm
        - covariance: OPG covariance matrix (if compute_covariance=True)
        - model_config: Configuration dict for recreating the model
    """
    # Clean up extra memory
    empty_cache(device)
    gc.collect()
    
    # Create list of models
    models = create_model_list(
        batch_data_dicts=batch_data_dicts,
        n_factors_rand=n_factors_rand,
        model_kwargs=model_kwargs,
        n_factors_time=n_factors_time,
        base_seed=base_seed,
        use_parent_model=True,
    )

    n_params = models[0].n_params
    
    # Count total observations
    n_observations = sum(
        sum(
            model.data_dict[c]["X"].shape[0]
            for c in model.cs
        )
        for model in models
    )
    
    # Initialize start parameters with proper mapping
    if start_param is None:
        optim_start_param = torch.zeros(n_params, device=device)
    else:
        optim_start_param = map_params_to_larger_model(
            source_params=start_param,
            source_n_factors_rand=start_param_n_factors_rand,
            target_n_factors_rand=n_factors_rand,
            source_n_factors_time=start_param_n_factors_time,
            target_n_factors_time=n_factors_time,
            n_brand_forms=model_kwargs["n_brand_forms"],
            n_fixed_features=len([i for i in range(1, model_kwargs["X_dim"]) 
                                  if i - 1 not in model_kwargs.get("random_feature_indices", [])]),
            random_feature_indices=model_kwargs.get("random_feature_indices", []),
            include_uniquenesses_rand=model_kwargs.get("include_uniquenesses_rand", True),
            has_emulator=model_kwargs.get("emulator") is not None,
            device=device,
            anchored_rand=model_kwargs.get("anchored_rand", True),
        )
    
    if use_adam:
        adam_config = AdamOptimizationConfig()
        batched_optimizer = BatchedAdamOptimizer(adam_config, verbose=False)
    else:
        optimizer_config = OptimizationConfig()
        batched_optimizer = BatchedRobustLBFGSOptimizer(optimizer_config, verbose=False)
    
    # Fit with random restarts
    best_result = None
    best_loss = float('inf')
    
    for restart_idx in range(n_random_restarts + 1):
        if restart_idx == 0:
            current_start = optim_start_param
        else:
            current_start = models[0].perturb_starting_values(
                optim_start_param, 
                noise_std=restart_noise_std
            )
        
        result = batched_optimizer.optimize(current_start, models, best_loss, early_cutoff)
        
        if result.loss < best_loss:
            best_loss = result.loss
            best_result = result
    
    result = best_result
        
    # Extract values we need BEFORE cleanup
    params = result.params.clone()
    loss = float(result.loss)
    converged = bool(result.converged)
    n_iter = int(result.n_iter)
    grad_norm = float(result.grad_norm)
    
    # Calculate metrics
    loglike = -loss * n_households
    aic = 2.0 * n_params - 2.0 * loglike
    bic = np.log(n_households) * n_params - 2.0 * loglike
    caic = (np.log(n_households) + 1) * n_params - 2.0 * loglike
    
    # Extract model configuration from first model
    first_model = models[0]
    model_config = {
        "n_brand_forms": first_model.n_brand_forms,
        "n_total_households": first_model.n_total_households,
        "X_dim": first_model.X_dim,
        "n_factors_rand": first_model.n_factors_rand,
        "n_factors_time": first_model.n_factors_time,
        "include_uniquenesses_rand": first_model.include_uniquenesses_rand,
        "random_feature_indices": first_model.random_feature_indices,
        "n_draws": first_model.n_draws,
        "emulator_name": None,  # Caller should set this if applicable
        "model_family": first_model.model_family,
        "anchored_rand": first_model.anchored_rand,
        "has_time_params": first_model.emulator is not None,
        "param_version": PARAM_VERSION,
    }
    
    # Optionally compute covariance
    covariance = None
    if compute_covariance:
        covariance = compute_opg_covariance(params, models)
    
    # Clean up models to free memory
    del models, result, batched_optimizer, optim_start_param, first_model
    gc.collect()
    empty_cache(device)
    
    result_dict = {
        "params": params,
        "loss": loss,
        "loglike": loglike,
        "aic": aic,
        "bic": bic,
        "caic": caic,
        "n_params": n_params,
        "n_observations": n_observations,
        "n_households": n_households,
        "converged": converged,
        "n_iter": n_iter,
        "grad_norm": grad_norm,
        "model_config": model_config,
    }
    
    if compute_covariance:
        result_dict["covariance"] = covariance
    
    return result_dict


def result_dict_to_summary(result_dict: dict, model_name: str, param_names: list) -> dict:
    """
    Convert fit_single_model output to a saveable summary.

    If result_dict contains a "covariance" entry (fit with
    compute_covariance=True), the full joint covariance matrix is saved under
    summary["covariance"].
    
    Parameters
    ----------
    result_dict : dict
        Output from fit_single_model or fit_single_model_with_emulator.
    model_name : str
        Descriptive name for the model.
    param_names : list
        List of parameter names.
        
    Returns
    -------
    summary : dict
        JSON-serializable summary dictionary.
    """
    # Helper function to convert numpy types to native Python types
    def to_python_type(val):
        if isinstance(val, (np.integer, np.int64, np.int32)):
            return int(val)
        elif isinstance(val, (np.floating, np.float64, np.float32)):
            return float(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, dict):
            return {k: to_python_type(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [to_python_type(v) for v in val]
        else:
            return val
    
    # Convert model_config values to native Python types
    model_config = {k: to_python_type(v) for k, v in result_dict["model_config"].items()}
    
    summary = {
        "model_name": model_name,
        "n_observations": int(result_dict["n_observations"]),
        "n_households": int(result_dict["n_households"]),
        "n_params": int(result_dict["n_params"]),
        "log_likelihood": float(result_dict["loglike"]),
        "aic": float(result_dict["aic"]),
        "bic": float(result_dict["bic"]),
        "caic": float(result_dict["caic"]),
        "converged": bool(result_dict["converged"]),
        "n_iter": int(result_dict["n_iter"]),
        "grad_norm": float(result_dict["grad_norm"]),
        "loss": float(result_dict["loss"]),
        "config": model_config,
        "param_names": param_names,
        "param_values": result_dict["params"].detach().cpu().numpy().tolist(),
    }
    # Joint covariance of the parameters, if it was computed during fitting
    if result_dict.get("covariance") is not None:
        add_covariance_to_summary(summary, result_dict["covariance"])
    return summary


def _coefficient_table(params, covariance, param_names) -> pd.DataFrame:
    """Estimate / SE / z / p-value table."""
    ses = ses_from_cov(covariance)
    results_df = pd.DataFrame({
            "estimate": params.cpu().detach().numpy(),
            "SE": ses.cpu().detach().numpy()},
        index=param_names)
    results_df["z"] = results_df["estimate"] / results_df["SE"]
    results_df["Pr(|Z| > z)"] = 2 * norm.cdf(-results_df["z"].abs())
    return results_df


def _covariance_frame(covariance, param_names) -> pd.DataFrame:
    """Joint covariance as a DataFrame labelled by parameter names."""
    return pd.DataFrame(
        covariance.cpu().detach().numpy(), index=param_names, columns=param_names
    )


def create_coefficient_table(
    selected_idx: int,
    results_list: list,
    batch_data_dicts: list,
    model_kwargs: dict,
    n_factors_rand: int,
    intercept_names: list,
    numeric_names: list,
    n_factors_time: int = -1,
    base_seed: int = 42,
    recompute_covariance: bool = False,
    return_covariance: bool = False,
):
    """
    Create a coefficient table with estimates, standard errors, z-statistics, and p-values.
    
    Parameters
    ----------
    selected_idx : int
        Index into results_list for the model to summarize.
    results_list : list
        List of result dictionaries from fit_single_model.
    batch_data_dicts : list
        List of data dictionaries for creating models (used for covariance computation).
    model_kwargs : dict
        Model keyword arguments passed to MixedModel constructor.
    n_factors_rand : int
        Number of factors for household-level random effects.
    intercept_names : list of str
        Names for brand-form intercepts (excluding reference brand-form).
    numeric_names : list of str
        Names for numeric features.
    n_factors_time : int, optional
        Number of factors for time-specific error covariance. Default is -1
        (fixed identity covariance).
    base_seed : int, optional
        Base random seed for model creation. Default is 42.
    recompute_covariance : bool, optional
        If False (default) and the selected result already contains a
        "covariance" entry (fit with compute_covariance=True), that matrix is
        reused. Otherwise the OPG covariance is computed here.
    return_covariance : bool, optional
        If True, also return the full joint covariance matrix. Default is False.
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: estimate, SE, z, Pr(|Z| > z)
        Index contains parameter names.
    covariance_df : pd.DataFrame
        Only if return_covariance=True. Joint covariance of the parameters
        (unconstrained scale), with parameter names as index and columns.
    """
    selected_results = results_list[selected_idx]
    params = selected_results["params"]
    saved_cov = selected_results.get("covariance")
    use_saved = saved_cov is not None and not recompute_covariance

    # Create model list for this configuration (one batch suffices for names)
    selected_models = create_model_list(
        batch_data_dicts=batch_data_dicts[:1] if use_saved else batch_data_dicts,
        n_factors_rand=n_factors_rand,
        model_kwargs=model_kwargs,
        n_factors_time=n_factors_time,
        base_seed=base_seed,
        use_parent_model=True,
    )
    selected_param_names = selected_models[0].get_param_names(intercept_names, numeric_names)
    
    # Covariance and standard errors
    if use_saved:
        selected_cov = saved_cov.to(params.device) if isinstance(saved_cov, torch.Tensor) \
            else torch.tensor(np.asarray(saved_cov), dtype=params.dtype, device=params.device)
    else:
        selected_cov = compute_opg_covariance(params, selected_models)
    
    results_df = _coefficient_table(params, selected_cov, selected_param_names)
    covariance_df = (
        _covariance_frame(selected_cov, selected_param_names) if return_covariance else None
    )
    
    # Clean up models to free memory
    del selected_models, selected_cov
    gc.collect()
    empty_cache(model_kwargs.get("device", torch.device("cpu")))
    
    if return_covariance:
        return results_df, covariance_df
    return results_df


def create_coefficient_table_from_summary(
    summary_path: str,
    batch_data_dicts: list,
    model_kwargs: dict,
    intercept_names: list,
    numeric_names: list,
    base_seed: int = 42,
    recompute_covariance: bool = False,
    return_covariance: bool = False,
):
    """
    Create a coefficient table from a saved model summary JSON file.
    
    Parameters
    ----------
    summary_path : str
        Path to the saved model summary JSON file.
    batch_data_dicts : list
        List of data dictionaries for creating models (used for covariance computation).
    model_kwargs : dict
        Model keyword arguments passed to MixedModel constructor.
        Note: model_family and anchored_rand will be overridden by values
        from the saved summary to ensure consistency.
    intercept_names : list of str
        Names for brand-form intercepts (excluding reference brand-form).
    numeric_names : list of str
        Names for numeric features.
    base_seed : int, optional
        Base random seed for model creation. Default is 42.
    recompute_covariance : bool, optional
        If False (default) and the summary contains a saved "covariance", that
        matrix is used. Otherwise the OPG covariance is computed here.
    return_covariance : bool, optional
        If True, also return the full joint covariance matrix. Default is False.
        To store it in the summary file, pass it to add_covariance_to_summary
        and save with save_model_summary.
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: estimate, SE, z, Pr(|Z| > z)
        Index contains parameter names.
    covariance_df : pd.DataFrame
        Only if return_covariance=True. Joint covariance of the parameters
        (unconstrained scale), with parameter names as index and columns.
    """
    # Load summary from disk (raises for summaries with an older parameter layout)
    summary = load_model_summary(summary_path)
    _check_param_version(summary, source=f"Model summary '{summary_path}'")
    
    # Extract configuration
    config = summary["config"]
    n_factors_rand = config["n_factors_rand"]
    n_factors_time = config.get("n_factors_time", -1)
    
    # Extract model_family and anchored_rand from saved config
    model_family = config.get("model_family", "logit")
    anchored_rand = config.get("anchored_rand", True)
    
    # Get parameters
    device = model_kwargs.get("device", torch.device("cpu"))
    params = get_params_from_summary(summary, device)
    saved_cov = get_covariance_from_summary(summary, device, dtype=params.dtype)
    use_saved = saved_cov is not None and not recompute_covariance
    
    # Create a copy of model_kwargs and override with saved config values
    local_model_kwargs = dict(model_kwargs)
    local_model_kwargs["model_family"] = model_family
    local_model_kwargs["anchored_rand"] = anchored_rand
    
    # Create model list (one batch suffices when only names are needed)
    selected_models = create_model_list(
        batch_data_dicts=batch_data_dicts[:1] if use_saved else batch_data_dicts,
        n_factors_rand=n_factors_rand,
        model_kwargs=local_model_kwargs,  # Use local copy with overrides
        n_factors_time=n_factors_time,
        base_seed=base_seed,
        use_parent_model=True,
    )
    
    # Get parameter names
    param_names = selected_models[0].get_param_names(intercept_names, numeric_names)
    if len(param_names) != params.shape[0]:
        raise ValueError(
            f"Saved parameter vector has length {params.shape[0]}, but the model "
            f"built from the summary and model_kwargs has {len(param_names)} parameters."
        )
    
    # Covariance and standard errors
    selected_cov = saved_cov if use_saved else compute_opg_covariance(params, selected_models)
    
    results_df = _coefficient_table(params, selected_cov, param_names)
    covariance_df = _covariance_frame(selected_cov, param_names) if return_covariance else None
    
    # Clean up to free memory
    del selected_models, selected_cov, params
    gc.collect()
    empty_cache(device)
    
    if return_covariance:
        return results_df, covariance_df
    return results_df


def _n_lower_tri_params(n_rows: int, n_cols: int) -> int:
    """Number of free elements in the lower-triangular part of an (n_rows x n_cols) matrix."""
    return n_rows * n_cols - n_cols * (n_cols - 1) // 2


def _random_block_size(config: dict) -> int:
    """
    Number of parameters preceding the time-specific block in the flat parameter
    vector: intercepts (or random-effect means), fixed coefficients, L_rand, psi_rand.
    """
    return ParamLayout.from_config(config, has_time_params=False).n_params


def _time_block_sizes(config: dict) -> tuple:
    """(n_L_time_params, n_psi_time_params) as they would appear if an emulator was used."""
    layout = ParamLayout.from_config(config, has_time_params=True)
    return layout.n_L_time_params, layout.n_psi_time_params


def _matrix_function_se(matrix_fn, params: torch.Tensor, covariance: torch.Tensor) -> np.ndarray:
    """
    Delta-method standard errors for every element of matrix_fn(params):
    SE[a, b] = sqrt(g_ab' C g_ab), where g_ab is the gradient of element (a, b)
    with respect to the unconstrained parameters and C is their covariance.
    """
    jac = torch.func.jacrev(matrix_fn)(params)            # (rows, cols, n_params)
    rows, cols = jac.shape[0], jac.shape[1]
    jac_flat = jac.reshape(rows * cols, params.shape[0])
    var = (torch.matmul(jac_flat, covariance) * jac_flat).sum(dim=1)
    var = torch.clamp(var, min=0.0)
    return torch.sqrt(var).reshape(rows, cols).detach().cpu().numpy()


def _summary_params_and_cov(summary: dict, need_cov: bool, source: str):
    """Saved parameters (float64) and, if requested, the saved covariance (float64)."""
    params = torch.tensor(summary["param_values"], dtype=torch.float64)
    covariance = None
    if need_cov:
        covariance = get_covariance_from_summary(summary, dtype=torch.float64)
        if covariance is None:
            raise ValueError(
                f"{source} does not contain a saved covariance matrix; fit with "
                "compute_covariance=True or add one with add_covariance_to_summary."
            )
    return params, covariance


def load_time_covariance(summary_path: str, return_se: bool = False):
    """
    Reconstruct the full time-specific error covariance matrix from a model summary.

    Parameters
    ----------
    summary_path : str
        Path to a JSON file written by `save_model_summary`.
    return_se : bool, optional
        If True, also return delta-method standard errors for every element,
        computed from the saved joint covariance matrix. Default is False.

    Returns
    -------
    Sigma_time : (n_brand_forms, n_brand_forms) ndarray
        Covariance matrix of the time-specific errors, with the reference
        brand-form in position 0. Following the estimation-time convention, the
        reference alternative has unit variance and zero covariance with the
        remaining alternatives; the lower-right block is L L' + diag(psi).

        If the model carries no free time-specific parameters (no emulator, or
        n_factors_time < 0) the identity matrix is returned.
    Sigma_time_se : (n_brand_forms, n_brand_forms) ndarray
        Only if return_se=True. Entries that are fixed by normalization have
        standard error 0.

    Raises
    ------
    ValueError
        If the summary was written with an older parameter layout, if the saved
        parameter vector is not consistent with the saved configuration, or if
        return_se=True and no covariance matrix was saved.
    """
    summary = load_model_summary(summary_path)
    source = f"Model summary '{summary_path}'"
    _check_param_version(summary, source=source)
    layout = ParamLayout.from_summary(summary)
    config = summary["config"]

    n_time = layout.n_L_time_params + layout.n_psi_time_params
    if n_time == 0 and config.get("model_family", "logit") == "logit" and not layout.has_time_params:
        warnings.warn(
            "This model has no time-specific covariance parameters. If it was fit "
            "as a plain logit, the errors are Gumbel rather than Gaussian and an "
            "identity covariance matrix is only a formal placeholder."
        )

    params, covariance = _summary_params_and_cov(summary, return_se, source)
    Sigma_time = layout.Sigma_time(params).detach().cpu().numpy()
    if not return_se:
        return Sigma_time
    Sigma_time_se = _matrix_function_se(layout.Sigma_time, params, covariance)
    return Sigma_time, Sigma_time_se


def load_rand_covariance(summary_path: str, return_se: bool = False):
    """
    Reconstruct the random-effects covariance matrix Σ_rand from a model summary.

    Parameters
    ----------
    summary_path : str
        Path to a JSON file written by `save_model_summary`.
    return_se : bool, optional
        If True, also return delta-method standard errors for every element,
        computed from the saved joint covariance matrix. Default is False.

    Returns
    -------
    Sigma_rand : (k, k) ndarray or None
        Covariance of the household-level random effects, ordered as the
        n_brand_forms - 1 intercepts (brand-forms 1, 2, ...) followed by the
        random feature coefficients in the order of random_feature_indices.
        L L' + diag(psi) for factor models, diag(psi) for n_factors_rand = 0.
        None for a fixed-effects model (n_factors_rand < 0).
    Sigma_rand_se : (k, k) ndarray or None
        Only if return_se=True.

    Raises
    ------
    ValueError
        If the summary was written with an older parameter layout, if the saved
        parameter vector is not consistent with the saved configuration, or if
        return_se=True and no covariance matrix was saved.
    """
    summary = load_model_summary(summary_path)
    source = f"Model summary '{summary_path}'"
    _check_param_version(summary, source=source)
    layout = ParamLayout.from_summary(summary)

    if layout.r_rand < 0:
        return (None, None) if return_se else None

    params, covariance = _summary_params_and_cov(summary, return_se, source)
    Sigma_rand = layout.Sigma_rand(params).detach().cpu().numpy()
    if not return_se:
        return Sigma_rand
    Sigma_rand_se = _matrix_function_se(layout.Sigma_rand, params, covariance)
    return Sigma_rand, Sigma_rand_se