# emulator.py
"""
Neural network emulator for discrete choice probabilities with correlated errors.

The same architecture serves the multinomial probit model (include_c_star=False)
and the correlated-Gumbel model (include_c_star=True, which adds index features
because that model is not factorization invariant).

Architecture:
1. Per-alternative DS-CI encoder produces a representation for each alternative
   based on its relationship with all other alternatives
2. Equivariant layers for final processing
3. Output logits
"""

import torch
import torch.nn as nn


# =============================================================================
# Basic Building Blocks
# =============================================================================


def smooth_softsign(x, k=0.1):
    """
    Smooth, C^∞ bijective mapping from R to (-1, 1).
    
    Properties:
        - Bijective mapping from R to (-1, 1)
        - C^∞ (infinitely differentiable)
        - Near origin: f(x) ≈ kx
        - Large |x|: f(x) → sign(x)
    
    Formula: f(x) = kx / sqrt(1 + (kx)^2)
    
    Args:
        x: Input tensor
        k: Scaling parameter controlling transition sharpness (default: 0.1)
    """
    kx = k * x
    return kx / torch.sqrt(1 + kx ** 2)


class MLP(nn.Module):
    """
    Multi-layer perceptron with SiLU (Swish) activations.
    
    If num_layers=-1, acts as identity (input_dim must equal output_dim).
    If num_layers=0, acts as a single linear projection (input_dim -> output_dim).
    If num_layers>0, has that many hidden layers plus input and output projections.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers (-1 for identity/passthrough)
        dropout_rate: Dropout rate
        include_k: Whether to include K as an input feature
        include_c_star: Whether to include index features (required for the
            correlated-Gumbel model; False for multinomial probit)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout_rate=0.01):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if num_layers == -1:
            assert input_dim == output_dim, \
                f"For num_layers=-1 (identity), input_dim ({input_dim}) must equal output_dim ({output_dim})"
            self.network = nn.Identity()
        elif num_layers == 0:
            self.network = nn.Linear(input_dim, output_dim)
        else:
            layers = []
            
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.SiLU())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.Linear(hidden_dim, output_dim))
            
            self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def compute_effective_output_dim(input_dim, output_dim, num_layers):
    """Compute the effective output dimension of an MLP given its configuration."""
    if num_layers == -1:
        return input_dim
    return output_dim


# =============================================================================
# Equivariant Layers
# =============================================================================

class EquivariantLayer(nn.Module):
    """
    Permutation-equivariant linear layer.
    
    L(X) = X @ A + (1/K) * 1 * 1^T * X @ B + 1 * c^T
    
    When use_bias=False:
    L(X) = X @ A
    
    Args:
        d_in: Input dimension
        d_out: Output dimension
        use_bias: Whether to include the mean-pooling term B and bias c (default: True)
    """
    def __init__(self, d_in, d_out, use_bias=True):
        super(EquivariantLayer, self).__init__()
        
        self.use_bias = use_bias
        
        self.A = nn.Linear(d_in, d_out, bias=False)
        
        if use_bias:
            self.B = nn.Linear(d_in, d_out, bias=False)
            self.c = nn.Parameter(torch.zeros(d_out))
            nn.init.xavier_uniform_(self.B.weight)
        else:
            self.register_parameter('B', None)
            self.register_parameter('c', None)
        
        nn.init.xavier_uniform_(self.A.weight)
    
    def forward(self, X):
        """
        Args:
            X: Input of shape (batch_size, k, d_in)
        Returns:
            Output of shape (batch_size, k, d_out)
        """
        result = self.A(X)
        
        if self.use_bias:
            batch_size, k, _ = X.shape
            X_mean = X.mean(dim=1, keepdim=True)
            term2 = self.B(X_mean).expand(-1, k, -1)
            result = result + term2 + self.c
        
        return result


class EquivariantBlock(nn.Module):
    """
    Equivariant block with equivariant layer, activation, and dropout.
    """
    def __init__(self, d_in, d_out, dropout_rate=0.01, use_residual=True):
        super(EquivariantBlock, self).__init__()
        
        self.use_residual = use_residual and (d_in == d_out)
        
        self.equiv_layer = EquivariantLayer(d_in, d_out)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, X):
        out = self.equiv_layer(X)
        out = self.activation(out)
        out = self.dropout(out)
        
        if self.use_residual:
            out = out + X
        
        return out


# =============================================================================
# Per-Alternative DS-CI Encoder
# =============================================================================

class PerAlternativeDSCIEncoder(nn.Module):
    """
    Per-alternative DS-CI encoder that produces a representation for each alternative
    based on its relationship with all other alternatives.
    
    For factorization-invariant models (MNP): include_c_star=False
    For non-factorization-invariant models (Gumbel): include_c_star=True
    
    When include_c_star=True, index features are added to break permutation equivariance:
    - Focal alternative is_first indicator is passed to combining MLP
    - Rival alternative is_first indicator is passed to diagonal DeepSet
    """
    def __init__(
        self,
        diag_phi_hidden_dim=32,
        diag_phi_output_dim=32,
        diag_phi_num_layers=2,
        diag_rho_hidden_dim=32,
        diag_rho_output_dim=32,
        diag_rho_num_layers=2,
        offdiag_phi_hidden_dim=32,
        offdiag_phi_output_dim=32,
        offdiag_phi_num_layers=2,
        offdiag_rho_hidden_dim=32,
        offdiag_rho_output_dim=32,
        offdiag_rho_num_layers=2,
        combine_hidden_dim=64,
        combine_output_dim=64,
        combine_num_layers=4,
        dropout_rate=0.01,
        include_k=False,
        include_c_star=False,
    ):
        super(PerAlternativeDSCIEncoder, self).__init__()
        
        self.combine_output_dim = combine_output_dim
        self.include_k = include_k
        self.include_c_star = include_c_star
        
        diag_phi_input_dim = 9
        if include_c_star:
            diag_phi_input_dim += 1  # rival is_first indicator
        
        offdiag_phi_input_dim = 6
        alt_features_dim = 8
        if include_k:
            alt_features_dim += 1
        if include_c_star:
            alt_features_dim += 2  # c_star and focal is_first indicator
        
        self.diag_phi = MLP(
            input_dim=diag_phi_input_dim,
            hidden_dim=diag_phi_hidden_dim,
            output_dim=diag_phi_output_dim,
            num_layers=diag_phi_num_layers,
            dropout_rate=dropout_rate
        )
        self.diag_phi_output_dim = compute_effective_output_dim(
            diag_phi_input_dim, diag_phi_output_dim, diag_phi_num_layers
        )
        
        diag_rho_input_dim = self.diag_phi_output_dim
        
        self.diag_rho = MLP(
            input_dim=diag_rho_input_dim,
            hidden_dim=diag_rho_hidden_dim,
            output_dim=diag_rho_output_dim,
            num_layers=diag_rho_num_layers,
            dropout_rate=dropout_rate
        )
        self.diag_rho_output_dim = compute_effective_output_dim(
            diag_rho_input_dim, diag_rho_output_dim, diag_rho_num_layers
        )
        
        self.offdiag_phi = MLP(
            input_dim=offdiag_phi_input_dim,
            hidden_dim=offdiag_phi_hidden_dim,
            output_dim=offdiag_phi_output_dim,
            num_layers=offdiag_phi_num_layers,
            dropout_rate=dropout_rate
        )
        self.offdiag_phi_output_dim = compute_effective_output_dim(
            offdiag_phi_input_dim, offdiag_phi_output_dim, offdiag_phi_num_layers
        )
        
        offdiag_rho_input_dim = self.offdiag_phi_output_dim
        
        self.offdiag_rho = MLP(
            input_dim=offdiag_rho_input_dim,
            hidden_dim=offdiag_rho_hidden_dim,
            output_dim=offdiag_rho_output_dim,
            num_layers=offdiag_rho_num_layers,
            dropout_rate=dropout_rate
        )
        self.offdiag_rho_output_dim = compute_effective_output_dim(
            offdiag_rho_input_dim, offdiag_rho_output_dim, offdiag_rho_num_layers
        )
        
        combine_input_dim = self.diag_rho_output_dim + self.offdiag_rho_output_dim + alt_features_dim
        self.combine_mlp = MLP(
            input_dim=combine_input_dim,
            hidden_dim=combine_hidden_dim,
            output_dim=combine_output_dim,
            num_layers=combine_num_layers,
            dropout_rate=dropout_rate
        )
        
        self._j_indices_cache = {}
        self._tril_cache = {}
        self._involves_i_cache = {}
    
    def _get_j_indices(self, k, device):
        """Get indices for all j != i, vectorized construction."""
        key = (k, device)
        if key not in self._j_indices_cache:
            all_indices = torch.arange(k, device=device).unsqueeze(0).expand(k, k)
            mask = ~torch.eye(k, dtype=torch.bool, device=device)
            j_indices = all_indices[mask].view(k, k - 1)
            i_indices = torch.arange(k, device=device).unsqueeze(1).expand(k, k - 1)
            self._j_indices_cache[key] = (i_indices, j_indices)
        return self._j_indices_cache[key]
    
    def _get_tril_indices(self, k, device):
        """Get lower triangular indices."""
        key = (k, device)
        if key not in self._tril_cache:
            self._tril_cache[key] = torch.tril_indices(k, k, offset=-1, device=device)
        return self._tril_cache[key]
    
    def _get_involves_i_mask(self, k, device):
        """Get mask for pairs involving each alternative i, pre-squeezed."""
        key = (k, device)
        if key not in self._involves_i_cache:
            tril_idx = self._get_tril_indices(k, device)
            i_range = torch.arange(k, device=device).unsqueeze(1)
            involves_i = (tril_idx[0].unsqueeze(0) == i_range) | (tril_idx[1].unsqueeze(0) == i_range)
            self._involves_i_cache[key] = involves_i.float()
        return self._involves_i_cache[key]
    
    def forward(self, v_star, cov_star, c_star=None):
        batch_size, k = v_star.shape
        device = v_star.device
        if self.include_k:
            k_mat = torch.full_like(v_star, k)
        if self.include_c_star:
            if c_star is None:
                raise ValueError("Must pass c_star if include_c_star=True")
            # c_star has shape (batch_size,), need to broadcast to (batch_size, k)
            # Also, compute log to handle outliers
            c_star_transformed = smooth_softsign(torch.log(c_star + 1e-8))
            c_star_mat = c_star_transformed.unsqueeze(1).expand(-1, k)
            # Focal alternative is_first indicator: shape (batch_size, k)
            is_first_alt = torch.zeros(batch_size, k, device=device, dtype=v_star.dtype)
            is_first_alt[:, 0] = 1.0
        
        var_diag = torch.diagonal(cov_star, dim1=1, dim2=2)
        std_diag = torch.sqrt(torch.abs(var_diag) + 1e-8)
        snr = smooth_softsign(v_star / std_diag)
        
        std_outer = std_diag.unsqueeze(2) * std_diag.unsqueeze(1)
        corr_matrix = (cov_star / (std_outer + 1e-8)).clamp(-1, 1)
        
        v_diff_matrix = v_star.unsqueeze(2) - v_star.unsqueeze(1)
        var_sum_matrix = var_diag.unsqueeze(2) + var_diag.unsqueeze(1)
        var_diff_matrix = var_sum_matrix - 2 * cov_star
        std_diff_matrix = torch.sqrt(torch.abs(var_diff_matrix) + 1e-8)
        z_matrix = smooth_softsign(v_diff_matrix / std_diff_matrix)
        
        cov_row_sums = cov_star.sum(dim=2) - var_diag
        corr_row_sums = corr_matrix.sum(dim=2) - 1.0
        cov_sq_row_sums = (cov_star ** 2).sum(dim=2) - var_diag ** 2
        corr_sq_row_sums = (corr_matrix ** 2).sum(dim=2) - 1.0
        
        denom = k - 1
        mean_cov_i = cov_row_sums / denom
        mean_corr_i = corr_row_sums / denom
        mean_sq_cov_i = cov_sq_row_sums / denom
        mean_sq_corr_i = corr_sq_row_sums / denom
        
        alt_features = torch.stack([v_star, var_diag, std_diag, mean_corr_i, mean_sq_corr_i, mean_cov_i, mean_sq_cov_i, snr], dim=-1)
        if self.include_k:
            k_mat_expanded = k_mat.unsqueeze(-1)  # (batch, K) -> (batch, K, 1)
            alt_features = torch.cat([k_mat_expanded, alt_features], dim=-1)
        if self.include_c_star:
            c_star_mat_expanded = c_star_mat.unsqueeze(-1)  # (batch, K) -> (batch, K, 1)
            is_first_alt_expanded = is_first_alt.unsqueeze(-1)  # (batch, K) -> (batch, K, 1)
            alt_features = torch.cat([c_star_mat_expanded, is_first_alt_expanded, alt_features], dim=-1)
        
        z_diag = self._process_diagonal(
            v_star, var_diag, std_diag, cov_star, corr_matrix,
            z_matrix, batch_size, k, device
        )
        
        z_offdiag = self._process_offdiag(
            cov_star, corr_matrix, v_diff_matrix, z_matrix, 
            var_sum_matrix, std_diag, batch_size, k, device
        )
        
        combined = torch.cat([z_diag, z_offdiag, alt_features], dim=-1)
        return self.combine_mlp(combined)

    def _process_diagonal(self, v_star, var_diag, std_diag, cov_star, corr_matrix,
                          z_matrix, batch_size, k, device):
        """Process diagonal (pairwise i,j) relationships."""
        i_idx, j_idx = self._get_j_indices(k, device)
        
        v_j = v_star[:, j_idx]
        var_jj = var_diag[:, j_idx]
        std_jj = std_diag[:, j_idx]
        cov_ij = cov_star[:, i_idx, j_idx]
        corr_ij = corr_matrix[:, i_idx, j_idx]
        z_ij = z_matrix[:, i_idx, j_idx]
        
        v_i = v_star.unsqueeze(2)
        var_ii = var_diag.unsqueeze(2)
        std_ii = std_diag.unsqueeze(2)
        
        features = [
            v_j, var_jj, std_jj, cov_ij, corr_ij,
            v_i.expand_as(v_j), var_ii.expand_as(v_j), std_ii.expand_as(v_j), z_ij
        ]
        
        if self.include_c_star:
            # Rival is_first indicator: shape (k, k-1) -> (batch_size, k, k-1)
            j_is_first = (j_idx == 0).float()
            j_is_first = j_is_first.unsqueeze(0).expand(batch_size, -1, -1)
            features.append(j_is_first)
        
        features = torch.stack(features, dim=-1)
        
        diag_phi_input_dim = 10 if self.include_c_star else 9
        phi_out = self.diag_phi(features.view(-1, diag_phi_input_dim))
        phi_out = phi_out.view(batch_size, k, k - 1, -1)
        
        aggregated = phi_out.mean(dim=2)
        
        return self.diag_rho(aggregated)
        
    def _process_offdiag(self, cov_star, corr_matrix, v_diff_matrix, z_matrix,
                         var_sum_matrix, std_diag, batch_size, k, device):
        """Process off-diagonal (pair of pairs) relationships."""
        tril_idx = self._get_tril_indices(k, device)
        
        cov_jl = cov_star[:, tril_idx[0], tril_idx[1]]
        corr_jl = corr_matrix[:, tril_idx[0], tril_idx[1]]
        v_diff_sq_jl = v_diff_matrix[:, tril_idx[0], tril_idx[1]] ** 2
        z_sq_jl = z_matrix[:, tril_idx[0], tril_idx[1]] ** 2
        var_sum_jl = var_sum_matrix[:, tril_idx[0], tril_idx[1]]
        
        std_sum_jl = std_diag[:, tril_idx[0]] + std_diag[:, tril_idx[1]]
        
        features = torch.stack([
            cov_jl, corr_jl, v_diff_sq_jl, z_sq_jl, var_sum_jl, std_sum_jl
        ], dim=-1)
        
        phi_out = self.offdiag_phi(features.view(-1, 6))
        phi_out = phi_out.view(batch_size, -1, self.offdiag_phi_output_dim)
        
        S_full = phi_out.sum(dim=1)
        
        involves_i_mask = self._get_involves_i_mask(k, device)
        sum_involving_i = torch.einsum('bpd,kp->bkd', phi_out, involves_i_mask)
        sum_excluding_i = S_full.unsqueeze(1) - sum_involving_i
        
        num_pairs_excluding_i = (k - 1) * (k - 2) // 2
        if num_pairs_excluding_i > 0:
            aggregated = sum_excluding_i / num_pairs_excluding_i
        else:
            aggregated = sum_excluding_i
        
        return self.offdiag_rho(aggregated)

    
# =============================================================================
# Main Emulator Model
# =============================================================================


class Emulator(nn.Module):
    """
    Neural network emulator for correlated discrete choice models using per-alternative
    DS-CI encoding: multinomial probit (include_c_star=False) or correlated Gumbel
    (include_c_star=True).
    
    Architecture:
    1. Per-alternative DS-CI encoder produces a representation for each alternative
    2. Equivariant layers for final processing
    3. Output logits via final equivariant layer
    
    Args:
        name: Model name
        
        # Per-alternative DS-CI Encoder - Diagonal DeepSet
        diag_phi_hidden_dim, diag_phi_output_dim, diag_phi_num_layers
        diag_rho_hidden_dim, diag_rho_output_dim, diag_rho_num_layers
        
        # Per-alternative DS-CI Encoder - Off-diagonal DeepSet
        offdiag_phi_hidden_dim, offdiag_phi_output_dim, offdiag_phi_num_layers
        offdiag_rho_hidden_dim, offdiag_rho_output_dim, offdiag_rho_num_layers
        
        # Per-alternative DS-CI Encoder - Combining MLP
        combine_hidden_dim, combine_output_dim, combine_num_layers
        
        # Equivariant layers
        equiv_hidden_dim: Hidden dimension for equivariant layers
        num_equiv_layers: Number of equivariant blocks (can be 0)
        
        dropout_rate: Dropout rate
    """
    def __init__(
        self,
        name="emulator",
        # Diagonal DeepSet
        diag_phi_hidden_dim=32,
        diag_phi_output_dim=32,
        diag_phi_num_layers=2,
        diag_rho_hidden_dim=32,
        diag_rho_output_dim=32,
        diag_rho_num_layers=2,
        # Off-diagonal DeepSet
        offdiag_phi_hidden_dim=32,
        offdiag_phi_output_dim=32,
        offdiag_phi_num_layers=2,
        offdiag_rho_hidden_dim=32,
        offdiag_rho_output_dim=32,
        offdiag_rho_num_layers=2,
        # Combining MLP
        combine_hidden_dim=64,
        combine_output_dim=64,
        combine_num_layers=4,
        # Equivariant layers
        equiv_hidden_dim=64,
        num_equiv_layers=2,
        # Shared
        dropout_rate=0.3,
        include_k=False,
        include_c_star=False,
    ):
        super(Emulator, self).__init__()        
        self.name = name
        self.dropout_rate = dropout_rate
        self.num_equiv_layers = num_equiv_layers
        self.t = 0
        self.include_k = include_k
        self.include_c_star = include_c_star
        
        self.per_alt_encoder = PerAlternativeDSCIEncoder(
            diag_phi_hidden_dim=diag_phi_hidden_dim,
            diag_phi_output_dim=diag_phi_output_dim,
            diag_phi_num_layers=diag_phi_num_layers,
            diag_rho_hidden_dim=diag_rho_hidden_dim,
            diag_rho_output_dim=diag_rho_output_dim,
            diag_rho_num_layers=diag_rho_num_layers,
            offdiag_phi_hidden_dim=offdiag_phi_hidden_dim,
            offdiag_phi_output_dim=offdiag_phi_output_dim,
            offdiag_phi_num_layers=offdiag_phi_num_layers,
            offdiag_rho_hidden_dim=offdiag_rho_hidden_dim,
            offdiag_rho_output_dim=offdiag_rho_output_dim,
            offdiag_rho_num_layers=offdiag_rho_num_layers,
            combine_hidden_dim=combine_hidden_dim,
            combine_output_dim=combine_output_dim,
            combine_num_layers=combine_num_layers,
            dropout_rate=dropout_rate,
            include_k=include_k,
            include_c_star=include_c_star,
        )
        
        self.feature_dim = self.per_alt_encoder.combine_output_dim
        
        if num_equiv_layers > 0:
            equiv_layers = []
            equiv_layers.append(EquivariantBlock(
                self.feature_dim, equiv_hidden_dim, dropout_rate, use_residual=False
            ))
            for _ in range(num_equiv_layers - 1):
                equiv_layers.append(EquivariantBlock(
                    equiv_hidden_dim, equiv_hidden_dim, dropout_rate, use_residual=False
                ))
            self.equiv_layers = nn.ModuleList(equiv_layers)
            self.output_proj = EquivariantLayer(equiv_hidden_dim, 1, use_bias=False)
        else:
            self.equiv_layers = nn.ModuleList([])
            self.output_proj = EquivariantLayer(self.feature_dim, 1, use_bias=False)
    
    def step(self):
        """Decay dropout rate over training."""
        self.t += 1
        new_rate = self.dropout_rate / (1 + self.t)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = new_rate
    
    def forward(self, v_star, cov_star, c_star=None):
        """
        Args:
            v_star: Centered/scaled utilities of shape (batch_size, k)
            cov_star: Centered/scaled covariance of shape (batch_size, k, k)
        
        Returns:
            logits: Choice logits of shape (batch_size, k)
        """
        if not self.include_c_star:
            c_star = None
        z = self.per_alt_encoder(v_star, cov_star, c_star)
        
        X = z
        for layer in self.equiv_layers:
            X = layer(X)
        
        logits = self.output_proj(X).squeeze(-1)
        
        return logits