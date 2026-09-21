import torch


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu")
    return device


def sort_covariance_matrices(cov_matrices, indices):
    batch_size, n, _ = cov_matrices.shape
    
    # Expand indices for gathering
    indices_expanded = indices.unsqueeze(2).expand(-1, -1, n)
    
    # Sort rows
    cov_sorted_rows = torch.gather(cov_matrices, 1, indices_expanded)
    
    # Sort columns
    indices_expanded_cols = indices.unsqueeze(1).expand(-1, n, -1)
    cov_sorted = torch.gather(cov_sorted_rows, 2, indices_expanded_cols)
    
    return cov_sorted


def compute_d_sigma_dt(cov: torch.Tensor) -> torch.Tensor:
    """
    Efficiently compute D @ Sigma @ D.T where D = [-1_{K-1} | I_{K-1}]
    
    Args:
        cov: tensor of shape (..., K, K) - stack of covariance matrices
        
    Returns:
        tensor of shape (..., K-1, K-1)
    """
    # Extract components using slicing (no explicit D matrix needed)
    
    # Lower-right (K-1) x (K-1) block: Sigma[1:, 1:]
    block = cov[..., 1:, 1:]
    
    # First row (excluding corner): Sigma[0, 1:], needs broadcasting for rows
    first_row = cov[..., 0:1, 1:]  # shape (..., 1, K-1)
    
    # First column (excluding corner): Sigma[1:, 0], needs broadcasting for cols
    first_col = cov[..., 1:, 0:1]  # shape (..., K-1, 1)
    
    # Top-left corner: Sigma[0, 0]
    corner = cov[..., 0:1, 0:1]  # shape (..., 1, 1)
    
    # Apply the formula: Σ[1:,1:] - Σ[0,1:] - Σ[1:,0] + Σ[0,0]
    result = block - first_row - first_col + corner
    
    return result


def omega_to_cov(omega):
    batch_size, km1, _ = omega.shape
    k = km1 + 1
    cov = torch.zeros(batch_size, k, k, device=omega.device, dtype=omega.dtype)
    cov[:, 0, 0] = 1.
    cov[:, 1:, 1:] = omega
    return cov


def center_v(v):
    return v - v.mean(dim=1, keepdim=True)    


def canonicalize(v, cov, eps=1e-8):
    """
    Canonicalization using centering and scale normalization.
    
    Args:
        v: Utility vector of shape (batch_size, k)
        cov: Covariance matrix of shape (batch_size, k, k)
        eps: Small constant for numerical stability
    
    Returns:
        v_star: Centered and scaled utilities of shape (batch_size, k)
        cov_star: Centered and scaled covariance of shape (batch_size, k, k)
    """
    # Centering projection M = I - (1/K) * 1 * 1^T
    # M @ v = v - mean(v)
    v_centered = center_v(v)
    
    # M @ cov @ M^T
    # Since M is symmetric and idempotent: M @ cov @ M
    row_means = cov.mean(dim=2, keepdim=True)  # (batch, k, 1)
    col_means = cov.mean(dim=1, keepdim=True)  # (batch, 1, k)
    total_mean = cov.mean(dim=(1, 2), keepdim=True)  # (batch, 1, 1)
    
    cov_centered = cov - row_means - col_means + total_mean

    # Scale normalization (c_star)
    # c_star = K / tr(M @ cov @ M^T)
    c_star_inv = torch.diagonal(cov_centered, dim1=1, dim2=2).mean(dim=1)  # (batch,)
    eps_tensor = torch.tensor(eps, device=v.device, dtype=v.dtype)
    c_star_inv = torch.clamp(c_star_inv, min=eps_tensor)
    c_star = 1. / c_star_inv
    
    # Apply scaling
    v_star = v_centered * torch.sqrt(c_star).unsqueeze(1)
    cov_star = cov_centered * c_star.unsqueeze(1).unsqueeze(2)
    
    return v_star, cov_star, c_star
