"""
DASH-Q: Robust Ultra Low-Bit Post-Training Quantization via
         Stable Diagonal Curvature Estimate.

Reference: arXiv:2604.13806v1

The key idea is to approximate the Hessian with only its diagonal
(feature-importance) entries, which are statistically stable even
with tiny calibration sets. This decouples the optimisation into
independent weighted-least-squares sub-problems per group, each
solved in closed form.
"""

import torch

# Paper-recommended group sizes (Section 6.1):
#   "group sizes of 128 for 4-bit, 64 for 3-bit, and 32 for 2-bit"
PAPER_GROUP_SIZES = {4: 128, 3: 64, 2: 32}


def dash_q_group(
    W_group: torch.Tensor,
    H_group: torch.Tensor,
    b: int,
    T: int = 9,
    alpha: float = 0.5,
    lambda_reg: float = 1e-2,
):
    """
    Quantize a weight group using DASH-Q coordinate descent (Algorithm 1).

    The algorithm alternates between:
      (1) Integer Refinement -- round weights to nearest integer grid point
      (2) Parameter Regression -- solve closed-form weighted least squares
          for scale s and zero-point z (Eq. 10, 11), then apply alpha-damping.

    Args:
        W_group:    Float tensor (out_features, group_size).
        H_group:    Diagonal Hessian (group_size,), i.e. h_jj = sum(x_kj^2).
        b:          Number of quantization bits.
        T:          Coordinate descent iterations (default: 9).
        alpha:      Damping factor (default: 0.5).  New parameters are
                    blended as  s <- alpha * s_new + (1 - alpha) * s_old.
        lambda_reg: Ridge regulariser on s^2 (default: 1e-2). Prevents
                    scale blow-up in near-constant weight groups.

    Returns:
        Q: Quantised integers (out_features, group_size), values in [0, 2^b-1].
        s: Per-row scales   (out_features, 1).
        z: Per-row offsets  (out_features, 1).
    """
    # out_features, group_size = W_group.shape
    q_max = (2 ** b) - 1

    # Broadcast H to (1, group_size) so it multiplies across rows
    H_group = H_group.view(1, -1)

    # -- Initialisation (Eq. 12) --
    # Standard min-max affine quantisation as the starting point.
    w_min = W_group.min(dim=1, keepdim=True).values
    w_max = W_group.max(dim=1, keepdim=True).values

    s = (w_max - w_min) / q_max
    s = torch.clamp(s, min=1e-8)
    # z is a full-precision offset: q = round((w + z) / s), w_hat = s*q - z
    z = -w_min

    # Precompute the Hessian-weighted mean of the *original* weights.
    # This is constant across iterations since W does not change.
    H_sum = torch.clamp(H_group.sum(), min=1e-8)
    w_bar = (W_group * H_group).sum(dim=1, keepdim=True) / H_sum

    # -- Coordinate descent loop --
    for _t in range(T):
        # Step 1 -- Integer Refinement:  fix (s, z), update Q
        Q = torch.clamp(torch.round((W_group + z) / s), 0, q_max)

        # Step 2 -- Parameter Regression:  fix Q, update (s, z)
        q_bar = (Q * H_group).sum(dim=1, keepdim=True) / H_sum

        # Weighted covariance Cov_h(w, q) and variance Var_h(q)
        cov_wq = (H_group * (W_group - w_bar) * (Q - q_bar)).sum(
            dim=1, keepdim=True
        ) / H_sum
        var_q = (H_group * (Q - q_bar) ** 2).sum(dim=1, keepdim=True) / H_sum

        # Closed-form optimal scale (Eq. 10)
        s_new = cov_wq / (var_q + lambda_reg)
        s_new = torch.clamp(s_new, min=1e-8)

        # Closed-form optimal zero-point (Eq. 11)
        z_new = s_new * q_bar - w_bar

        # alpha-damping: blend new solution with previous iterate
        s = alpha * s_new + (1 - alpha) * s
        z = alpha * z_new + (1 - alpha) * z

    # -- Final integer refinement --
    Q = torch.clamp(torch.round((W_group + z) / s), 0, q_max)
    return Q, s, z


def quantize_layer(
    W: torch.Tensor,
    X: torch.Tensor,
    b: int,
    group_size: int | None = None,
    T: int = 9,
    alpha: float = 0.5,
    lambda_reg: float = 1e-2,
):
    """
    Quantize a full linear-layer weight matrix using DASH-Q.

    Args:
        W:          Weight tensor (out_features, in_features).
        X:          Calibration activations (N, in_features).
        b:          Bit width (2, 3, or 4).
        group_size: Quantization group size.  If None, uses the paper's
                    recommended value for the given bit width.
        T:          Coordinate descent iterations (default: 9).
        alpha:      Damping factor (default: 0.5).
        lambda_reg: Ridge regulariser (default: 1e-2).

    Returns:
        Q_full: Integer codes  (out_features, in_features).
        S_full: Scales         (out_features, num_groups).
        Z_full: Zero-points    (out_features, num_groups).
    """
    # Default to paper-recommended group size if not specified
    if group_size is None:
        group_size = PAPER_GROUP_SIZES.get(b, 128)

    out_features, in_features = W.shape
    assert in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )
    num_groups = in_features // group_size

    # Diagonal Hessian: h_jj = sum_k x_{kj}^2  (Section 5)
    H_diag = (X ** 2).sum(dim=0)  # shape: (in_features,)

    Q_full = torch.zeros_like(W)
    S_full = torch.zeros((out_features, num_groups), dtype=W.dtype, device=W.device)
    Z_full = torch.zeros((out_features, num_groups), dtype=W.dtype, device=W.device)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size

        Q, s, z = dash_q_group(
            W[:, start:end], H_diag[start:end], b, T, alpha, lambda_reg
        )

        Q_full[:, start:end] = Q
        S_full[:, g] = s.squeeze(-1)
        Z_full[:, g] = z.squeeze(-1)

    return Q_full, S_full, Z_full


def dequantize(
    Q: torch.Tensor,
    S: torch.Tensor,
    Z: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Reconstruct float weights from DASH-Q output.

    Per group g:
        W_hat[:, g*gs:(g+1)*gs] = S[:, g] * Q[:, g*gs:(g+1)*gs] - Z[:, g]

    Args:
        Q:          Integer codes  (out_features, in_features).
        S:          Scales         (out_features, num_groups).
        Z:          Zero-points    (out_features, num_groups).
        group_size: Number of weights per group.

    Returns:
        W_hat: Reconstructed weight matrix (same shape as Q).
    """
    _, in_features = Q.shape
    num_groups = in_features // group_size
    W_hat = torch.zeros_like(Q, dtype=S.dtype)

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        W_hat[:, start:end] = S[:, g:g + 1] * Q[:, start:end] - Z[:, g:g + 1]

    return W_hat


# -----------------------------------------------------------------------
#  Quick self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    in_features = 256
    out_features = 128

    W = torch.randn(out_features, in_features)
    X = torch.randn(16, in_features)

    for bits in [2, 3, 4]:
        gs = PAPER_GROUP_SIZES[bits]
        Q, S, Z = quantize_layer(W, X, b=bits)  # uses default group_size
        W_hat = dequantize(Q, S, Z, gs)
        mse = torch.nn.functional.mse_loss(W_hat, W)
        print(f"{bits}-bit  group_size={gs:3d}  MSE={mse.item():.4f}")
