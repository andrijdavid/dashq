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
    Quantize a weight group using DASH-Q (Algorithm 1) with a k-quant-style
    scale search.

    Two stages, both minimising the Hessian-weighted reconstruction error:
      (1) Scale search -- sweep candidate ranges around min-max; for each,
          assign integer codes and solve the closed-form weighted least squares
          for scale s and zero-point z (Eq. 10, 11). Keep the per-row best.
          This is what makes k-quant robust to outliers; here it runs on the
          diagonal-Hessian weights so it stays activation-aware.
      (2) Coordinate-descent refinement -- a few rounds of re-round then re-fit
          from the best candidate. Every iterate is kept only if it lowers the
          error, so the result never regresses.

    Args:
        W_group:    Float tensor (out_features, group_size).
        H_group:    Diagonal Hessian (group_size,), i.e. h_jj = sum(x_kj^2).
        b:          Number of quantization bits.
        T:          Refinement iterations (default: 9).
        alpha:      Unused. Retained for backward compatibility; the
                    best-error tracking supersedes the old damping.
        lambda_reg: Ridge regulariser (default: 1e-2). Stabilises the least
                    squares on near-constant weight groups.

    Returns:
        Q: Quantised integers (out_features, group_size), values in [0, 2^b-1].
        s: Per-row scales   (out_features, 1).
        z: Per-row offsets  (out_features, 1), with w_hat = s*Q - z.
    """
    out_features, group_size = W_group.shape
    q_max = (2 ** b) - 1

    # Broadcast H to (1, group_size) so it multiplies across rows. The diagonal
    # Hessian weights are per input-feature (column), shared across output rows.
    h = H_group.view(1, -1)
    w = W_group
    w_min = w.min(dim=1, keepdim=True).values
    w_max = w.max(dim=1, keepdim=True).values
    rng = (w_max - w_min).clamp(min=1e-9)

    # Per-row Hessian-weighted sums that don't depend on the integer codes.
    sw = h.sum()                                   # scalar (weights shared)
    sx = (h * w).sum(dim=1, keepdim=True)          # (out, 1)

    def _solve(L):
        """Closed-form Hessian-weighted least squares for (scale, min) given the
        integer codes L. Reconstruction is w_hat = scale*L + min, i.e. Eq. 10/11
        with z = -min. The ridge term lambda_reg keeps it stable on flat groups.
        """
        sl = (h * L).sum(dim=1, keepdim=True)
        sl2 = (h * L * L).sum(dim=1, keepdim=True)
        sxl = (h * w * L).sum(dim=1, keepdim=True)
        D = sw * sl2 - sl * sl + lambda_reg
        scale = ((sw * sxl - sx * sl) / D).clamp(min=1e-8)
        minv = (sl2 * sx - sl * sxl) / D
        return scale, minv

    best_err = torch.full((out_features, 1), float("inf"),
                          dtype=w.dtype, device=w.device)
    best_s = torch.ones((out_features, 1), dtype=w.dtype, device=w.device)
    best_z = torch.zeros((out_features, 1), dtype=w.dtype, device=w.device)
    best_Q = torch.zeros_like(w)

    def _consider(L, scale, minv):
        nonlocal best_err, best_s, best_z, best_Q
        w_hat = scale * L + minv
        err = (h * (w - w_hat) ** 2).sum(dim=1, keepdim=True)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_s = torch.where(better, scale, best_s)
        best_z = torch.where(better, -minv, best_z)
        best_Q = torch.where(better, L, best_Q)

    # Candidate 0: collapse the whole group to its Hessian-weighted mean. This
    # is the optimal code for a (near-)constant group, where the LS solve below
    # is degenerate.
    _consider(torch.zeros_like(w),
              torch.full((out_features, 1), 1e-8, dtype=w.dtype, device=w.device),
              sx / sw)

    # -- Scale search (brings k-quant's make_qkx2 robustness to DASH-Q) --
    # Sweep candidate ranges around min-max; for each, assign integer codes and
    # solve Eq. 10/11 in closed form, keeping whichever minimises the
    # Hessian-weighted error per row. Outliers no longer dictate the scale.
    n_steps, r_min, r_delta = 40, -2.0, 0.1
    for i in range(n_steps + 1):
        iscale = (r_min + r_delta * i + q_max) / rng
        L = torch.clamp(torch.round(iscale * (w - w_min)), 0, q_max)
        scale, minv = _solve(L)
        _consider(L, scale, minv)

    # -- Coordinate-descent refinement from the best candidate (Algorithm 1) --
    scale, minv = best_s.clone(), -best_z
    for _t in range(T):
        L = torch.clamp(torch.round((w - minv) / scale), 0, q_max)
        scale, minv = _solve(L)
        _consider(L, scale, minv)

    return best_Q, best_s, best_z


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
    # Diagonal Hessian: h_jj = sum_k x_{kj}^2  (Section 5)
    H_diag = (X ** 2).sum(dim=0)  # shape: (in_features,)
    return quantize_layer_from_hessian(
        W, H_diag, b, group_size=group_size, T=T, alpha=alpha, lambda_reg=lambda_reg
    )


def quantize_layer_from_hessian(
    W: torch.Tensor,
    H_diag: torch.Tensor,
    b: int,
    group_size: int | None = None,
    T: int = 9,
    alpha: float = 0.5,
    lambda_reg: float = 1e-2,
):
    """Same as :func:`quantize_layer` but takes a precomputed diagonal Hessian.

    Lets the caller accumulate ``H = sum_k x_k^2`` for every layer in a single
    forward pass instead of re-running the model once per layer.

    Args:
        W:      Weight tensor (out_features, in_features).
        H_diag: Diagonal Hessian (in_features,), i.e. sum over calibration
                tokens of x_{kj}^2.
    """
    # Default to paper-recommended group size if not specified
    if group_size is None:
        group_size = PAPER_GROUP_SIZES.get(b, 128)

    out_features, in_features = W.shape
    assert in_features % group_size == 0, (
        f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    )
    assert H_diag.shape[-1] == in_features, (
        f"H_diag length ({H_diag.shape[-1]}) must match in_features ({in_features})"
    )
    num_groups = in_features // group_size
    H_diag = H_diag.to(W.device, W.dtype)

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
