"""
Pack DASH-Q's Hessian-optimised quantisation into stock k-quant blocks
(GGML_TYPE_Q2_K / Q3_K) at 16-weight granularity, so the result runs on
unmodified llama.cpp while carrying DASH-Q's scales instead of llama.cpp's
default magnitude-weighted ones.

Why reuse Q2_K/Q3_K rather than a new block type: a bit-matched 16-group block
*is* the k-quant layout, so a custom type would only duplicate its kernels.

Byte layouts mirror gguf-py's dequantisers exactly (validated in demo()):
  Q2_K (256 w = 16 sub-blocks of 16, 2.625 bpw):
    scales[16] : (min4<<4)|scale4   qs[64] : 2-bit codes   d, dmin : fp16
    w = d*scale4 * q - dmin*min4,  q in [0,3]
  Q3_K (256 w, symmetric, 3.4375 bpw):
    hmask[32] qs[64] scales[12] (6-bit) d:fp16
    w = d*(scale6-32) * q,  q in [-4,3]
"""

import numpy as np
import torch

from .dash_q import quantize_layer_from_hessian

QK_K = 256


def _f16(x):
    return np.float16(x).view(np.uint8).reshape(*x.shape, 2)


def dashq_q2k(W: torch.Tensor, H: torch.Tensor):
    """W (out,in) float, H (in,) diagonal Hessian -> (uint8 bytes, [out,in])."""
    out_f, in_f = W.shape
    assert in_f % QK_K == 0, f"in_features {in_f} must be a multiple of {QK_K}"

    # DASH-Q at 16-group granularity gives the optimised per-sub-block scale s
    # and offset z (w_hat = s*q - z). We keep s/z and re-derive codes once the
    # scales are quantised to the Q2_K 4-bit grid.
    _, S, Z = quantize_layer_from_hessian(W.float(), H.float(), b=2, group_size=16)
    s = S.cpu().numpy().reshape(-1, 16)                 # (nb, 16) sub-block scales
    z = np.clip(Z.cpu().numpy().reshape(-1, 16), 0, None)  # min >= 0 for Q2_K
    w = W.float().cpu().numpy().reshape(-1, QK_K)        # (nb, 256)
    nb = s.shape[0]

    d = s.max(axis=1) / 15.0
    dmin = z.max(axis=1) / 15.0
    inv_d = np.where(d > 0, 1.0 / d, 0.0)[:, None]
    inv_dmin = np.where(dmin > 0, 1.0 / dmin, 0.0)[:, None]
    sc = np.clip(np.rint(s * inv_d), 0, 15).astype(np.uint8)   # 4-bit scales
    mn = np.clip(np.rint(z * inv_dmin), 0, 15).astype(np.uint8)  # 4-bit mins

    dl = (d[:, None] * sc).astype(np.float32)     # reconstructed scale per sub-block
    ml = (dmin[:, None] * mn).astype(np.float32)
    dl_w = np.repeat(dl, 16, axis=1)              # broadcast to 256
    ml_w = np.repeat(ml, 16, axis=1)
    inv_dl = np.divide(1.0, dl_w, out=np.zeros_like(dl_w), where=dl_w > 0)
    q = np.clip(np.rint((w + ml_w) * inv_dl), 0, 3).astype(np.uint8)

    scales = ((mn << 4) | sc).astype(np.uint8)    # (nb,16)
    qs = np.zeros((nb, 64), dtype=np.uint8)
    for n in range(QK_K):
        byte = (n // 128) * 32 + (n % 32)
        shift = 2 * ((n % 128) // 32)
        qs[:, byte] |= q[:, n] << shift

    block = np.concatenate(
        [scales, qs, _f16(d.astype(np.float16).astype(np.float32)).reshape(nb, 2),
         _f16(dmin.astype(np.float16).astype(np.float32)).reshape(nb, 2)],
        axis=1)
    return block.reshape(-1), np.array([out_f, in_f], dtype=np.int64)


def _sym_scale(w, h, qmin=-4, qmax=3, lam=1e-2):
    """Hessian-weighted symmetric scale per sub-block (Q3_K has no zero point).

    w, h: (nsub, 16). Returns a > 0 of shape (nsub, 1) minimising
    sum h*(w - a*q)^2 with q = clip(round(w/a), qmin, qmax). Same search/refit
    idea as dash_q_group, scale-only.
    """
    a0 = np.abs(w).max(axis=1, keepdims=True) / abs(qmin)
    a0 = np.maximum(a0, 1e-8)
    best_err = np.full((w.shape[0], 1), np.inf)
    best_a = a0.copy()

    def consider(a):
        nonlocal best_err, best_a
        a = np.maximum(a, 1e-8)
        q = np.clip(np.rint(w / a), qmin, qmax)
        a_fit = (h * w * q).sum(1, keepdims=True) / ((h * q * q).sum(1, keepdims=True) + lam)
        a_fit = np.maximum(a_fit, 1e-8)
        err = (h * (w - a_fit * q) ** 2).sum(1, keepdims=True)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_a = np.where(better, a_fit, best_a)

    for f in np.linspace(0.5, 1.3, 25):
        consider(a0 * f)
    for _ in range(3):
        consider(best_a)
    return best_a


def dashq_q3k(W: torch.Tensor, H: torch.Tensor):
    """W (out,in) float, H (in,) diagonal Hessian -> (uint8 bytes, [out,in])."""
    out_f, in_f = W.shape
    assert in_f % QK_K == 0, f"in_features {in_f} must be a multiple of {QK_K}"
    w_sub = W.float().cpu().numpy().reshape(-1, 16)            # (nsub, 16)
    h_sub = np.tile(H.float().cpu().numpy().reshape(-1, 16), (out_f, 1))
    a = _sym_scale(w_sub, h_sub)                              # (nsub, 1)

    nb = out_f * in_f // QK_K
    a = a.reshape(nb, 16)
    d = np.abs(a).max(axis=1) / 31.0
    inv_d = np.where(d > 0, 1.0 / d, 0.0)[:, None]
    sc6 = np.clip(np.rint(a * inv_d), -32, 31).astype(np.int32)   # signed 6-bit
    dl = (d[:, None] * sc6).astype(np.float32)                   # (nb,16)

    w = W.float().cpu().numpy().reshape(nb, QK_K)
    dl_w = np.repeat(dl, 16, axis=1)
    inv_dl = np.divide(1.0, dl_w, out=np.zeros_like(dl_w), where=dl_w != 0)
    q = np.clip(np.rint(w * inv_dl), -4, 3).astype(np.int32)     # (nb,256) in [-4,3]

    # codes: ql = q + (4 if q<0 else 0) in [0,3]; stored hmask bit = (q>=0)
    ql = (q + np.where(q < 0, 4, 0)).astype(np.uint8)
    hbit = (q >= 0).astype(np.uint8)

    qs = np.zeros((nb, 64), dtype=np.uint8)
    hmask = np.zeros((nb, 32), dtype=np.uint8)
    for n in range(QK_K):
        qs[:, (n // 128) * 32 + (n % 32)] |= ql[:, n] << (2 * ((n % 128) // 32))
        hmask[:, n % 32] |= hbit[:, n] << (n // 32)

    # 16 6-bit unsigned scales -> 8 low-nibble bytes + 4 high-2-bit bytes
    u = (sc6 + 32).astype(np.uint8)                              # [0,63]
    scales = np.zeros((nb, 12), dtype=np.uint8)
    for k in range(8):
        scales[:, k] = (u[:, k] & 0xF) | ((u[:, k + 8] & 0xF) << 4)
    for k in range(4):
        scales[:, 8 + k] = (((u[:, k] >> 4) & 3) | (((u[:, k + 4] >> 4) & 3) << 2)
                            | (((u[:, k + 8] >> 4) & 3) << 4) | (((u[:, k + 12] >> 4) & 3) << 6))

    block = np.concatenate(
        [hmask, qs, scales, _f16(d.astype(np.float16).astype(np.float32)).reshape(nb, 2)],
        axis=1)
    return block.reshape(-1), np.array([out_f, in_f], dtype=np.int64)


def demo():
    import gguf
    from gguf.quants import Q2_K
    torch.manual_seed(0)
    out_f, in_f = 32, QK_K * 2
    W = torch.randn(out_f, in_f) * 0.04
    X = torch.randn(2000, in_f) * (torch.rand(in_f) ** 3 * 5 + 0.1)
    H = (X ** 2).sum(0)

    raw, shape = dashq_q2k(W, H)
    # gguf-py is llama.cpp's reference dequantiser: if it reads back our blocks
    # as valid Q2_K with sensible reconstruction, the byte layout is correct.
    deq = Q2_K.dequantize_blocks(raw.reshape(-1, 84)).reshape(out_f, in_f)
    rmse = np.sqrt(((deq - W.numpy()) ** 2).mean())
    base = np.sqrt((W.numpy() ** 2).mean())
    print(f"Q2_K {gguf.GGMLQuantizationType.Q2_K.name}: recon RMSE={rmse:.4f} "
          f"(|W| RMS {base:.4f})")
    assert rmse < 0.5 * base, "Q2_K reconstruction worse than expected -> packing bug"
    print("Q2_K OK")

    from gguf.quants import Q3_K
    raw3, _ = dashq_q3k(W, H)
    deq3 = Q3_K.dequantize_blocks(raw3.reshape(-1, 110)).reshape(out_f, in_f)
    rmse3 = np.sqrt(((deq3 - W.numpy()) ** 2).mean())
    print(f"Q3_K {gguf.GGMLQuantizationType.Q3_K.name}: recon RMSE={rmse3:.4f} "
          f"(|W| RMS {base:.4f})")
    assert rmse3 < 0.4 * base, "Q3_K reconstruction worse than expected -> packing bug"
    print("Q3_K OK")


if __name__ == "__main__":
    demo()
