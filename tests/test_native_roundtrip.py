"""Round-trip tests for the native DASHQ_2 / DASHQ_3 packers."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dashq.dash_q import dequantize, quantize_layer
from dashq.export_native import (
    dequant_dashq_2,
    dequant_dashq_3,
    pack_dashq_2,
    pack_dashq_3,
)


def _roundtrip(bits: int, group_size: int, packer, dequant) -> None:
    torch.manual_seed(0)
    out_features, in_features = 8, group_size * 4
    W = torch.randn(out_features, in_features)
    X = torch.randn(64, in_features)

    Q, S, Z = quantize_layer(W, X, b=bits, group_size=group_size)
    W_hat = dequantize(Q, S, Z, group_size).numpy()

    # Pack with the native exporter, then unpack with the pure-Python
    # mirror of the C dequantiser.
    Q_grouped = Q.cpu().numpy().astype(np.uint8).reshape(-1, group_size)
    S_flat = S.cpu().numpy().astype(np.float32).flatten()
    Z_flat = Z.cpu().numpy().astype(np.float32).flatten()

    raw = packer(S_flat, Z_flat, Q_grouped)
    flat = dequant(raw, out_features * in_features).reshape(out_features, in_features)

    # The only loss going through the packer is fp16 rounding of d, z.
    # That keeps the absolute error well below max(|d|) * 1e-3.
    max_d = float(S.abs().max())
    max_z = float(Z.abs().max())
    tol = 1e-2 * max(max_d, max_z, 1.0)
    assert np.max(np.abs(flat - W_hat)) < tol, (
        f"bits={bits}: max diff "
        f"{np.max(np.abs(flat - W_hat)):.4g}, tol {tol:.4g}"
    )


def test_dashq_2_roundtrip():
    _roundtrip(2, 32, pack_dashq_2, dequant_dashq_2)


def test_dashq_3_roundtrip():
    _roundtrip(3, 64, pack_dashq_3, dequant_dashq_3)


if __name__ == "__main__":
    test_dashq_2_roundtrip()
    test_dashq_3_roundtrip()
    print("native pack/unpack round-trip OK")
