"""
DASH-Q -> llama.cpp GGUF exporter.

Provides bit-packing routines for the three quantisation types that
DASH-Q naturally maps onto:

    bits=2  ->  Q2_K  (super-block of 256 weights, 16 sub-blocks of 16)
    bits=3  ->  Q3_K  (super-block of 256 weights, 16 sub-blocks of 16)
    bits=4  ->  Q4_1  (block of 32 weights, one scale + one min)

The packing faithfully reproduces the binary layout defined in
ggml-common.h / ggml-quants.c so that llama.cpp can dequantise the
weights without any code changes.
"""

import numpy as np
import torch
import gguf

# -----------------------------------------------------------------------
#  Constants matching llama.cpp
# -----------------------------------------------------------------------
QK_K = 256  # super-block size for K-quants


# -----------------------------------------------------------------------
#  Q4_1 packer  (block_size = 32, 20 bytes/block)
# -----------------------------------------------------------------------
# struct block_q4_1 {
#     ggml_fp16_t d;      // scale
#     ggml_fp16_t m;      // min value
#     uint8_t qs[16];     // nibbles for 32 weights
# };
# Dequant:  w_i = d * q_i + m

def pack_q4_1(W_hat: torch.Tensor, group_size: int = 32) -> bytes:
    """
    Quantise a float weight matrix into Q4_1 blocks and return raw bytes.

    Args:
        W_hat: Reconstructed weight matrix, shape (out, in).
        group_size: Must be 32 for Q4_1.

    Returns:
        Raw bytes ready for GGUF.
    """
    assert group_size == 32, "Q4_1 block size is always 32"
    W_hat = W_hat.float().cpu()
    out_features, in_features = W_hat.shape
    assert in_features % group_size == 0
    num_groups = in_features // group_size

    buf = bytearray()

    for row in range(out_features):
        for g in range(num_groups):
            start = g * group_size
            block = W_hat[row, start:start + group_size].numpy()

            # Compute per-block min and scale (affine: w = d*q + m)
            bmin = float(block.min())
            bmax = float(block.max())
            d = (bmax - bmin) / 15.0 if bmax != bmin else 1.0
            m = bmin

            # Quantise to [0, 15]
            q = np.clip(np.round((block - m) / d), 0, 15).astype(np.uint8)

            # Write d and m as float16 (little-endian)
            buf.extend(np.float16(d).tobytes())
            buf.extend(np.float16(m).tobytes())

            # Pack 32 nibbles into 16 bytes
            # Byte layout: low nibble = q[i], high nibble = q[i+16]
            for i in range(16):
                packed = (q[i] & 0x0F) | ((q[i + 16] & 0x0F) << 4)
                buf.append(packed)

    return bytes(buf)


# -----------------------------------------------------------------------
#  Q2_K packer  (super-block of 256, 84 bytes/block)
# -----------------------------------------------------------------------
# struct block_q2_K {
#     uint8_t scales[QK_K/16];   // 16 bytes -- 4-bit sub-scale | 4-bit sub-min
#     uint8_t qs[QK_K/4];        // 64 bytes -- 2-bit quants
#     ggml_fp16_t d;              // super-block scale multiplier
#     ggml_fp16_t dmin;           // super-block min multiplier
# };
# Dequant per sub-block j (16 weights each):
#   dl = d * (scales[j] & 0xF)
#   ml = dmin * (scales[j] >> 4)
#   w_i = dl * q_i - ml

def pack_q2_k(W_hat: torch.Tensor, group_size: int = 256) -> bytes:
    """
    Quantise a float weight matrix into Q2_K blocks and return raw bytes.

    Args:
        W_hat: Reconstructed weight matrix, shape (out, in).
        group_size: Must be 256 for Q2_K.

    Returns:
        Raw bytes ready for GGUF.
    """
    assert group_size == QK_K, f"Q2_K super-block size is {QK_K}"
    W_hat = W_hat.float().cpu()
    out_features, in_features = W_hat.shape
    assert in_features % QK_K == 0
    num_blocks = in_features // QK_K

    buf = bytearray()

    for row in range(out_features):
        for blk in range(num_blocks):
            start = blk * QK_K
            block = W_hat[row, start:start + QK_K].numpy()

            # Step 1: per-sub-block (16 weights) affine quantisation
            n_sub = QK_K // 16  # 16 sub-blocks
            sub_scales = np.zeros(n_sub, dtype=np.float32)
            sub_mins = np.zeros(n_sub, dtype=np.float32)
            L = np.zeros(QK_K, dtype=np.uint8)

            for j in range(n_sub):
                sb = block[j * 16:(j + 1) * 16]
                sb_min = float(sb.min())
                sb_max = float(sb.max())
                if sb_max == sb_min:
                    sub_scales[j] = 0.0
                    sub_mins[j] = -sb_min if sb_min <= 0 else 0.0
                    L[j * 16:(j + 1) * 16] = 0
                    continue
                # Ensure min <= 0 (llama.cpp convention)
                if sb_min > 0:
                    sb_min = 0.0
                d = (sb_max - sb_min) / 3.0
                sub_scales[j] = d
                sub_mins[j] = -sb_min  # stored as positive "the_min"
                q = np.clip(np.round((sb - sb_min) / d), 0, 3).astype(np.uint8)
                L[j * 16:(j + 1) * 16] = q

            # Step 2: quantise sub-scales and sub-mins to 4 bits
            max_scale = float(sub_scales.max())
            max_min = float(sub_mins.max())

            if max_scale > 0:
                inv_scale = 15.0 / max_scale
                q_scales = np.clip(np.round(inv_scale * sub_scales), 0, 15).astype(np.uint8)
                d_val = max_scale / 15.0
            else:
                q_scales = np.zeros(n_sub, dtype=np.uint8)
                d_val = 0.0

            if max_min > 0:
                inv_min = 15.0 / max_min
                q_mins = np.clip(np.round(inv_min * sub_mins), 0, 15).astype(np.uint8)
                dmin_val = max_min / 15.0
            else:
                q_mins = np.zeros(n_sub, dtype=np.uint8)
                dmin_val = 0.0

            # Step 3: re-quantise with the rounded sub-scales
            for j in range(n_sub):
                d_eff = d_val * float(q_scales[j])
                if d_eff == 0:
                    L[j * 16:(j + 1) * 16] = 0
                    continue
                m_eff = dmin_val * float(q_mins[j])
                sb = block[j * 16:(j + 1) * 16]
                q = np.clip(np.round((sb + m_eff) / d_eff), 0, 3).astype(np.uint8)
                L[j * 16:(j + 1) * 16] = q

            # Step 4: pack into block_q2_K byte layout
            # scales[16]: low nibble = q_scales, high nibble = q_mins
            scales_bytes = bytearray(16)
            for j in range(n_sub):
                scales_bytes[j] = (q_scales[j] & 0x0F) | ((q_mins[j] & 0x0F) << 4)
            buf.extend(scales_bytes)

            # qs[64]: pack four 2-bit values per byte
            qs = bytearray(64)
            for j_off in range(0, QK_K, 128):
                base = j_off // 4
                for l in range(32):
                    qs[base + l] = (
                        (L[j_off + l] & 3)
                        | ((L[j_off + l + 32] & 3) << 2)
                        | ((L[j_off + l + 64] & 3) << 4)
                        | ((L[j_off + l + 96] & 3) << 6)
                    )
            buf.extend(qs)

            # d and dmin as float16
            buf.extend(np.float16(d_val).tobytes())
            buf.extend(np.float16(dmin_val).tobytes())

    return bytes(buf)


# -----------------------------------------------------------------------
#  Q3_K packer  (super-block of 256, 110 bytes/block)
# -----------------------------------------------------------------------
# struct block_q3_K {
#     uint8_t hmask[QK_K/8];   // 32 bytes -- high bit of each quant
#     uint8_t qs[QK_K/4];      // 64 bytes -- low 2 bits of each quant
#     uint8_t scales[12];      // 12 bytes -- 6-bit sub-block scales
#     ggml_half d;              // 2 bytes -- super-block scale
# };
# Values are in [-4, 3], stored as unsigned [0, 7]:
#   low 2 bits in qs, high bit in hmask.
# Dequant:
#   scale_j = unpack_6bit(scales, j) - 32
#   w = d * scale_j * (q3_value - 4)   where q3_value in {0..7}

def pack_q3_k(W_hat: torch.Tensor, group_size: int = 256) -> bytes:
    """
    Quantise a float weight matrix into Q3_K blocks and return raw bytes.

    Args:
        W_hat: Reconstructed weight matrix, shape (out, in).
        group_size: Must be 256 for Q3_K.

    Returns:
        Raw bytes ready for GGUF.
    """
    assert group_size == QK_K, f"Q3_K super-block size is {QK_K}"
    W_hat = W_hat.float().cpu()
    out_features, in_features = W_hat.shape
    assert in_features % QK_K == 0
    num_blocks = in_features // QK_K
    n_sub = QK_K // 16  # 16 sub-blocks

    buf = bytearray()

    for row in range(out_features):
        for blk in range(num_blocks):
            start = blk * QK_K
            block = W_hat[row, start:start + QK_K].numpy()

            # Step 1: per-sub-block symmetric quantisation to [-4, 3]
            sub_scales = np.zeros(n_sub, dtype=np.float32)
            L = np.zeros(QK_K, dtype=np.int8)

            for j in range(n_sub):
                sb = block[j * 16:(j + 1) * 16]
                amax = float(np.abs(sb).max())
                if amax < 1e-10:
                    sub_scales[j] = 0.0
                    L[j * 16:(j + 1) * 16] = 4  # zero-point
                    continue
                d = amax / 4.0
                sub_scales[j] = d
                q = np.clip(np.round(sb / d), -4, 3).astype(np.int8)
                L[j * 16:(j + 1) * 16] = q + 4  # unsigned [0, 7]

            # Step 2: quantise sub-scales to 6 bits
            amax_scale = 0.0
            max_scale_val = 0.0
            for j in range(n_sub):
                a = abs(sub_scales[j])
                if a > amax_scale:
                    amax_scale = a
                    max_scale_val = sub_scales[j]

            if amax_scale > 0:
                iscale = -32.0 / max_scale_val
                d_super = 1.0 / iscale
            else:
                iscale = 0.0
                d_super = 0.0

            q_scales_6bit = np.zeros(n_sub, dtype=np.int32)
            for j in range(n_sub):
                l = int(round(iscale * sub_scales[j]))
                l = max(-32, min(31, l)) + 32  # now in [0, 63]
                q_scales_6bit[j] = l

            # Step 3: re-quantise with the rounded scales
            for j in range(n_sub):
                sc_restored = d_super * (q_scales_6bit[j] - 32)
                if abs(sc_restored) < 1e-30:
                    L[j * 16:(j + 1) * 16] = 0
                    continue
                sb = block[j * 16:(j + 1) * 16]
                q = np.clip(np.round(sb / sc_restored), -4, 3).astype(np.int8)
                L[j * 16:(j + 1) * 16] = q + 4

            # Step 4: pack into block_q3_K byte layout
            L_u8 = L.astype(np.uint8)

            # hmask[32]: high bit extraction
            hmask = bytearray(32)
            m_idx = 0
            hm = 1
            for j in range(QK_K):
                if L_u8[j] > 3:
                    hmask[m_idx] |= hm
                    L_u8[j] -= 4
                m_idx += 1
                if m_idx == 32:
                    m_idx = 0
                    hm <<= 1
            buf.extend(hmask)

            # qs[64]: pack low 2 bits
            qs = bytearray(64)
            for j_off in range(0, QK_K, 128):
                base = j_off // 4
                for l in range(32):
                    qs[base + l] = (
                        (L_u8[j_off + l] & 3)
                        | ((L_u8[j_off + l + 32] & 3) << 2)
                        | ((L_u8[j_off + l + 64] & 3) << 4)
                        | ((L_u8[j_off + l + 96] & 3) << 6)
                    )
            buf.extend(qs)

            # scales[12]: pack 16 x 6-bit values into 12 bytes
            sc12 = bytearray(12)
            for j in range(n_sub):
                l = q_scales_6bit[j]
                if j < 8:
                    sc12[j] = l & 0x0F
                else:
                    sc12[j - 8] |= (l & 0x0F) << 4
                sc12[j % 4 + 8] |= ((l >> 4) & 3) << (2 * (j // 4))
            buf.extend(sc12)

            # d as float16
            buf.extend(np.float16(d_super).tobytes())

    return bytes(buf)


# -----------------------------------------------------------------------
#  Unified helpers
# -----------------------------------------------------------------------

# Map bits -> (packer function, GGUF type, block size, bytes per block)
QUANT_REGISTRY = {
    2: (pack_q2_k, gguf.GGMLQuantizationType.Q2_K, 256, 84),
    3: (pack_q3_k, gguf.GGMLQuantizationType.Q3_K, 256, 110),
    4: (pack_q4_1, gguf.GGMLQuantizationType.Q4_1, 32, 20),
}


def pack_tensor(W_hat: torch.Tensor, bits: int) -> tuple[bytes, int, int]:
    """
    Quantise and pack a weight tensor for GGUF export.

    Args:
        W_hat: Reconstructed weight matrix (out, in).
        bits: Target bit width (2, 3, or 4).

    Returns:
        (raw_bytes, block_size, bytes_per_block)
    """
    if bits not in QUANT_REGISTRY:
        raise ValueError(f"Unsupported bit width {bits}. Choose from {list(QUANT_REGISTRY)}")
    pack_fn, _, block_size, bpb = QUANT_REGISTRY[bits]
    raw = pack_fn(W_hat, group_size=block_size)
    return raw, block_size, bpb


def gguf_type_for_bits(bits: int) -> gguf.GGMLQuantizationType:
    """Return the GGUF quantisation enum for a given bit width."""
    return QUANT_REGISTRY[bits][1]


def byte_shape_for_tensor(shape: tuple[int, int], bits: int) -> tuple[int, int]:
    """
    Compute the raw byte-array shape that the gguf library expects.
    """
    block_size = QUANT_REGISTRY[bits][2]
    bpb = QUANT_REGISTRY[bits][3]
    return (shape[0], (shape[1] // block_size) * bpb)
