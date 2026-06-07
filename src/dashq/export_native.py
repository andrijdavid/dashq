"""
Native DASH-Q GGUF Exporter
============================
Packs DASH-Q quantized weights directly into the custom GGML_TYPE_DASHQ_2 and
GGML_TYPE_DASHQ_3 block formats, avoiding the accuracy loss of re-quantising
through intermediate K-quant formats.

Block layout (from ggml-common.h):

    block_dashq_2 (group_size=32, 2-bit):
        ggml_half d      -- scale
        ggml_half z      -- zero-point
        uint8_t qs[8]    -- 4 x 2-bit quants per byte

    block_dashq_3 (group_size=64, 3-bit):
        ggml_half d      -- scale
        ggml_half z      -- zero-point
        uint8_t qs[16]   -- low 2 bits, 4 per byte
        uint8_t qh[8]    -- high bit, 8 per byte

    Dequant: w[i] = d * q[i] - z
"""

import numpy as np


def pack_dashq_2(scales, zeros, quants):
    """
    Pack DASH-Q 2-bit quantized data into block_dashq_2 format.
    """
    n_groups = scales.shape[0]
    assert quants.shape == (n_groups, 32)
    assert np.all(quants <= 3)

    buf = bytearray()

    for g in range(n_groups):
        d_f16 = np.float16(scales[g]).tobytes()
        z_f16 = np.float16(zeros[g]).tobytes()
        buf.extend(d_f16)
        buf.extend(z_f16)

        q = quants[g]
        for byte_idx in range(8):
            packed = 0
            for bit_idx in range(4):
                j = byte_idx * 4 + bit_idx
                packed |= int(q[j]) << (2 * bit_idx)
            buf.append(packed)

    return bytes(buf)


def pack_dashq_3(scales, zeros, quants):
    """
    Pack DASH-Q 3-bit quantized data into block_dashq_3 format.
    """
    n_groups = scales.shape[0]
    assert quants.shape == (n_groups, 64)
    assert np.all(quants <= 7)

    buf = bytearray()

    for g in range(n_groups):
        d_f16 = np.float16(scales[g]).tobytes()
        z_f16 = np.float16(zeros[g]).tobytes()
        buf.extend(d_f16)
        buf.extend(z_f16)

        q = quants[g]

        for byte_idx in range(16):
            packed = 0
            for bit_idx in range(4):
                j = byte_idx * 4 + bit_idx
                packed |= (int(q[j]) & 0x03) << (2 * bit_idx)
            buf.append(packed)

        for byte_idx in range(8):
            packed = 0
            for bit_idx in range(8):
                j = byte_idx * 8 + bit_idx
                packed |= ((int(q[j]) >> 2) & 0x01) << bit_idx
            buf.append(packed)

    return bytes(buf)


def dequant_dashq_2(packed_bytes, n_weights):
    group_size = 32
    block_size = 12
    n_groups = n_weights // group_size
    assert len(packed_bytes) == n_groups * block_size

    result = np.zeros(n_weights, dtype=np.float32)
    offset = 0

    for g in range(n_groups):
        d = float(np.frombuffer(packed_bytes[offset:offset+2], dtype=np.float16)[0])
        z = float(np.frombuffer(packed_bytes[offset+2:offset+4], dtype=np.float16)[0])
        offset += 4

        for byte_idx in range(8):
            byte_val = packed_bytes[offset + byte_idx]
            for bit_idx in range(4):
                j = byte_idx * 4 + bit_idx
                q = (byte_val >> (2 * bit_idx)) & 0x03
                result[g * group_size + j] = d * q - z
        offset += 8

    return result


def dequant_dashq_3(packed_bytes, n_weights):
    group_size = 64
    block_size = 28
    n_groups = n_weights // group_size
    assert len(packed_bytes) == n_groups * block_size

    result = np.zeros(n_weights, dtype=np.float32)
    offset = 0

    for g in range(n_groups):
        d = float(np.frombuffer(packed_bytes[offset:offset+2], dtype=np.float16)[0])
        z = float(np.frombuffer(packed_bytes[offset+2:offset+4], dtype=np.float16)[0])
        offset += 4

        qs = packed_bytes[offset:offset+16]
        offset += 16

        qh = packed_bytes[offset:offset+8]
        offset += 8

        for j in range(group_size):
            lo = (qs[j // 4] >> (2 * (j % 4))) & 0x03
            hi = (qh[j // 8] >> (j % 8)) & 0x01
            q = lo | (hi << 2)
            result[g * group_size + j] = d * q - z

    return result
