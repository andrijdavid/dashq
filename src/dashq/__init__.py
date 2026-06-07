"""DASH-Q: Robust Ultra Low-Bit Post-Training Quantization."""

from dashq.dash_q import (  # noqa: F401
    PAPER_GROUP_SIZES,
    dash_q_group,
    dequantize,
    quantize_layer,
)
from dashq.export import (  # noqa: F401
    QUANT_REGISTRY,
    byte_shape_for_tensor,
    gguf_type_for_bits,
    pack_tensor,
)

__all__ = [
    "PAPER_GROUP_SIZES",
    "QUANT_REGISTRY",
    "byte_shape_for_tensor",
    "dash_q_group",
    "dequantize",
    "gguf_type_for_bits",
    "pack_tensor",
    "quantize_layer",
]
