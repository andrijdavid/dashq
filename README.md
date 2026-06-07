# DASH-Q

Implementation of **DASH-Q: Robust Ultra Low-Bit Post-Training Quantization
via Stable Diagonal Curvature Estimate** ([arXiv:2604.13806](https://arxiv.org/abs/2604.13806)).

DASH-Q approximates the Hessian with only its diagonal entries, which remain
statistically stable even with small calibration sets. This decouples the
optimisation into independent weighted-least-squares sub-problems per group,
each solved in closed form.

## Quick start

```bash
# Install
uv sync

# Quantise a HuggingFace model (4-bit, paper defaults)
uv run python quantize_hf.py --model_id Qwen/Qwen2.5-0.5B --bits 4

# 2-bit quantisation
uv run python quantize_hf.py --model_id Qwen/Qwen2.5-0.5B --bits 2

# Instruction-tuned model: calibrate on chat-format data (alpaca by default).
# Each row is rendered through the tokenizer's chat template.
uv run python quantize_hf.py --model_id Qwen/Qwen2.5-0.5B-Instruct --bits 4 \
    --calib_format chat

# Custom parameters
uv run python quantize_hf.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --bits 3 \
    --group_size 128 \
    --iters 15 \
    --alpha 0.7
```

## Paper defaults

| Parameter    | Default | Description                          |
|-------------|---------|--------------------------------------|
| `--bits`    | 4       | Bit width: 2, 3, or 4               |
| `--group_size` | auto | 128 (4-bit), 64 (3-bit), 32 (2-bit) |
| `--iters`   | 9       | Coordinate descent iterations        |
| `--alpha`   | 0.5     | Damping factor                       |
| `--lambda_reg` | 0.01 | Ridge regularisation                 |
| `--n_samples` | 16    | Calibration sequences                |
| `--seq_len` | 2048    | Sequence length                      |
| `--dataset` | wikitext | Calibration dataset                 |
| `--calib_format` | raw | `raw` (concat text) or `chat` (instruction/answer) |

## Output formats

The quantised weights are exported to GGUF files compatible with
[llama.cpp](https://github.com/ggerganov/llama.cpp):

| Bits | GGUF type | Block size | bpw   |
|------|-----------|------------|-------|
| 2    | Q2_K      | 256        | 2.625 |
| 3    | Q3_K      | 256        | 3.438 |
| 4    | Q4_1      | 32         | 5.0   |

For 2 and 3 bits there is also a `--native` path that writes a custom
GGUF block type. It avoids the dequant + repack round-trip:

| Bits | GGUF type (native) | Block size | bpw |
|------|--------------------|------------|-----|
| 2    | DASHQ_2            | 32         | 3.0 |
| 3    | DASHQ_3            | 64         | 3.5 |

DASHQ_2 / DASHQ_3 only load with the forked llama.cpp shipped in this
repo (see below). They are useful when the model's `in_features` is not
a multiple of 256 (Qwen2.5-0.5B's 896 hidden, for example), in which
case Q2_K / Q3_K cannot be applied to most layers and stay F16.

## Build the forked llama.cpp

The fork lives in `llama.cpp/` on branch `dashq-quant`. It adds two
quantisation types (`GGML_TYPE_DASHQ_2`, `GGML_TYPE_DASHQ_3`) plus
their dequant kernels and Q8_0 vec-dot. CPU-only; no SIMD or CUDA
kernels are wired up.

```bash
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd ..
```

Sanity-check the new types:

```bash
cd llama.cpp/build && ctest -R test-quantize-fns -V
# ... Testing dashq_2 / Testing dashq_3 / Passed
```

## End-to-end test on Qwen/Qwen3.5-0.8B

Quantise + run inference with each path. `llama-simple` is used
because it is a thin wrapper around `llama_decode` — the chat-aware
`llama-cli` insists on applying the model's chat template, which
masks the actual generation when you only want a raw completion.

```bash
# 4-bit -> Q4_1 (block 32, fits any in_features divisible by 32)
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 4 \
    --out out-q4.gguf
llama.cpp/build/bin/llama-simple -m out-q4.gguf "The capital of France is"

# 3-bit, native DASHQ_3 (block 64)
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 3 \
    --native --out out-dashq3.gguf
llama.cpp/build/bin/llama-simple -m out-dashq3.gguf "The capital of France is"

# 2-bit, native DASHQ_2 (block 32)
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 2 \
    --native --out out-dashq2.gguf
llama.cpp/build/bin/llama-simple -m out-dashq2.gguf "The capital of France is"

# 3-bit / 2-bit repack path -> Q3_K / Q2_K (block 256)
# Only the linear layers whose in_features is a multiple of 256 get
# quantised; the rest stay F16. Skipped layers are printed.
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 3 \
    --out out-q3.gguf
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 2 \
    --out out-q2.gguf
```

Runtime numbers, fewer calibration samples for a quick smoke test:
add `--n_samples 4 --seq_len 512`. The full DASH-Q pipeline runs the
model once per linear layer to capture activations, so it is noticeably
slower than naive PTQ; this is by design.

To verify a quantised file matches the original weights up to fp16
rounding (handy for debugging packers):

```bash
uv run pytest tests/ -q
```

## Project structure

```
dashq/
  src/dashq/
    __init__.py       # Package exports
    dash_q.py         # Core DASH-Q algorithm
    export.py         # GGUF packing (Q2_K, Q3_K, Q4_1)
  quantize_hf.py      # CLI: quantise HuggingFace model
  notebooks/           # Walkthrough notebooks
  pyproject.toml
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## Reference

```bibtex
@article{dashq2024,
  title={DASH-Q: Robust Ultra Low-Bit Post-Training Quantization
         via Stable Diagonal Curvature Estimate},
  year={2024},
  eprint={2604.13806},
  archivePrefix={arXiv}
}
```
