# DASH-Q

Implementation and evaluation of **DASH-Q: Robust Ultra Low-Bit Post-Training
Quantization via Stable Diagonal Curvature Estimate**
([arXiv:2604.13806](https://arxiv.org/abs/2604.13806)).

DASH-Q approximates the Hessian with only its diagonal entries (the per-channel
activation energy `h_jj = sum_k x_kj^2`), which stays stable even with small
calibration sets, and uses it to weight a closed-form per-group scale fit.

## Findings (TL;DR)

We tested DASH-Q against llama.cpp k-quants on Qwen/Qwen3.5-0.8B, wikitext-2
perplexity, calibrated on wikitext-2 train:

| bits | RTN (plain) | imatrix (Hessian) | DASH-Q kquant |
|------|-------------|-------------------|---------------|
| 2    | 103.07      | **29.74 (-71%)**  | 47.49 (-54%)  |
| 3    | 21.05       | **20.90**         | 23.29 (+11%)  |

Two conclusions:

1. **The diagonal Hessian is the whole win, and granularity is everything.**
   Feeding DASH-Q's diagonal Hessian to llama.cpp as an importance matrix and
   quantising to 16-group k-quant cuts 2-bit perplexity by **71%** over plain
   RTN, at identical size. Our earlier custom 32/64-group blocks lost to RTN
   purely because the group was too coarse, not because the algorithm was wrong.
2. **DASH-Q's own solver does not beat `make_qkx2`.** Packed into the same
   Q2_K/Q3_K grid (`dashq-kquant`), DASH-Q's closed-form solver beats plain RTN
   but loses to llama.cpp's importance-weighted search. DASH-Q optimises
   continuous fp16 scales; rounding those to the 4/6-bit k-quant grid discards
   the optimality, while `make_qkx2` searches directly in the quantised grid.

**So the recommended, deployable path is the `imatrix` variant** below. it is
DASH-Q's idea (diagonal-Hessian importance) delivered through stock llama.cpp.

### Reproduce

```bash
# build llama.cpp (the bundled fork works for every variant; the imatrix path
# also runs on upstream llama.cpp)
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -j --target llama-quantize llama-imatrix llama-perplexity && cd ..

# wikitext-2 train (calibration) and test (eval) as plain text files, then:
uv run python compare.py --model_id Qwen/Qwen3.5-0.8B --bits 2 3 \
    --ppl_file wiki.test.raw --imatrix_calib wiki.train.raw
```

`compare.py` builds each variant, runs `llama-perplexity`, and writes
`results.md` / `results.json` to `--out_dir`.

## Quick start

```bash
uv sync

# Recommended: DASH-Q Hessian -> llama.cpp k-quant, vs the RTN baseline (default)
uv run python compare.py --model_id Qwen/Qwen3.5-0.8B --bits 2 3 \
    --ppl_file wiki.test.raw --imatrix_calib wiki.train.raw
```

`compare.py --variants` selects which paths to build (default `rtn imatrix`):

| variant        | what it is                                                        |
|----------------|-------------------------------------------------------------------|
| `rtn`          | plain `llama-quantize` Q2_K/Q3_K/Q4_1, no calibration             |
| `imatrix`      | same k-quant weighted by the diagonal Hessian (**recommended**)   |
| `dashq-repack` | DASH-Q solver, dequantised then repacked into the k-quant block   |
| `dashq-native` | DASH-Q solver in custom DASHQ_2/3 blocks (needs the fork)         |
| `dashq-kquant` | DASH-Q solver packed straight into stock Q2_K/Q3_K (16-group)     |

## Experimental DASH-Q packers

The DASH-Q solver itself is in `src/dashq/dash_q.py`; `quantize_hf.py` runs it on
a HuggingFace model and writes GGUF. These paths lost to `imatrix` (see
findings) but remain available as `--method`:

```bash
# repack -> Q2_K/Q3_K/Q4_1 (default; stock llama.cpp)
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 2 --method repack
# kquant  -> DASH-Q values in stock Q2_K/Q3_K (16-group)
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 2 --method kquant
# native  -> custom DASHQ_2/3 blocks (needs the fork below)
uv run python quantize_hf.py --model_id Qwen/Qwen3.5-0.8B --bits 2 --method native
```

Tunables: `--bits {2,3,4}`, `--group_size`, `--iters`, `--alpha`,
`--lambda_reg`, `--n_samples`, `--seq_len`, `--dataset`, `--calib_format
{raw,chat}`. The Hessian for every layer is captured in a single forward pass.

## llama.cpp fork

The `native` method needs custom GGUF block types (`GGML_TYPE_DASHQ_2`,
`GGML_TYPE_DASHQ_3`) which only exist in the fork:

- **Repo:** <https://github.com/andrijdavid/llama.cpp>
- **Branch:** `dashq-quant`

It is vendored here under `llama.cpp/`. Every other variant (`rtn`, `imatrix`,
`dashq-repack`, `dashq-kquant`) produces standard GGUF and runs on upstream
llama.cpp.

```bash
git clone -b dashq-quant https://github.com/andrijdavid/llama.cpp
cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

## Development

```bash
uv sync --extra dev
uv run pytest -q        # algorithm + GGUF packer round-trips
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
