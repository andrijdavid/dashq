"""
Quantise a HuggingFace model with DASH-Q and export to GGUF.

Usage examples:
    # 4-bit (-> Q4_1), paper defaults
    uv run python quantize_hf.py --model_id Qwen/Qwen2.5-0.5B

    # 2-bit (-> Q2_K)
    uv run python quantize_hf.py --model_id Qwen/Qwen2.5-0.5B --bits 2

    # Custom group size and iterations
    uv run python quantize_hf.py --bits 3 --group_size 128 --iters 15
"""

import argparse
import gc
import sys
import os

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llama.cpp", "gguf-py"))
import gguf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dashq.dash_q import (
    quantize_layer,
    quantize_layer_from_hessian,
    dequantize,
    PAPER_GROUP_SIZES,
)
from dashq.export import (
    QUANT_REGISTRY,
    pack_tensor,
    gguf_type_for_bits,
    byte_shape_for_tensor,
)
from dashq.export_native import pack_dashq_2, pack_dashq_3

# Default calibration dataset. wikitext-2 is the standard for raw-text PTQ
# (AWQ, GPTQ). For instruction-tuned models we default to alpaca rendered
# through the model's chat template.
DEFAULT_DATASET = "wikitext"
DEFAULT_DATASET_CONFIG = "wikitext-2-raw-v1"
DEFAULT_CHAT_DATASET = "tatsu-lab/alpaca"

# Tensor categories that llama.cpp always keeps in F32. Mirrors the
# "always float32" list in convert_hf_to_gguf.py:prepare_tensors(). Several
# CPU kernels (e.g. ggml_compute_forward_ssm_conv_f32) assert their weight is
# F32, so a hybrid model (Mamba/SSM, RWKV time-mix, ...) crashes at inference
# if these get written as F16. Keep in sync with upstream when rebasing.
_F32_TENSOR_KEYS = (
    "FFN_GATE_INP", "FFN_GATE_INP_SHEXP", "POS_EMBD", "TOKEN_TYPES",
    "SSM_CONV1D", "SHORTCONV_CONV", "TIME_MIX_FIRST", "TIME_MIX_W1",
    "TIME_MIX_W2", "TIME_MIX_DECAY_W1", "TIME_MIX_DECAY_W2",
    "TIME_MIX_LERP_FUSED", "POSNET_NORM1", "POSNET_NORM2",
    "V_ENC_EMBD_POS", "A_ENC_EMBD_POS", "ALTUP_CORRECT_COEF",
    "ALTUP_PREDICT_COEF", "SSM_CONV1D_Q", "SSM_CONV1D_K", "SSM_CONV1D_V",
)


def nonquant_array(base_model, hf_name, ggml_name, bid, data_torch):
    """Numpy array for a tensor DASH-Q leaves unquantised, picking F16 vs F32
    with the same rules llama.cpp's converter applies in prepare_tensors().

    DASH-Q only touches the linear weights; everything else (norms, embeddings,
    SSM conv kernels, ...) is copied straight through. We must reproduce
    upstream's dtype choice here, otherwise tensors the runtime requires in F32
    get silently downcast to F16 and trip GGML_ASSERT at inference time.
    """
    n_dims = len(data_torch.shape)
    qtype = base_model.tensor_force_quant(hf_name, ggml_name, bid, n_dims)

    force_f32 = (
        n_dims <= 1
        or ggml_name.endswith("_norm.weight")
        or ggml_name[-7:] not in (".weight", ".lora_a", ".lora_b")
    )
    if not force_f32 and qtype is False:
        keys = [getattr(gguf.MODEL_TENSOR, k) for k in _F32_TENSOR_KEYS
                if hasattr(gguf.MODEL_TENSOR, k)]
        force_f32 = any(
            base_model.match_model_tensor_name(ggml_name, key, bid) for key in keys
        )

    if force_f32 or qtype == gguf.GGMLQuantizationType.F32:
        return data_torch.to(torch.float32).numpy()
    return data_torch.to(torch.float16).numpy()


# -----------------------------------------------------------------------
#  Calibration data loader
# -----------------------------------------------------------------------

def _render_chat_example(tokenizer, row) -> str | None:
    """Build a chat-template string from one row of an instruction dataset."""
    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    if instruction is None:
        return None
    extra = row.get("input") or ""
    if extra:
        instruction = f"{instruction}\n{extra}"
    answer = row.get("output") or row.get("response") or row.get("answer") or ""

    messages = [{"role": "user", "content": instruction}]
    if answer:
        messages.append({"role": "assistant", "content": answer})

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback if the tokenizer has no chat template configured.
        if answer:
            return f"User: {instruction}\nAssistant: {answer}\n"
        return f"User: {instruction}\n"


@torch.no_grad()
def get_calib_dataset(
    tokenizer,
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    n_samples: int = 128,
    seq_len: int = 2048,
    calib_format: str = "raw",
    max_rows: int | None = 128,
) -> torch.Tensor:
    """
    Load and tokenise a calibration dataset into fixed-length sequences.

    calib_format:
        "raw"  -- concatenate the dataset's `text` field, then chunk.
        "chat" -- render each row as instruction/answer via the tokenizer's
                  chat template, then concatenate and chunk.
    max_rows: if set, take only the first `max_rows` rows of the source
              dataset before rendering/concatenation. None means no cap.
    """
    if calib_format not in ("raw", "chat"):
        raise ValueError(f"--calib_format must be 'raw' or 'chat', got {calib_format!r}")

    if calib_format == "chat":
        if dataset_name == DEFAULT_DATASET:
            # User left dataset at its raw-text default. Switch.
            dataset_name = DEFAULT_CHAT_DATASET
            dataset_config = None
        elif dataset_config == DEFAULT_DATASET_CONFIG:
            # Custom chat dataset, drop the wikitext-specific config.
            dataset_config = None

    print(f"Loading calibration dataset '{dataset_name}'"
          f"{f' ({dataset_config})' if dataset_config else ''} [{calib_format}]...")
    dataset = load_dataset(dataset_name, dataset_config, split="train")

    if max_rows is not None and max_rows < len(dataset):
        dataset = dataset.select(range(max_rows))
        print(f"  Limited to first {max_rows} rows of the source dataset")

    if calib_format == "raw":
        if "text" not in dataset.column_names:
            raise ValueError(
                f"--calib_format raw needs a 'text' column; "
                f"{dataset_name} has {dataset.column_names}"
            )
        text = "\n\n".join(dataset["text"])
    else:
        rendered = []
        for row in dataset:
            s = _render_chat_example(tokenizer, row)
            if s:
                rendered.append(s)
        if not rendered:
            raise ValueError(
                f"No instruction/answer rows recognised in {dataset_name}. "
                f"Columns: {dataset.column_names}"
            )
        text = "".join(rendered)

    tokens = tokenizer(text, return_tensors="pt").input_ids[0]

    total_tokens = n_samples * seq_len
    if len(tokens) < total_tokens:
        n_samples = max(1, len(tokens) // seq_len)
        print(f"  Warning: only {n_samples} full sequences available.")

    tokens = tokens[: n_samples * seq_len]
    input_ids = tokens.view(n_samples, seq_len)
    print(f"  Calibration set: {input_ids.shape[0]} seqs x {input_ids.shape[1]} tokens")
    return input_ids


# -----------------------------------------------------------------------
#  Main quantisation + export pipeline
# -----------------------------------------------------------------------

@torch.no_grad()
def quantize_and_export(
    model_id: str,
    dataset_name: str,
    bits: int = 4,
    group_size: int | None = None,
    iters: int = 9,
    alpha: float = 0.5,
    lambda_reg: float = 1e-2,
    n_samples: int = 16,
    seq_len: int = 2048,
    out_file: str = "model-dash-q.gguf",
    native: bool = False,
    calib_format: str = "raw",
    max_rows: int | None = 128,
):
    """
    Load a HuggingFace model, quantise all linear layers with DASH-Q,
    and export the result to a GGUF file.
    """
    if bits not in QUANT_REGISTRY:
        raise ValueError(f"Unsupported --bits={bits}. Choose from {list(QUANT_REGISTRY)}.")

    # Resolve group size: user override or paper default
    dashq_group_size = group_size if group_size is not None else PAPER_GROUP_SIZES[bits]
    tensor_type = gguf_type_for_bits(bits)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"DASH-Q config: bits={bits}, group_size={dashq_group_size}, "
          f"T={iters}, alpha={alpha}, lambda={lambda_reg}")

    # -- Load model -------------------------------------------------------
    # "auto" lets a large model shard across whatever devices are available
    # (e.g. multiple GPUs). The per-layer ops below stay device-agnostic so
    # this works whether the model lands on one device or many.
    print(f"Loading model '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    calib_inputs = get_calib_dataset(
        tokenizer, dataset_name,
        n_samples=n_samples, seq_len=seq_len,
        calib_format=calib_format,
        max_rows=max_rows,
    )

    # -- Prepare GGUF writer -----------------------------------------------
    from huggingface_hub import snapshot_download
    import sys
    import os
    from pathlib import Path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "llama.cpp"))
    from convert_hf_to_gguf import ModelBase

    print(f"Downloading/Locating HF snapshot for '{model_id}'...")
    local_dir = snapshot_download(model_id, allow_patterns=["*.json", "*.model", "*.safetensors"])

    hparams = ModelBase.load_hparams(Path(local_dir), False)
    arch = hparams["architectures"][0]
    # Resolve the converter class through the lazy loader. `from_model_architecture`
    # alone never imports the per-arch module, so the registry is empty and every
    # lookup raises. For multimodal checkpoints (e.g. Qwen3.5 ships as
    # `*ForConditionalGeneration`) prefer the text `*ForCausalLM` converter, which
    # is what we quantise.
    candidates = [arch]
    if arch.endswith("ForConditionalGeneration"):
        candidates.insert(0, arch.replace("ForConditionalGeneration", "ForCausalLM"))
    try:
        from conversion import get_model_class
    except Exception:
        get_model_class = None
    ModelClass = None
    for cand in candidates:
        try:
            ModelClass = (get_model_class(cand) if get_model_class
                          else ModelBase.from_model_architecture(cand))
            break
        except (NotImplementedError, KeyError):
            continue
    if ModelClass is None:
        raise NotImplementedError(
            f"None of {candidates} is supported by the bundled converter")

    print("Extracting metadata for GGUF...")
    ftype = gguf.LlamaFileType.MOSTLY_Q2_K if bits == 2 else gguf.LlamaFileType.MOSTLY_Q3_K_M if bits == 3 else gguf.LlamaFileType.MOSTLY_Q4_0
    base_model = ModelClass(
        dir_model=Path(local_dir),
        ftype=ftype,
        fname_out=Path(out_file),
        is_big_endian=False,
        use_temp_file=False,
        hparams=hparams,
    )
    base_model.set_gguf_parameters()
    base_model.set_vocab()
    
    writer = base_model.gguf_writer

    quantized_hf_layers = {}

    # -- Find linear layers -------------------------------------------------
    linear_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }
    print(f"Found {len(linear_layers)} linear layers.\n")

    # -- Decide which layers to quantise -----------------------------------
    # Skip the language-model head (kept in fp16) and any layer whose
    # in_features is not divisible by the DASH-Q group size or the target
    # GGUF block size (Q2_K/Q3_K need multiples of 256, Q4_1 needs 32,
    # native DASHQ_2/3 need 32/64). Skipped layers stay F16.
    target_block = QUANT_REGISTRY[bits][2] if not (native and bits in (2, 3)) else (32 if bits == 2 else 64)
    targets = {}
    for name, module in linear_layers.items():
        if "lm_head" in name:
            continue
        if module.in_features % dashq_group_size != 0 or module.in_features % target_block != 0:
            print(f"  Skipping {name} (in_features={module.in_features} not "
                  f"divisible by dashq_group={dashq_group_size} or block={target_block})")
            continue
        targets[name] = module
    print(f"Quantising {len(targets)} layers.\n")

    # -- Accumulate the diagonal Hessian for every target in ONE pass -------
    # DASH-Q only needs h_jj = sum_k x_kj^2 per layer, not the full
    # activations, so we hook all targets at once and stream the calibration
    # set through the model a single time (instead of re-running it per layer).
    # Keep each layer's accumulator on that layer's own device so a sharded
    # model (weights spread across several GPUs) accumulates locally.
    hessians = {name: torch.zeros(m.in_features, dtype=torch.float32,
                                  device=m.weight.device)
                for name, m in targets.items()}

    def _make_hook(layer_name):
        def _hook(_mod, inp, _out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            contrib = (x * x).sum(dim=0)
            hessians[layer_name] += contrib.to(hessians[layer_name].device)
        return _hook

    handles = [m.register_forward_hook(_make_hook(name)) for name, m in targets.items()]
    chunk_size = max(1, min(4, n_samples))
    for i in tqdm(range(0, len(calib_inputs), chunk_size), desc="Calibrating"):
        batch = calib_inputs[i : i + chunk_size].to(device)
        model(batch)
    for h in handles:
        h.remove()

    # -- Quantise each target from its accumulated Hessian ------------------
    for name, module in tqdm(targets.items(), desc="Quantising"):
        # Run on the layer's own device (it may live on any shard); the
        # Hessian is already there, and packing moves results to CPU.
        W = module.weight.data.clone().float()
        Q, S, Z = quantize_layer_from_hessian(
            W, hessians[name], b=bits,
            group_size=dashq_group_size,
            T=iters,
            alpha=alpha,
            lambda_reg=lambda_reg,
        )

        # Pack weights into GGUF tensor format
        if native and bits in (2, 3):
            # Native DASH-Q format -- no re-quantisation loss
            n_rows, n_cols = W.shape
            Q_np = Q.cpu().numpy().astype(np.uint8)
            S_np = S.cpu().numpy().astype(np.float32)
            Z_np = Z.cpu().numpy().astype(np.float32)

            native_gs = 32 if bits == 2 else 64
            assert n_cols % native_gs == 0, (
                f"in_features ({n_cols}) not divisible by native group size {native_gs}"
            )
            n_groups_total = n_rows * (n_cols // native_gs)
            Q_grouped = Q_np.reshape(-1, native_gs)
            S_flat = S_np.flatten()
            Z_flat = Z_np.flatten()
            assert (
                Q_grouped.shape[0] == n_groups_total
                and S_flat.shape[0] == n_groups_total
                and Z_flat.shape[0] == n_groups_total
            ), "Q/S/Z group counts disagree -- check group_size and tensor shapes"

            if bits == 2:
                raw_bytes = pack_dashq_2(S_flat, Z_flat, Q_grouped)
                native_type = gguf.GGMLQuantizationType.DASHQ_2
            else:
                raw_bytes = pack_dashq_3(S_flat, Z_flat, Q_grouped)
                native_type = gguf.GGMLQuantizationType.DASHQ_3

            # Logical (out, in) shape -- gguf_writer keeps this as-is for DASHQ types.
            bshape = np.array([n_rows, n_cols], dtype=np.int64)

            raw_data = np.frombuffer(raw_bytes, dtype=np.uint8)
            quantized_hf_layers[name + ".weight"] = (raw_data, native_type, bshape)
        else:
            # Standard K-quant repacking path (dequant -> repack)
            W_hat = dequantize(Q, S, Z, dashq_group_size)
            raw_bytes, _, _ = pack_tensor(W_hat, bits)
            bshape = byte_shape_for_tensor(W.shape, bits)
            raw_data = np.frombuffer(raw_bytes, dtype=np.uint8)

            quantized_hf_layers[name + ".weight"] = (raw_data, tensor_type, bshape)

        # Free memory
        del hessians[name]
        del W, Q, S, Z
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # -- Write all tensors to GGUF -----------------------------------------
    print("\nWriting GGUF file...")
    from itertools import chain
    
    # Iterate over all tensors from the original HF model via convert_hf_to_gguf
    for hf_name, data_torch in chain(base_model.generate_extra_tensors(), base_model.get_tensors()):
        # skip useless ones
        if hf_name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue

        bid = None
        for part in hf_name.split("."):
            if part.isdecimal():
                bid = int(part)
                break
                
        for ggml_name, mapped_data_torch in base_model.modify_tensors(data_torch, hf_name, bid):
            if hf_name in quantized_hf_layers:
                # Write quantized tensor
                raw_data, dtype, shape = quantized_hf_layers[hf_name]
                writer.add_tensor(ggml_name, raw_data, raw_dtype=dtype, raw_shape=shape)
            else:
                # Write original (unquantised) tensor, matching llama.cpp's
                # F16/F32 choice so SSM/conv/embedding tensors keep the dtype
                # the runtime asserts on.
                data_np = nonquant_array(base_model, hf_name, ggml_name, bid, mapped_data_torch)
                writer.add_tensor(ggml_name, data_np)

    writer.write_header_to_file(path=Path(out_file))
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done -> {out_file}")


# -----------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quantise a HuggingFace model with DASH-Q and export to GGUF"
    )
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="HuggingFace model ID")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Calibration dataset name on HuggingFace")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4],
                        help="Target bit width (2=Q2_K, 3=Q3_K, 4=Q4_1)")
    parser.add_argument("--group_size", type=int, default=None,
                        help="DASH-Q group size (default: paper recommendation)")
    parser.add_argument("--iters", type=int, default=9,
                        help="Coordinate descent iterations (default: 9)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Damping factor (default: 0.5)")
    parser.add_argument("--lambda_reg", type=float, default=1e-2,
                        help="Ridge regularisation (default: 1e-2)")
    parser.add_argument("--n_samples", type=int, default=16,
                        help="Number of calibration sequences")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length for calibration")
    parser.add_argument("--out", type=str, default="model-dash-q.gguf",
                        help="Output GGUF file name")
    parser.add_argument("--native", action="store_true", default=False,
                        help="Use native DASHQ_2/DASHQ_3 block format (requires forked llama.cpp)")
    parser.add_argument("--calib_format", type=str, default="raw", choices=["raw", "chat"],
                        help="Calibration format: 'raw' (concat text) or 'chat' "
                             "(instruction/answer rendered via the tokenizer chat template)")
    parser.add_argument("--max_rows", type=int, default=128,
                        help="Cap the source calibration dataset to its first N rows. "
                             "Use 0 or negative for no cap.")

    args = parser.parse_args()

    quantize_and_export(
        model_id=args.model_id,
        dataset_name=args.dataset,
        bits=args.bits,
        group_size=args.group_size,
        iters=args.iters,
        alpha=args.alpha,
        lambda_reg=args.lambda_reg,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        out_file=args.out,
        native=args.native,
        calib_format=args.calib_format,
        max_rows=args.max_rows if args.max_rows > 0 else None,
    )
