"""
Compare DASH-Q variants against stock-llama.cpp baselines at matched bitwidths.

For each `--bits` and each path (stock RTN, DASH-Q repack, DASH-Q native), the
script:
  1. produces the GGUF file (skips if it already exists),
  2. records the on-disk size,
  3. optionally runs `llama-perplexity` against a wikitext-style file,
  4. tabulates everything to stdout (and a markdown file).

Outputs land in `--out_dir`. Each variant is one .gguf file named
`<bits>-<variant>.gguf`.

Typical use:
    uv run python compare.py --model_id Qwen/Qwen2.5-0.5B \
        --bits 2 3 4 \
        --ppl_file llama.cpp/build/wikitext-2-raw/wiki.test.raw

Notes:
- The "rtn" baseline runs `llama-quantize` against an F16 conversion. It is the
  same target format as the "dashq-repack" variant, isolating DASH-Q's value-add.
- "dashq-native" is only available for bits 2 and 3; bits 4 has no native type.
- F16 reference is generated once and reused as the perplexity baseline.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
LLAMA_DIR = REPO / "llama.cpp"
LLAMA_BIN = LLAMA_DIR / "build" / "bin"
CONVERT_PY = LLAMA_DIR / "convert_hf_to_gguf.py"

# bit -> stock K-quant name for `llama-quantize`
RTN_TYPE = {2: "Q2_K", 3: "Q3_K", 4: "Q4_1"}


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kw)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def ensure_f16(model_id: str, out_dir: Path) -> Path:
    """Convert the HF model to F16 GGUF (cached)."""
    f16 = out_dir / "f16.gguf"
    if f16.exists():
        return f16
    from huggingface_hub import snapshot_download
    local = Path(snapshot_download(model_id, allow_patterns=["*.json", "*.model", "*.safetensors"]))
    run(["uv", "run", "python", str(CONVERT_PY), str(local),
         "--outfile", str(f16), "--outtype", "f16"])
    return f16


def run_rtn(f16: Path, bits: int, out_dir: Path) -> Path:
    out = out_dir / f"{bits}-rtn.gguf"
    if out.exists():
        return out
    quantize_bin = LLAMA_BIN / "llama-quantize"
    run([str(quantize_bin), str(f16), str(out), RTN_TYPE[bits]])
    return out


def run_dashq(model_id: str, bits: int, out_dir: Path,
              native: bool, n_samples: int, seq_len: int,
              calib_format: str, max_rows: int) -> Path:
    tag = "dashq-native" if native else "dashq-repack"
    out = out_dir / f"{bits}-{tag}.gguf"
    if out.exists():
        return out
    cmd = ["uv", "run", "python", str(REPO / "quantize_hf.py"),
           "--model_id", model_id,
           "--bits", str(bits),
           "--n_samples", str(n_samples),
           "--seq_len", str(seq_len),
           "--calib_format", calib_format,
           "--max_rows", str(max_rows),
           "--out", str(out)]
    if native:
        cmd.append("--native")
    run(cmd)
    return out


def measure_ppl(gguf: Path, ppl_file: Path, ctx: int) -> dict:
    """Run llama-perplexity. Returns {'ppl': float, 'time_s': float} or {}."""
    bin_ = LLAMA_BIN / "llama-perplexity"
    if not bin_.exists():
        return {}
    t0 = time.time()
    proc = subprocess.run(
        [str(bin_), "-m", str(gguf), "-f", str(ppl_file), "-c", str(ctx)],
        capture_output=True, text=True, check=False,
    )
    dt = time.time() - t0
    # llama-perplexity prints "Final estimate: PPL = <value> +/- <err>"
    ppl = None
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if "Final estimate" in line and "PPL" in line:
            try:
                ppl = float(line.split("PPL =")[1].split()[0])
            except (IndexError, ValueError):
                pass
    if ppl is None:
        print(f"  ! could not parse PPL from llama-perplexity output for {gguf.name}")
    return {"ppl": ppl, "time_s": dt}


def variants_for(bits: int) -> list[tuple[str, str]]:
    """Return [(variant_tag, label), ...] for a given bit width."""
    out = [("rtn", f"{RTN_TYPE[bits]} (RTN)"),
           ("dashq-repack", f"{RTN_TYPE[bits]} (DASH-Q repack)")]
    if bits in (2, 3):
        out.append(("dashq-native", f"DASHQ_{bits} (native)"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--out_dir", default="compare_out")
    p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--ppl_file", default=None,
                   help="wiki.test.raw or similar; if omitted, only file sizes are reported")
    p.add_argument("--ctx", type=int, default=512, help="ctx size for llama-perplexity")
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--calib_format", default="raw", choices=["raw", "chat"])
    p.add_argument("--max_rows", type=int, default=128)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ppl_path = Path(args.ppl_file).resolve() if args.ppl_file else None

    print(f"== converting {args.model_id} to F16")
    f16 = ensure_f16(args.model_id, out_dir)

    rows = []
    f16_ppl = measure_ppl(f16, ppl_path, args.ctx) if ppl_path else {}
    rows.append({"bits": "16", "variant": "F16", "path": str(f16),
                 "size_mb": file_size_mb(f16), **f16_ppl})

    for bits in args.bits:
        for tag, label in variants_for(bits):
            print(f"\n== bits={bits} variant={label}")
            try:
                if tag == "rtn":
                    g = run_rtn(f16, bits, out_dir)
                else:
                    g = run_dashq(args.model_id, bits, out_dir,
                                  native=(tag == "dashq-native"),
                                  n_samples=args.n_samples, seq_len=args.seq_len,
                                  calib_format=args.calib_format,
                                  max_rows=args.max_rows)
            except subprocess.CalledProcessError as e:
                print(f"  ! build failed: {e}")
                rows.append({"bits": bits, "variant": label, "path": "FAILED",
                             "size_mb": None, "ppl": None})
                continue

            row = {"bits": bits, "variant": label, "path": str(g),
                   "size_mb": file_size_mb(g)}
            if ppl_path:
                row.update(measure_ppl(g, ppl_path, args.ctx))
            rows.append(row)

    # Markdown table
    md = ["| bits | variant | size (MB) | PPL | PPL vs F16 |",
          "|------|---------|-----------|-----|------------|"]
    base_ppl = f16_ppl.get("ppl") if f16_ppl else None
    for r in rows:
        size = f"{r['size_mb']:.1f}" if r.get("size_mb") is not None else "-"
        ppl = r.get("ppl")
        ppl_s = f"{ppl:.4f}" if ppl is not None else "-"
        delta = f"{(ppl - base_ppl):+.4f}" if (ppl is not None and base_ppl) else "-"
        md.append(f"| {r['bits']} | {r['variant']} | {size} | {ppl_s} | {delta} |")
    table = "\n".join(md)
    print("\n" + table)

    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    (out_dir / "results.md").write_text(table + "\n")
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
