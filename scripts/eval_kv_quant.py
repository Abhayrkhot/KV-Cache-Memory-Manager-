"""Real-model checks on CPU: greedy parity with transformers and KV-quantization perplexity.

For each model:

1. greedy parity: a ragged batch of WikiText prompts is decoded by the paged
   model and, one prompt at a time, by ``transformers.generate``; tokens must match;
2. perplexity on WikiText-2 (test) for each KV storage format. Windows are scored
   with chunked prefill, so every attention read goes through the quantized cache.
   Deltas are paired per window, with a 95% interval across windows.

    python scripts/eval_kv_quant.py --models HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import statistics
import time
import warnings
from pathlib import Path

import pyarrow.parquet as pq
import torch
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagedkv.generate import PagedGenerator
from pagedkv.models.llama import PagedLlama
from pagedkv.spec import from_hf_config

ROOT = Path(__file__).resolve().parents[1]
FORMATS = ["fp32", "bf16", "fp16", "fp8", "fp8-calibrated", "fp8-kchannel", "int8", "int8-kchannel"]


def wikitext_ids(tok) -> list[int]:  # type: ignore[no-untyped-def]
    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"
    )
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return list(tok(text).input_ids)


def parity(hf, model: PagedLlama, ids: list[int], n_prompts: int, new_tokens: int) -> dict:  # type: ignore[no-untyped-def]
    prompts = [ids[i * 997 : i * 997 + 16 + 23 * i] for i in range(n_prompts)]
    gen = PagedGenerator(model, num_blocks=512, block_size=16, kv_dtype="fp32")
    ours = gen.generate(prompts, new_tokens)
    match = total = exact = 0
    for p, o in zip(prompts, ours, strict=True):
        ref = hf.generate(
            torch.tensor([p]),
            max_new_tokens=new_tokens,
            min_new_tokens=new_tokens,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )[0, len(p) :].tolist()
        match += sum(a == b for a, b in zip(o, ref, strict=True))
        total += new_tokens
        exact += o == ref
    return {
        "prompts": n_prompts,
        "new_tokens": new_tokens,
        "token_match": match / total,
        "sequences_identical": exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", default=["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B"]
    )
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--window-len", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "results/kv_quant_ppl.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    torch.manual_seed(0)

    report = {
        "device": f"CPU ({platform.machine()}, {platform.system()}), float32 compute",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dataset": "WikiText-2 raw, test split",
        "window_len": args.window_len,
        "chunk": args.chunk,
        "models": [],
    }
    for repo in args.models:
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo)
        hf = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
        model = PagedLlama.from_hf_model(hf)
        spec = from_hf_config(hf.config.to_dict(), repo)
        ids = wikitext_ids(tok)
        L = args.window_len
        n_avail = len(ids) // L
        # Calibrate fp8 scales on the last two windows; evaluate on the first `windows`.
        eval_windows = [ids[i * L : (i + 1) * L] for i in range(min(args.windows, n_avail - 2))]
        calib_windows = [ids[(n_avail - 2 + i) * L : (n_avail - 1 + i) * L] for i in range(2)]
        blocks = L // 16 + 1

        par = parity(hf, model, ids, n_prompts=8, new_tokens=32)
        print(f"{repo}: greedy token match {par['token_match']:.4f}")

        per_window: dict[str, list[float]] = {}
        for fmt in FORMATS:
            kv_dtype = fmt.split("-")[0]
            gen = PagedGenerator(model, num_blocks=blocks, block_size=16, kv_dtype=kv_dtype)
            if "-" in fmt:  # calibrated variants: per-layer fp8 scales, or per-channel key scales
                gen.cache.start_calibration()
                for w in calib_windows:
                    gen.score(w, args.chunk)
                gen.cache.finish_calibration(key_per_channel=fmt.endswith("kchannel"))
            per_window[fmt] = [gen.score(w, args.chunk).mean().item() for w in eval_windows]
            print(f"  {fmt:15s} ppl={math.exp(statistics.mean(per_window[fmt])):.4f}")

        base = per_window["fp32"]
        rows = []
        for fmt in FORMATS:
            nll = per_window[fmt]
            deltas = [a - b for a, b in zip(nll, base, strict=True)]
            se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
            kv_name = fmt.split("-")[0]
            rows.append(
                {
                    "kv_format": fmt,
                    "ppl": math.exp(statistics.mean(nll)),
                    "ppl_delta_pct": 100
                    * (math.exp(statistics.mean(nll) - statistics.mean(base)) - 1),
                    "nll_delta_mean": statistics.mean(deltas),
                    "nll_delta_ci95": [
                        statistics.mean(deltas) - 1.96 * se,
                        statistics.mean(deltas) + 1.96 * se,
                    ],
                    "kv_bytes_per_token": spec.bytes_per_token(kv_name),
                }
            )
        report["models"].append(
            {
                "model": repo,
                "kv_layout": spec.describe(),
                "eval_tokens": len(eval_windows) * L,
                "parity": par,
                "results": rows,
                "seconds": round(time.time() - t0, 1),
            }
        )
        Path(args.out).parent.mkdir(exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
