"""Regenerate the results sections of README.md from results/*.json.

Each section lives between ``<!-- BEGIN:name -->`` and ``<!-- END:name -->``;
everything else in the README is hand-written.

    python scripts/render_readme.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LABELS = {
    "paged-16": "Paged, 16-token blocks",
    "reserve-max": "Reserve `max_model_len` (2,048)",
    "reserve-pow2": "Reserve next power of two",
    "reserve-oracle": "Reserve exact final length (oracle)",
}


def _load(name: str) -> dict | None:  # type: ignore[type-arg]
    p = ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def _kib(b: float) -> str:
    return f"{b / 1024:,.2f} KiB" if b % 1024 else f"{b / 1024:,.0f} KiB"


def allocator() -> str:
    sim = _load("allocator_sim.json")
    if sim is None:
        return "_Run `python benchmarks/bench_allocator.py`._"
    w = sim["workload"]
    out = [
        f"{sim['num_requests']:,} requests sampled from ShareGPT (mean prompt {w['prompt_mean']:.0f} "
        f"tokens, mean output {w['output_mean']:.0f}; Llama 2 tokenizer; vLLM's benchmark filters), all "
        f"queued at t=0, `max_num_seqs={sim['max_num_seqs']}`. The pool is the KV budget left on the GPU "
        "after bf16 weights, 90% memory utilization and 2 GiB of overhead (`pagedkv capacity`). "
        '"Batch" is the mean number of running sequences on steps where memory was the binding limit; '
        '"steps" is forward passes to drain the queue.',
        "",
    ]
    for _key, sc in sim["scenarios"].items():
        out += [f"**{sc['description']}** ({sc['pool_tokens']:,}-token pool)", ""]
        out += [
            "| Allocator | Batch | Steps to drain | Live K/V | Reserved | Internal frag. | External frag. | Preemptions |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        base = sc["allocators"]["paged-16"]["steps"]
        for name, r in sc["allocators"].items():
            b = r["breakdown"]
            batch = r["mean_batch_saturated"] if r["saturated_steps"] else r["mean_batch"]
            star = "" if r["saturated_steps"] else "*"
            out.append(
                f"| {LABELS[name]} | {batch:.1f}{star} | {r['steps']:,} ({r['steps'] / base:.2f}x) | "
                + (
                    " | ".join(
                        f"{100 * b[k]:.1f}%"
                        for k in ("token_states", "reserved", "internal_frag", "external_frag")
                    )
                    if r["saturated_steps"]
                    else "- | - | - | -"
                )
                + f" | {r['preemptions']:,} |"
            )
        if any(not r["saturated_steps"] for r in sc["allocators"].values()):
            out.append("")
            out.append(
                "\\* Memory never became the limit (`max_num_seqs` did), so this is the mean batch over all steps."
            )
        out.append("")
    sw = sim["block_size_sweep"]
    out += [
        f"**Block size** ({sw['scenario']}): smaller blocks waste less of the last block but mean bigger block tables.",
        "",
        "| Block size | 1 | 8 | 16 | 32 | 64 | 128 | 256 |",
        "|---|" + "---:|" * 7,
    ]
    rows = sw["rows"]
    out.append(
        "| Internal fragmentation | "
        + " | ".join(f"{100 * rows[k]['internal_frag']:.1f}%" for k in rows)
        + " |"
    )
    out.append(
        "| Batch | " + " | ".join(f"{rows[k]['mean_batch_saturated']:.1f}" for k in rows) + " |"
    )
    wm = sim.get("watermark_sweep")
    if wm:
        r = wm["rows"]
        out += [
            "",
            f"**Admission watermark** ({wm['scenario']}): holding a few percent of blocks back from new "
            "requests leaves room for running ones to grow. Without it the paged allocator over-admits and "
            f"thrashes (preempt, recompute, repeat); the tables above use {100 * sim['paged_watermark']:.0f}%.",
            "",
            "| Watermark | " + " | ".join(f"{100 * float(k):g}%" for k in r) + " |",
            "|---|" + "---:|" * len(r),
            "| Batch | "
            + " | ".join(f"{v['mean_batch_saturated']:.1f}" for v in r.values())
            + " |",
            "| Preemptions | " + " | ".join(f"{v['preemptions']:,}" for v in r.values()) + " |",
            "| Recomputed tokens | "
            + " | ".join(f"{v['recomputed_tokens']:,}" for v in r.values())
            + " |",
        ]
    ps = sim["parallel_sampling"]["allocators"]
    pc = sim["prefix_caching"]["allocators"]
    on, off = pc["paged-16"], pc["paged-16-nocache"]
    out += [
        "",
        f"**Parallel sampling** (n=4 per prompt, {sim['parallel_sampling']['scenario']}): copy-on-write lets the four samples share "
        f"prompt blocks, so the paged pool runs a batch of {ps['paged-16']['mean_batch_saturated']:.0f} vs. "
        f"{ps['reserve-oracle']['mean_batch_saturated']:.0f} for exact-length reservation and drains in "
        f"{ps['paged-16']['steps']:,} steps vs. {ps['reserve-oracle']['steps']:,}.",
        "",
        f"**Prefix caching** ({sim['prefix_caching']['num_system_prompts']} shared {sim['prefix_caching']['system_prompt_tokens']}-token "
        f"system prompts): {100 * on['prefill_tokens_skipped'] / on['prefill_tokens']:.0f}% of prefill tokens are served from the cache, "
        f"prefill compute drops from {off['prefill_tokens']:,} to {on['prefill_tokens'] - on['prefill_tokens_skipped']:,} tokens, and "
        f"shared blocks raise the batch from {off['mean_batch_saturated']:.0f} to {on['mean_batch_saturated']:.0f}.",
    ]
    return "\n".join(out)


def kvquant() -> str:
    r = _load("kv_quant_ppl.json")
    if r is None:
        return "_Run `python scripts/eval_kv_quant.py`._"
    names = {
        "fp32": "FP32 (reference)",
        "bf16": "BF16",
        "fp16": "FP16",
        "fp8": "FP8 e4m3, scale 1.0",
        "fp8-calibrated": "FP8, per-layer calibrated scale",
        "fp8-kchannel": "FP8, per-channel key scales",
        "int8": "INT8, per-token scales",
        "int8-kchannel": "INT8, per-channel key scales",
    }
    models = r["models"]
    out = [
        f"{r['dataset']}, {models[0]['eval_tokens']:,} tokens per model ({r['device']}). "
        f"Scored with {r['chunk']}-token chunked prefill, so every attention read goes through the "
        "quantized paged cache. Δ is relative to an FP32 cache; the interval is a 95% CI over windows "
        "of the paired per-window difference in mean NLL.",
        "",
    ]
    hdr = (
        "| KV format | Bytes/token | "
        + " | ".join(f"{m['model'].split('/')[-1]} PPL | Δ" for m in models)
        + " |"
    )
    out += [hdr, "|---|---:|" + "---:|---:|" * len(models)]
    for i, row in enumerate(models[0]["results"]):
        cells = [names.get(row["kv_format"], row["kv_format"])]
        cells.append(" / ".join(_kib(m["results"][i]["kv_bytes_per_token"]) for m in models))
        for m in models:
            x = m["results"][i]
            ci = x["nll_delta_ci95"]
            delta = (
                "-"
                if row["kv_format"] == "fp32"
                else f"{x['ppl_delta_pct']:+.2f}% <sub>[{ci[0]:+.4f}, {ci[1]:+.4f}]</sub>"
            )
            cells += [f"{x['ppl']:.3f}", delta]
        out.append("| " + " | ".join(cells) + " |")
    par = ", ".join(
        f"{m['model'].split('/')[-1]} {100 * m['parity']['token_match']:.0f}% "
        f"({m['parity']['sequences_identical']}/{m['parity']['prompts']} sequences)"
        for m in models
    )
    out += [
        "",
        f"Greedy decoding, paged model vs. `transformers.generate`, {models[0]['parity']['new_tokens']} new tokens "
        f"per prompt, ragged batch of {models[0]['parity']['prompts']}: token match {par}.",
    ]
    return "\n".join(out)


def estimator() -> str:
    r = _load("estimator_validation.json")
    if r is None:
        return "_Run `python scripts/validate_estimator.py`._"
    rows = r["rows"]
    ok = sum(x["per_layer_match"] for x in rows)
    out = [
        f"{ok}/{len(rows)} model x length checks match transformers {r['transformers_version']} layer by layer "
        "(cached tokens and bytes per layer). Lengths past the largest window exercise sliding-window caps.",
        "",
        "| Model | Layers | KV/token (bf16) | Params | Checked at | Match |",
        "|---|---|---:|---:|---|:-:|",
    ]
    seen: dict[str, list] = {}  # type: ignore[type-arg]
    for x in rows:
        seen.setdefault(x["model"], []).append(x)
    for name, xs in seen.items():
        d = xs[0]["description"].split(": ", 1)[1]
        out.append(
            f"| `{name}` | {d} | {_kib(xs[0]['bytes_per_token_bf16'])} | {xs[0]['params'] / 1e9:,.2f}B | "
            f"{', '.join(str(x['seq_len']) for x in xs)} | {'yes' if all(x['per_layer_match'] for x in xs) else 'NO'} |"
        )
    return "\n".join(out)


def gpu() -> str:
    d = ROOT / "results" / "gpu"
    runs = sorted(d.glob("*/")) if d.exists() else []
    if not runs:
        return (
            "_Not yet measured._ The Triton kernel passes its tests under Triton's interpreter "
            "(`scripts/test_triton_interpreter.sh`), but it has not been timed on a GPU. "
            "`sbatch slurm/gpu.sbatch` runs the GPU test suite and both GPU benchmarks, then re-renders this section."
        )
    out = []
    for run in runs:
        k = (
            json.loads((run / "kernel.json").read_text())
            if (run / "kernel.json").exists()
            else None
        )
        if k:
            impls = [
                i
                for i in ("triton", "triton_nosplit", "flashinfer", "gather_sdpa")
                if i in k["rows"][0]
            ]
            out += [
                f"**Decode attention on {k['gpu']}** (peak {k['peak_gbps']} GB/s; {k['heads']} query / "
                f"{k['kv_heads']} KV heads, head_dim {k['head_dim']}, bf16, {k['block_size']}-token pages). "
                "Cells: latency, and K/V bytes read per second as % of peak.",
                "",
                "| Batch | Context | " + " | ".join(impls) + " |",
                "|---:|---:|" + "---:|" * len(impls),
            ]
            for row in k["rows"]:
                cells = []
                for i in impls:
                    v = row[i]
                    pct = f" ({v['pct_peak']:.0f}%)" if v.get("pct_peak") else ""
                    cells.append(f"{v['ms'] * 1000:.0f} µs{pct}")
                out.append(f"| {row['batch']} | {row['context']:,} | " + " | ".join(cells) + " |")
            out.append("")
        for f in sorted(run.glob("serving_*.json")):
            s = json.loads(f.read_text())
            p, r = s["runs"]
            out += [
                f"**Continuous batching, {s['model']} on {s['gpu']}** ({s['kv_gib']} GiB {s['kv_dtype']} KV pool, "
                f"{s['num_requests']} ShareGPT requests): paged {p['output_tok_per_s']:,.0f} tok/s at mean batch "
                f"{p['mean_batch']:.1f} vs. reserve-max {r['output_tok_per_s']:,.0f} tok/s at {r['mean_batch']:.1f} "
                f"(**{s['speedup']:.2f}x**; {p['preemptions']} preemptions).",
                "",
            ]
    return "\n".join(out).rstrip()


def main() -> None:
    text = README.read_text()
    for name, fn in {
        "allocator": allocator,
        "kvquant": kvquant,
        "estimator": estimator,
        "gpu": gpu,
    }.items():
        pat = re.compile(rf"(<!-- BEGIN:{name} -->\n)(?:.*?\n)?(<!-- END:{name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README is missing the {name} markers")
        body = fn()
        text = pat.sub(lambda m, body=body: m.group(1) + body + "\n" + m.group(2), text)
    README.write_text(text)
    print("README.md updated")


if __name__ == "__main__":
    main()
