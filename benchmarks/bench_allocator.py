"""Allocator simulation on ShareGPT request lengths (CPU only, deterministic).

Scenarios are sized from real deployments with ``CapacityPlan`` (weights, 90%
memory utilization, 2 GiB overhead), so the KV pool is the number of tokens that
would actually fit. Writes ``results/allocator_sim.json``.

    python benchmarks/bench_allocator.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from pagedkv.estimate import CapacityPlan
from pagedkv.presets import load_spec, preset_params
from pagedkv.sim import (
    PagedAllocator,
    ReservedAllocator,
    from_lengths,
    load_lengths,
    shared_prefix,
    simulate,
)

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "benchmarks/workloads/sharegpt_llama2.json"
MAX_MODEL_LEN = 2048  # ShareGPT pairs are filtered to prompt + output <= 2048
WATERMARK = 0.05  # share of blocks held back from admission so running requests can grow

SCENARIOS = {
    # name: (preset, GPU GiB, description)
    "llama-2-13b@a100-40g": (
        "llama-2-13b",
        40,
        "Llama-2-13B (MHA) on A100-40GB: the vLLM paper's memory regime",
    ),
    "llama-3.1-8b@l4-24g": ("llama-3.1-8b", 24, "Llama-3.1-8B (GQA) on L4-24GB"),
    "llama-3.1-8b@a100-80g": ("llama-3.1-8b", 80, "Llama-3.1-8B on A100-80GB: memory rarely binds"),
}


def pool_tokens(preset: str, gpu_gib: float) -> int:
    spec = load_spec(preset)
    plan = CapacityPlan.build(spec, gpu_gib, preset_params(preset) or 0)
    return int(plan.kv_budget // spec.bytes_per_token())


def run_scenario(
    pairs, preset: str, gpu: float, num: int, max_num_seqs: int, block_size: int
) -> dict:  # type: ignore[no-untyped-def]
    tokens = pool_tokens(preset, gpu)
    out = {"pool_tokens": tokens, "allocators": {}}
    allocs = [
        PagedAllocator(tokens // block_size, block_size, watermark=WATERMARK),
        ReservedAllocator(tokens, "max", MAX_MODEL_LEN),
        ReservedAllocator(tokens, "pow2", MAX_MODEL_LEN),
        ReservedAllocator(tokens, "oracle", MAX_MODEL_LEN),
    ]
    for alloc in allocs:
        t = time.time()
        r = simulate(from_lengths(pairs, num, seed=0), alloc, max_num_seqs=max_num_seqs)
        d = asdict(r)
        d.pop("timeline")
        d["output_tokens_per_step"] = r.output_tokens_per_step
        out["allocators"][alloc.name] = d
        print(
            f"  {alloc.name:15s} batch={r.mean_batch_saturated:7.1f} steps={r.steps:7d} "
            f"token_states={r.breakdown['token_states']:.3f} preempt={r.preemptions} ({time.time() - t:.1f}s)"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=2000)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "results/allocator_sim.json"))
    args = ap.parse_args()
    pairs = load_lengths(WORKLOAD)
    meta = json.loads(WORKLOAD.read_text())["meta"]
    results: dict = {
        "workload": meta,
        "num_requests": args.num,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": MAX_MODEL_LEN,
        "paged_watermark": WATERMARK,
        "scenarios": {},
    }

    for name, (preset, gpu, desc) in SCENARIOS.items():
        print(name)
        res = run_scenario(pairs, preset, gpu, args.num, args.max_num_seqs, 16)
        res["description"] = desc
        results["scenarios"][name] = res

    # Block-size sweep: internal fragmentation vs. block size (paged only).
    print("block size sweep (llama-2-13b@a100-40g)")
    tokens = pool_tokens("llama-2-13b", 40)
    sweep = {}
    for bs in (1, 8, 16, 32, 64, 128, 256):
        r = simulate(
            from_lengths(pairs, args.num, seed=0),
            PagedAllocator(tokens // bs, bs, watermark=WATERMARK),
            max_num_seqs=args.max_num_seqs,
        )
        sweep[bs] = {
            "mean_batch_saturated": r.mean_batch_saturated,
            "steps": r.steps,
            "internal_frag": r.breakdown["internal_frag"],
            "preemptions": r.preemptions,
        }
        print(
            f"  bs={bs:3d} batch={r.mean_batch_saturated:6.1f} internal={r.breakdown['internal_frag']:.4f}"
        )
    results["block_size_sweep"] = {
        "scenario": "llama-2-13b@a100-40g",
        "pool_tokens": tokens,
        "rows": sweep,
    }

    # Admission watermark: memory held back from new requests vs. preemption churn.
    print("watermark sweep (llama-2-13b@a100-40g)")
    tokens = pool_tokens("llama-2-13b", 40)
    wm_rows = {}
    for wm in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2):
        r = simulate(
            from_lengths(pairs, args.num, seed=0),
            PagedAllocator(tokens // 16, 16, watermark=wm),
            max_num_seqs=args.max_num_seqs,
        )
        wm_rows[str(wm)] = {
            "mean_batch_saturated": r.mean_batch_saturated,
            "steps": r.steps,
            "preemptions": r.preemptions,
            "recomputed_tokens": r.recomputed_tokens,
            "token_states": r.breakdown["token_states"],
        }
        print(f"  wm={wm:<5} batch={r.mean_batch_saturated:6.1f} preempt={r.preemptions}")
    results["watermark_sweep"] = {"scenario": "llama-2-13b@a100-40g", "rows": wm_rows}

    # Parallel sampling: n=4 samples per prompt share prompt blocks via copy-on-write.
    print("parallel sampling n=4 (llama-3.1-8b@l4-24g)")
    tokens = pool_tokens("llama-3.1-8b", 24)
    ps = {}
    for alloc in (
        PagedAllocator(tokens // 16, 16, watermark=WATERMARK),
        ReservedAllocator(tokens, "oracle", MAX_MODEL_LEN),
    ):
        r = simulate(
            from_lengths(pairs, args.num // 4, seed=0, n=4), alloc, max_num_seqs=args.max_num_seqs
        )
        ps[alloc.name] = {
            "mean_batch_saturated": r.mean_batch_saturated,
            "steps": r.steps,
            "token_states": r.breakdown["token_states"],
            "preemptions": r.preemptions,
        }
        print(f"  {alloc.name:15s} batch={r.mean_batch_saturated:6.1f} steps={r.steps}")
    results["parallel_sampling"] = {
        "scenario": "llama-3.1-8b@l4-24g",
        "n": 4,
        "pool_tokens": tokens,
        "allocators": ps,
    }

    # Shared system prompt: prefix caching on vs off.
    print("shared 1024-token system prompt (llama-3.1-8b@l4-24g)")
    user = [(min(p, 512), o) for p, o in pairs]
    pc = {}
    for caching in (True, False):
        alloc = PagedAllocator(tokens // 16, 16, prefix_caching=caching, watermark=WATERMARK)
        r = simulate(
            shared_prefix(args.num, 1024, user, num_system_prompts=4, seed=0),
            alloc,
            max_num_seqs=args.max_num_seqs,
        )
        pc[alloc.name] = {
            "mean_batch_saturated": r.mean_batch_saturated,
            "steps": r.steps,
            "prefill_tokens": r.prefill_tokens,
            "prefill_tokens_skipped": r.prefill_tokens_skipped,
            "preemptions": r.preemptions,
        }
        print(
            f"  {alloc.name:20s} batch={r.mean_batch_saturated:6.1f} prefill={r.prefill_tokens} "
            f"skipped={r.prefill_tokens_skipped}"
        )
    results["prefix_caching"] = {
        "scenario": "llama-3.1-8b@l4-24g",
        "system_prompt_tokens": 1024,
        "num_system_prompts": 4,
        "allocators": pc,
    }

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
