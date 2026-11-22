"""End-to-end continuous batching on a GPU: paged admission vs. reserve-max admission.

Same model, same kernels, same KV pool; the only difference is the admission
rule. ``paged`` admits a request when its prompt fits and grows it block by block
(preempting and recomputing the newest request if memory runs out, as vLLM
does). ``reserve`` admits a request only if ``max_model_len`` tokens can be set
aside for it, as systems without paging must. Request lengths come from
ShareGPT; prompt contents are random token ids (they do not affect speed) and
every request decodes exactly its recorded output length.

The KV pool is capped with ``--kv-gib`` to put a small model in the
memory-bound regime that a larger model would be in on the same GPU.

    python benchmarks/bench_serving.py --model Qwen/Qwen2.5-1.5B-Instruct --kv-gib 2 \\
        --out results/serving_<gpu>.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import torch

from pagedkv.generate import PagedGenerator
from pagedkv.models.llama import PagedLlama
from pagedkv.ops.attention import BACKENDS
from pagedkv.sim.workload import load_lengths

ROOT = Path(__file__).resolve().parents[1]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class Req:
    idx: int
    prompt: list[int]
    output_len: int
    rid: int = -1
    generated: list[int] = field(default_factory=list)
    reserved: int = 0


def run(
    gen: PagedGenerator, reqs: list[Req], mode: str, max_num_seqs: int, max_model_len: int
) -> dict:
    bs = gen.kv.block_size
    num_blocks = gen.kv.pool.num_blocks
    reserve_blocks = -(-max_model_len // bs)
    waiting: deque[Req] = deque(reqs)
    running: list[Req] = []
    reserved = 0
    steps = preemptions = out_tokens = prefill_tokens = 0
    batch_sum = 0
    _sync()
    t0 = time.perf_counter()
    while waiting or running:
        steps += 1
        # 1. one more slot for every decoding request, preempting the newest if needed
        decode: list[Req] = []
        i = 0
        while i < len(running):
            r = running[i]
            if gen.kv.can_allocate(r.rid, 1):
                gen.append(r.rid, r.generated[-1])
                decode.append(r)
                i += 1
                continue
            victim = running.pop()
            gen.free(victim.rid)
            preemptions += 1
            if victim in decode:
                decode.remove(victim)
            victim.prompt = victim.prompt + victim.generated  # recompute prompt + output so far
            victim.output_len -= len(victim.generated)
            victim.generated = []
            waiting.appendleft(victim)
        # 2. admission
        prefill: list[tuple[Req, int]] = []
        while waiting and len(running) < max_num_seqs:
            r = waiting[0]
            if mode == "reserve" and reserved + reserve_blocks > num_blocks:
                break
            hit = gen.kv.find_prefix(r.prompt)
            if not gen.kv.can_allocate(None, len(r.prompt) - hit.num_tokens, hit, admit=True):
                break
            waiting.popleft()
            r.rid, cached = gen.new_request(r.prompt)
            if mode == "reserve":
                r.reserved = reserve_blocks
                reserved += reserve_blocks
            prefill.append((r, len(r.prompt) - cached))
            prefill_tokens += len(r.prompt) - cached
            running.append(r)
        if not running:
            raise RuntimeError("request does not fit in an empty KV pool")
        # 3. one forward pass for prefills and decodes together
        order = [r for r, _ in prefill] + decode
        qlens = [q for _, q in prefill] + [1] * len(decode)
        logits = gen.run(gen.build_step([r.rid for r in order], qlens))
        nxt = logits.argmax(-1).tolist()
        batch_sum += len(order)
        # 4. emit tokens, retire finished requests
        for r, tok in zip(order, nxt, strict=True):
            gen.kv.commit(r.rid, gen.tokens[r.rid])
            r.generated.append(tok)
            out_tokens += 1
        still = []
        for r in running:
            if len(r.generated) >= r.output_len:
                gen.free(r.rid)
                reserved -= r.reserved
            else:
                still.append(r)
        running = still
    _sync()
    wall = time.perf_counter() - t0
    return {
        "mode": mode,
        "seconds": wall,
        "steps": steps,
        "output_tokens": out_tokens,
        "output_tok_per_s": out_tokens / wall,
        "mean_batch": batch_sum / steps,
        "preemptions": preemptions,
        "prefill_tokens": prefill_tokens,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--kv-gib", type=float, default=2.0)
    ap.add_argument("--num", type=int, default=600)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--kv-dtype", default="bf16")
    ap.add_argument("--backend", default="triton", choices=sorted(BACKENDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = PagedLlama.from_pretrained(args.model, dtype=dtype, device=args.device)
    cfg = model.config
    elem = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}[args.kv_dtype]
    bytes_per_block = (
        2 * cfg.num_hidden_layers * model.num_kv_heads * model.head_dim * args.block_size * elem
    )
    num_blocks = int(args.kv_gib * 1024**3 // bytes_per_block)
    pairs = load_lengths(ROOT / "benchmarks/workloads/sharegpt_llama2.json")
    rng = random.Random(0)
    sample = rng.sample(pairs, args.num)

    def requests() -> list[Req]:
        r2 = random.Random(1)
        return [
            Req(i, [r2.randrange(10, cfg.vocab_size - 10) for _ in range(p)], o)
            for i, (p, o) in enumerate(sample)
        ]

    device_name = torch.cuda.get_device_name() if args.device == "cuda" else args.device
    results = {
        "gpu": device_name,
        "model": args.model,
        "kv_gib": args.kv_gib,
        "num_blocks": num_blocks,
        "block_size": args.block_size,
        "kv_dtype": args.kv_dtype,
        "backend": args.backend,
        "num_requests": args.num,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "torch": torch.__version__,
        "runs": [],
    }
    # warm up kernels (Triton JIT) outside the timed runs
    warm = PagedGenerator(
        model, 256, args.block_size, args.kv_dtype, attn_fn=BACKENDS[args.backend]
    )
    run(warm, requests()[:8], "paged", 8, args.max_model_len)
    del warm
    for mode in ("paged", "reserve"):
        gen = PagedGenerator(
            model,
            num_blocks,
            args.block_size,
            args.kv_dtype,
            enable_prefix_caching=False,
            attn_fn=BACKENDS[args.backend],
        )
        res = run(gen, requests(), mode, args.max_num_seqs, args.max_model_len)
        print(
            f"{mode:8s} {res['output_tok_per_s']:8.1f} tok/s  batch={res['mean_batch']:.1f} "
            f"steps={res['steps']} preempt={res['preemptions']} ({res['seconds']:.1f}s)"
        )
        results["runs"].append(res)
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    p, r = results["runs"]
    results["speedup"] = p["output_tok_per_s"] / r["output_tok_per_s"]
    print(f"paged / reserve throughput: {results['speedup']:.2f}x")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
