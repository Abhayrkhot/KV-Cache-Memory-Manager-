"""Paged decode attention: Triton kernel vs. FlashInfer and a gather + SDPA baseline (CUDA).

Reports latency and achieved HBM bandwidth (bytes of K/V read per second) as a
share of the GPU's peak. All implementations read the same
``[num_blocks, 2, block_size, kv_heads, head_dim]`` cache (FlashInfer's NHD
paged layout), with block tables shuffled so pages are scattered.

    python benchmarks/bench_kernel.py --out results/kernel_<gpu>.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from pagedkv.ops.triton_paged import paged_attention_decode

PEAK_GBPS = {  # dense HBM bandwidth from vendor datasheets
    "A100-SXM4-80GB": 2039,
    "A100-SXM4-40GB": 1555,
    "A100-PCIE-40GB": 1555,
    "A100 80GB PCIe": 1935,
    "H100 80GB HBM3": 3350,
    "H100 PCIe": 2000,
    "H200": 4800,
    "NVIDIA H200": 4800,
    "L4": 300,
    "NVIDIA GeForce RTX 3050 Ti Laptop GPU": 192,
}


def peak_for(name: str) -> float | None:
    for key, val in PEAK_GBPS.items():
        if key in name:
            return float(val)
    return None


def bench(fn, iters: int = 50) -> float:  # type: ignore[no-untyped-def]
    """Median milliseconds over ``iters`` CUDA-event timings after warmup."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    times.sort()
    return times[len(times) // 2]


def make_case(batch: int, ctx: int, H: int, KVH: int, D: int, BS: int, dtype: torch.dtype):  # type: ignore[no-untyped-def]
    blocks_per_seq = -(-ctx // BS)
    num_blocks = batch * blocks_per_seq + 1
    kv = torch.randn(num_blocks, 2, BS, KVH, D, device="cuda", dtype=dtype)
    perm = list(range(num_blocks))
    random.Random(0).shuffle(perm)
    tables = torch.tensor(perm[: batch * blocks_per_seq], dtype=torch.int32, device="cuda").view(
        batch, -1
    )
    seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    q = torch.randn(batch, H, D, device="cuda", dtype=dtype)
    return q, kv, tables, seq_lens


def sdpa_baseline(q, kv, tables, ctx, scale):  # type: ignore[no-untyped-def]
    """Gather pages into contiguous K/V, then SDPA: what you get without a paged kernel."""
    B, _H, D = q.shape
    KVH = kv.shape[3]
    pages = kv[tables.long()]  # [B, nblk, 2, BS, KVH, D]
    k = pages[:, :, 0].reshape(B, -1, KVH, D)[:, :ctx].transpose(1, 2)
    v = pages[:, :, 1].reshape(B, -1, KVH, D)[:, :ctx].transpose(1, 2)
    return F.scaled_dot_product_attention(
        q.unsqueeze(2), k, v, scale=scale, enable_gqa=True
    ).squeeze(2)


def run_case(
    batch: int,
    ctx: int,
    H: int,
    KVH: int,
    D: int,
    BS: int,
    dtype: torch.dtype,
    peak: float | None,
    has_fi: bool,
) -> dict:  # type: ignore[type-arg]
    scale = D**-0.5
    q, kv, tables, seq_lens = make_case(batch, ctx, H, KVH, D, BS, dtype)
    kv_bytes = batch * ctx * KVH * D * 2 * kv.element_size()
    row: dict = {"batch": batch, "context": ctx, "kv_bytes": kv_bytes}  # type: ignore[type-arg]
    ref = sdpa_baseline(q, kv, tables, ctx, scale)
    out = paged_attention_decode(q, kv, tables, seq_lens, scale, ctx)
    row["max_abs_err_vs_sdpa"] = (out.float() - ref.float()).abs().max().item()
    impls = {
        "triton": lambda: paged_attention_decode(q, kv, tables, seq_lens, scale, ctx),
        "triton_nosplit": lambda: paged_attention_decode(
            q, kv, tables, seq_lens, scale, ctx, partition_size=0
        ),
        "gather_sdpa": lambda: sdpa_baseline(q, kv, tables, ctx, scale),
    }
    if has_fi:
        try:
            import flashinfer

            ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
            nblk = tables.shape[1]
            indptr = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * nblk
            last = torch.full((batch,), ctx - (nblk - 1) * BS, dtype=torch.int32, device="cuda")
            wrapper.plan(indptr, tables.flatten(), last, H, KVH, D, BS, q_data_type=dtype)
            fi_out = wrapper.run(q, kv)
            row["max_abs_err_flashinfer_vs_sdpa"] = (
                (fi_out.float() - ref.float()).abs().max().item()
            )
            impls["flashinfer"] = lambda: wrapper.run(q, kv)
        except Exception as err:  # version skew in FlashInfer's API must not sink the run
            row["flashinfer_error"] = repr(err)
    for impl, fn in impls.items():
        ms = bench(fn)
        gbps = kv_bytes / (ms / 1e3) / 1e9
        row[impl] = {"ms": ms, "gbps": gbps, "pct_peak": 100 * gbps / peak if peak else None}
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--batches", default="1,8,32,128")
    ap.add_argument("--contexts", default="512,2048,8192,32768")
    ap.add_argument("--max-kv-gib", type=float, default=16.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = torch.bfloat16
    name = torch.cuda.get_device_name()
    peak = peak_for(name)
    H, KVH, D, BS = args.heads, args.kv_heads, args.head_dim, args.block_size
    try:
        import flashinfer  # noqa: F401

        has_fi = True
    except ImportError:
        has_fi = False
    rows = []
    for batch in map(int, args.batches.split(",")):
        for ctx in map(int, args.contexts.split(",")):
            kv_bytes = batch * ctx * KVH * D * 2 * 2
            if kv_bytes > args.max_kv_gib * 1024**3:
                continue
            row = run_case(batch, ctx, H, KVH, D, BS, dtype, peak, has_fi)
            rows.append(row)
            msg = "  ".join(
                f"{k}={v['ms']:.3f}ms/{v['gbps']:.0f}GB/s"
                for k, v in row.items()
                if isinstance(v, dict)
            )
            print(f"B={batch:4d} ctx={ctx:6d}  {msg}")
            torch.cuda.empty_cache()
    result = {
        "gpu": name,
        "peak_gbps": peak,
        "torch": torch.__version__,
        "heads": H,
        "kv_heads": KVH,
        "head_dim": D,
        "block_size": BS,
        "dtype": "bf16",
        "flashinfer": has_fi,
        "rows": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
