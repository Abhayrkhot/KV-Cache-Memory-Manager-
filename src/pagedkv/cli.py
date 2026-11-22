"""Command-line interface: ``pagedkv {presets,estimate,capacity}``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from pagedkv.estimate import WEIGHT_BITS, CapacityPlan
from pagedkv.presets import load_config, preset_names, preset_params
from pagedkv.spec import KV_DTYPES, from_hf_config, human_bytes

GPUS_GIB = {
    "a100-40g": 40,
    "a100-80g": 80,
    "h100-80g": 80,
    "h200": 141,
    "l4": 24,
    "l40s": 48,
    "rtx4090": 24,
}


def _ints(s: str) -> list[int]:
    try:
        vals = [int(x) for x in s.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {s!r}") from None
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return vals


def _dtypes(s: str) -> list[str]:
    vals = [x.strip().lower() for x in s.split(",") if x.strip()]
    bad = [v for v in vals if v not in KV_DTYPES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"unknown KV dtype(s) {bad}; known: {', '.join(sorted(KV_DTYPES))}"
        )
    return vals


def _cmd_presets(args: argparse.Namespace) -> int:
    for name in preset_names():
        _, cfg = load_config(name)
        spec = from_hf_config(cfg, name)
        print(
            f"{name:20s} {human_bytes(spec.bytes_per_token()):>12s}/token  {spec.describe().split(': ', 1)[1]}"
        )
    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    name, cfg = load_config(args.model)
    spec = from_hf_config(cfg, name)
    rows = []
    for dt in args.kv_dtype:
        for seq in args.seq_len:
            for batch in args.batch:
                b = spec.seq_bytes(seq, dt) * batch
                rows.append(
                    {
                        "kv_dtype": dt,
                        "seq_len": seq,
                        "batch": batch,
                        "bytes": b,
                        "human": human_bytes(b),
                    }
                )
    if args.json:
        print(
            json.dumps(
                {
                    "model": name,
                    "describe": spec.describe(),
                    "notes": list(spec.notes),
                    "bytes_per_token": {dt: spec.bytes_per_token(dt) for dt in args.kv_dtype},
                    "rows": rows,
                },
                indent=2,
            )
        )
        return 0
    print(spec.describe())
    for note in spec.notes:
        print(f"  note: {note}")
    for dt in args.kv_dtype:
        print(f"  {dt:6s} {human_bytes(spec.bytes_per_token(dt))}/token")
    print()
    print(f"{'kv_dtype':8s} {'seq_len':>9s} {'batch':>6s} {'KV cache':>12s}")
    for r in rows:
        print(f"{r['kv_dtype']:8s} {r['seq_len']:>9d} {r['batch']:>6d} {r['human']:>12s}")
    return 0


def _cmd_capacity(args: argparse.Namespace) -> int:
    name, cfg = load_config(args.model)
    spec = from_hf_config(cfg, name)
    params = args.params or preset_params(name)
    if params is None:
        try:
            from pagedkv.hf_meta import count_params
        except ImportError:
            print("error: pass --params (or install transformers to count them)", file=sys.stderr)
            return 2
        params = count_params(cfg)
    gpu = GPUS_GIB.get(args.gpu.lower()) if args.gpu else None
    gib = args.gpu_mem_gib or gpu
    if not gib:
        print(f"error: pass --gpu ({', '.join(GPUS_GIB)}) or --gpu-mem-gib", file=sys.stderr)
        return 2
    print(f"{spec.describe()}; {params / 1e9:.2f}B params ({args.weight_dtype}), TP={args.tp}")
    print(f"GPU {gib} GiB x util {args.util} - weights/rank - {args.overhead_gib} GiB overhead")
    print()
    hdr = f"{'kv_dtype':8s} {'KV budget':>11s} {'seq_len':>8s} {'paged seqs':>11s} {'reserve max_len':>16s}"
    print(hdr)
    for dt in args.kv_dtype:
        plan = CapacityPlan.build(
            spec,
            gib,
            params,
            args.weight_dtype,
            dt,
            args.util,
            args.overhead_gib,
            args.block_size,
            args.tp,
        )
        if not plan.fits:
            need = (plan.weight_bytes + plan.overhead_bytes) / 1024**3
            print(f"{dt:8s} does not fit: weights + overhead need {need:.1f} GiB per GPU; try --tp")
            continue
        max_len = args.max_model_len or spec.max_position_embeddings or max(args.seq_len)
        for seq in args.seq_len:
            print(
                f"{dt:8s} {human_bytes(plan.kv_budget):>11s} {seq:>8d} {plan.max_seqs_paged(seq):>11d} "
                f"{plan.max_seqs_reserved(max_len):>10d} @{max_len}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pagedkv", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("presets", help="list bundled model configs").set_defaults(fn=_cmd_presets)

    e = sub.add_parser("estimate", help="KV-cache size for a model")
    e.add_argument("model", help="preset name, path to config.json, or Hugging Face repo id")
    e.add_argument("--seq-len", type=_ints, default=[4096])
    e.add_argument("--batch", type=_ints, default=[1])
    e.add_argument("--kv-dtype", type=_dtypes, default=["bf16"])
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=_cmd_estimate)

    c = sub.add_parser("capacity", help="how many sequences fit on a GPU")
    c.add_argument("model")
    c.add_argument("--gpu", help=f"one of {', '.join(GPUS_GIB)}")
    c.add_argument("--gpu-mem-gib", type=float)
    c.add_argument("--params", type=int, help="parameter count (default: measured for presets)")
    c.add_argument("--weight-dtype", default="bf16", choices=sorted(WEIGHT_BITS))
    c.add_argument("--kv-dtype", type=_dtypes, default=["bf16", "fp8"])
    c.add_argument("--seq-len", type=_ints, default=[1024, 4096, 32768])
    c.add_argument(
        "--max-model-len", type=int, help="reservation size for the reserve-up-front column"
    )
    c.add_argument("--util", type=float, default=0.9)
    c.add_argument("--overhead-gib", type=float, default=2.0)
    c.add_argument("--block-size", type=int, default=16)
    c.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    c.set_defaults(fn=_cmd_capacity)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (ValueError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
