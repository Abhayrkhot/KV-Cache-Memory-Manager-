"""Check the estimator against the KV cache Hugging Face transformers allocates.

For every bundled config this builds the model on the meta device (no weights),
runs a prefill, and compares each layer's cached tokens and bytes with
``KVSpec``. Sequence lengths include one past the largest sliding window so the
window cap is exercised. Also records exact parameter counts for capacity
planning.

    python scripts/validate_estimator.py   # writes results/estimator_validation.json
                                           # and src/pagedkv/configs/_params.json
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import transformers

from pagedkv.hf_meta import count_params, measure_hf_cache
from pagedkv.presets import load_preset, preset_names
from pagedkv.spec import AttnKind, from_hf_config, get_kv_dtype

ROOT = Path(__file__).resolve().parents[1]
BF16 = get_kv_dtype("bf16")


def main() -> int:
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    rows, params = [], {}
    failures = 0
    for name in preset_names():
        cfg = load_preset(name)
        spec = from_hf_config(cfg, name)
        params[name] = count_params(cfg)
        windows = [la.window for la in spec.layers if la.window]
        for seq_len in [300] + ([max(windows) + 300] if windows else []):
            measured = measure_hf_cache(cfg, seq_len)
            per_layer_ok = True
            for m, layer in zip(measured, spec.layers, strict=True):
                want = layer.resident_tokens(seq_len)
                # HF keeps window-1 tokens between steps; the spec counts the window
                # including the token being computed (peak residency during a step).
                if layer.kind is AttnKind.SLIDING and layer.window and seq_len >= layer.window:
                    want = layer.window - 1
                want_bytes = want * layer.bytes_per_token(BF16)
                if m.tokens != want or m.bytes != want_bytes:
                    per_layer_ok = False
            hf_bytes = sum(m.bytes for m in measured)
            rows.append(
                {
                    "model": name,
                    "hf_repo": cfg.get("_hf_repo"),
                    "description": spec.describe(),
                    "seq_len": seq_len,
                    "hf_cache_bytes": hf_bytes,
                    "estimator_bytes": spec.seq_bytes(seq_len),
                    "bytes_per_token_bf16": spec.bytes_per_token(),
                    "per_layer_match": per_layer_ok,
                    "params": params[name],
                }
            )
            failures += not per_layer_ok
            print(
                f"{'ok ' if per_layer_ok else 'BAD'} {name:20s} T={seq_len:<6d} hf={hf_bytes:>13,d}"
            )
    out = {
        "transformers_version": transformers.__version__,
        "note": "sliding layers: HF stores window-1 tokens between steps; estimator counts the window",
        "rows": rows,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "estimator_validation.json").write_text(json.dumps(out, indent=2) + "\n")
    (ROOT / "src/pagedkv/configs/_params.json").write_text(
        json.dumps(params, indent=2, sort_keys=True) + "\n"
    )
    print(f"{len(rows) - failures}/{len(rows)} configurations match layer by layer")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
