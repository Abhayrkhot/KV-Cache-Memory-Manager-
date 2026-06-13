"""KV cache memory calculator.

This module provides utilities to estimate the GPU memory required to
store the key‑value (KV) cache for transformer decoders during
inference.  The KV cache stores the attention key and value tensors for
all previously generated tokens.  Its size grows linearly with both
sequence length and batch size and can easily dominate the memory
footprint for long context windows【555339060125421†L115-L132】.

The memory required for the KV cache can be computed with the formula

    bytes = 2 * layers * heads * head_dim * seq_len * batch_size * bytes_per_element

The factor of two accounts for storing both the key and value
tensors【459119712734931†L158-L162】.  You can adjust the number of bytes per
element to model different numeric precisions (e.g. 2 bytes for
FP16, 1 byte for INT8, 0.5 bytes for INT4).  The number of heads and
head dimension determine the hidden size of the model.

The command‑line interface supports computing the KV cache memory for
common LLaMA model sizes or custom configurations.  You can vary
sequence lengths and batch sizes to explore how the cache scales.  See
`python kv_cache.py --help` for details.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable, List, Dict


@dataclass
class ModelSpec:
    """Simple container for model architecture parameters."""

    layers: int
    heads: int
    head_dim: int

    @property
    def hidden_size(self) -> int:
        """Return the model hidden size (heads * head_dim)."""
        return self.heads * self.head_dim


# Predefined specifications for a few open models.  These values are
# approximate; consult original model definitions for exact numbers.
MODEL_SPECS: Dict[str, ModelSpec] = {
    # LLaMA v2/3 7B and 13B variants share similar head counts but
    # different layer counts.  See open source docs for details.
    "llama7b": ModelSpec(layers=32, heads=32, head_dim=128),
    "llama13b": ModelSpec(layers=40, heads=40, head_dim=128),
    "llama70b": ModelSpec(layers=80, heads=64, head_dim=128),
    # GPT‑3 175B (approximate) uses 96 layers and 96 heads with head
    # dimension 128.  Hidden size is 12288 (96*128).
    "gpt3": ModelSpec(layers=96, heads=96, head_dim=128),
}

# Bytes per element for different numeric types.  Values less than 1
# represent fractional bytes, e.g. int4/FP4 uses half a byte per element.
BYTES_PER_DTYPE: Dict[str, float] = {
    "fp16": 2.0,
    "float16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "uint8": 1.0,
    "int4": 0.5,
    "fp4": 0.5,
}


def kv_cache_bytes(layers: int, heads: int, head_dim: int, seq_len: int,
                   batch_size: int, dtype: str) -> float:
    """Return the number of bytes required for the KV cache.

    Parameters
    ----------
    layers : int
        Number of transformer layers.
    heads : int
        Number of attention heads.
    head_dim : int
        Dimension of each attention head.
    seq_len : int
        Sequence length (context window).  Must be positive.
    batch_size : int
        Number of sequences in the batch.  Must be positive.
    dtype : str
        Name of the numeric dtype (fp16, int8, int4, etc.).  Case‑insensitive.

    Returns
    -------
    float
        The number of bytes required to store the KV cache.

    Notes
    -----
    The size grows linearly with sequence length and batch size【459119712734931†L109-L118】.
    """
    if seq_len <= 0 or batch_size <= 0:
        raise ValueError("seq_len and batch_size must be positive")
    dtype_key = dtype.lower()
    if dtype_key not in BYTES_PER_DTYPE:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported dtypes: {list(BYTES_PER_DTYPE)}")
    bytes_per_elem = BYTES_PER_DTYPE[dtype_key]
    # 2 accounts for key and value
    return 2.0 * layers * heads * head_dim * seq_len * batch_size * bytes_per_elem


def human_readable_bytes(num_bytes: float) -> str:
    """Convert a byte count into a human‑readable string (e.g. MB, GB)."""
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    value = num_bytes
    while idx < len(suffixes) - 1 and value >= 1024:
        value /= 1024
        idx += 1
    return f"{value:.2f} {suffixes[idx]}"


def parse_int_list(s: str) -> List[int]:
    """Parse a comma‑separated list of integers."""
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point for the command‑line interface."""
    parser = argparse.ArgumentParser(description="Compute KV cache memory requirements.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS.keys()), help="Predefined model name")
    parser.add_argument("--layers", type=int, help="Number of transformer layers (overrides --model)")
    parser.add_argument("--heads", type=int, help="Number of attention heads (overrides --model)")
    parser.add_argument("--head-dim", type=int, help="Dimension of each attention head (overrides --model)")
    parser.add_argument("--seq-lens", type=str, default="1024", help="Comma‑separated list of sequence lengths")
    parser.add_argument("--batch-sizes", type=str, default="1", help="Comma‑separated list of batch sizes")
    parser.add_argument("--dtype", type=str, default="fp16", help="Numeric precision (fp16, int8, int4, etc.)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON instead of human‑readable text")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Determine architecture parameters
    if args.model:
        spec = MODEL_SPECS[args.model]
        layers = spec.layers
        heads = spec.heads
        head_dim = spec.head_dim
    else:
        if args.layers is None or args.heads is None or args.head_dim is None:
            parser.error("Either --model or --layers, --heads, --head-dim must be provided")
        layers = args.layers
        heads = args.heads
        head_dim = args.head_dim

    seq_lens = parse_int_list(args.seq_lens)
    batch_sizes = parse_int_list(args.batch_sizes)
    dtype = args.dtype

    results = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            bytes_needed = kv_cache_bytes(layers, heads, head_dim, seq_len, batch_size, dtype)
            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "bytes": bytes_needed,
                "readable": human_readable_bytes(bytes_needed),
            })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for item in results:
            print(f"seq_len={item['seq_len']}, batch_size={item['batch_size']} -> {item['readable']} ({int(item['bytes'])} bytes)")


if __name__ == "__main__":
    main()