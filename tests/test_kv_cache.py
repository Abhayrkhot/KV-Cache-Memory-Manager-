import math

import pytest

import os
import sys

# Modify sys.path so that the kv_cache module can be imported when tests
# are executed from the project root.  Without this adjustment,
# Python does not treat ``04-kv-cache-memory-manager`` as a package.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv_cache import kv_cache_bytes, human_readable_bytes, MODEL_SPECS


def test_simple_kv_cache_bytes():
    """Sanity check the KV cache size formula for a small example."""
    # Example: 1 layer, 1 head, head_dim=64, seq_len=10, batch_size=2
    # dtype=fp16 (2 bytes per element).  Expected bytes:
    # 2 * layers * heads * head_dim * seq_len * batch_size * bytes_per_elem
    # = 2 * 1 * 1 * 64 * 10 * 2 * 2 = 5120 bytes.
    expected = 5120
    actual = kv_cache_bytes(layers=1, heads=1, head_dim=64, seq_len=10, batch_size=2, dtype="fp16")
    assert math.isclose(actual, expected)
    # Human readable should be ~5 KB
    readable = human_readable_bytes(actual)
    assert readable.startswith("5.00 KB")


def test_llama7b_per_token():
    """Verify that the per‑token KV cache size for LLaMA‑7B matches
    published estimates (approx. 512 KB per token in FP16)."""
    spec = MODEL_SPECS["llama7b"]
    bytes_per_token = kv_cache_bytes(
        layers=spec.layers,
        heads=spec.heads,
        head_dim=spec.head_dim,
        seq_len=1,
        batch_size=1,
        dtype="fp16",
    )
    # Should be about 512 KiB (524288 bytes).  Allow for rounding.
    assert abs(bytes_per_token - 524288) < 1e-6