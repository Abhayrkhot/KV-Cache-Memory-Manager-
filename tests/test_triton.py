"""Triton decode kernel vs. the PyTorch reference.

Runs on a CUDA GPU, or on the CPU under Triton's interpreter
(``TRITON_INTERPRET=1``, see ``scripts/test_triton_interpreter.sh``), which
executes the same kernel source with NumPy and is what checks the kernel on
machines without a GPU.
"""

from __future__ import annotations

import os
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from pagedkv.cache import CacheLayout, PagedKVCache  # noqa: E402
from pagedkv.ops.attention import triton_attention  # noqa: E402
from pagedkv.ops.reference import AttentionMetadata, paged_attention_ref  # noqa: E402
from pagedkv.ops.triton_paged import choose_partition, paged_attention_decode  # noqa: E402

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
if not torch.cuda.is_available() and not INTERPRET:
    pytest.skip("needs CUDA or TRITON_INTERPRET=1", allow_module_level=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.gpu
DTYPES = [torch.float16] if INTERPRET else [torch.float16, torch.bfloat16]
TOL = {torch.float16: 3e-3, torch.bfloat16: 2e-2}


def _setup(seq_lens, H, KVH, D, BS, dtype, kv_dtype=None, seed=0):  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    need = sum(-(-s // BS) for s in seq_lens)
    num_blocks = need + 7
    kv_name = kv_dtype or {torch.float16: "fp16", torch.bfloat16: "bf16"}[dtype]
    cache = PagedKVCache(
        CacheLayout(1, num_blocks, BS, KVH, D), kv_name, compute_dtype=dtype, device=DEVICE
    )
    free = list(range(num_blocks))
    rng.shuffle(free)  # scattered, non-contiguous physical blocks
    tables = [[free.pop() for _ in range(-(-s // BS))] for s in seq_lens]
    meta = AttentionMetadata.build(seq_lens, seq_lens, tables, BS, DEVICE)
    n = sum(seq_lens)
    k = torch.randn(n, KVH, D, generator=g).to(DEVICE, dtype)
    v = torch.randn(n, KVH, D, generator=g).to(DEVICE, dtype)
    if kv_dtype == "fp8":
        cache.fp8_scales[0] = torch.tensor([0.05, 0.03])
    cache.write(0, k, v, meta.slot_mapping)
    q = torch.randn(len(seq_lens), H, D, generator=g).to(DEVICE, dtype)
    dmeta = AttentionMetadata.build([1] * len(seq_lens), seq_lens, tables, BS, DEVICE)
    return cache, dmeta, q


CASES = [
    # seq_lens,                H,  KVH, D,   BS
    ([1, 17, 64, 200], 8, 2, 64, 16),  # GQA group 4
    ([5, 33], 4, 4, 64, 16),  # MHA (group 1)
    ([40, 130, 7], 14, 2, 64, 32),  # group 7 (Qwen2.5-0.5B), not a power of two
    ([100, 3], 4, 1, 128, 16),  # MQA
    ([50, 77], 4, 2, 80, 16),  # head_dim not a power of two
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("seq_lens", "H", "KVH", "D", "BS"), CASES)
@pytest.mark.parametrize("partition", [0, 32])
def test_decode_matches_reference(seq_lens, H, KVH, D, BS, dtype, partition) -> None:  # type: ignore[no-untyped-def]
    if partition and partition % BS:
        partition = BS * 2
    cache, meta, q = _setup(seq_lens, H, KVH, D, BS, dtype)
    scale = D**-0.5
    ref = paged_attention_ref(q, cache, 0, meta, scale)
    out = paged_attention_decode(
        q,
        cache.layer(0),
        meta.block_tables,
        meta.seq_lens,
        scale,
        max(seq_lens),
        partition_size=partition,
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=TOL[dtype], rtol=TOL[dtype])


def test_fp8_cache_dequantizes_with_scales() -> None:
    dtype = torch.float16
    cache, meta, q = _setup([9, 70], 8, 2, 64, 16, dtype, kv_dtype="fp8")
    ref = paged_attention_ref(q, cache, 0, meta, 0.125)
    out = paged_attention_decode(
        q, cache.layer(0), meta.block_tables, meta.seq_lens, 0.125, 70, k_scale=0.05, v_scale=0.03
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=TOL[dtype], rtol=TOL[dtype])


def test_mixed_prefill_and_decode_batch() -> None:
    dtype = DTYPES[-1]
    H, KVH, D, BS = 8, 2, 64, 16
    seq_lens, query_lens = [30, 45, 12], [1, 20, 1]
    cache, _, _ = _setup(seq_lens, H, KVH, D, BS, dtype)
    meta = AttentionMetadata.build(
        query_lens, seq_lens, _tables_from(cache, seq_lens, BS), BS, DEVICE
    )
    q = torch.randn(sum(query_lens), H, D).to(DEVICE, dtype)
    ref = paged_attention_ref(q, cache, 0, meta, D**-0.5)
    out = triton_attention(q, cache, 0, meta, D**-0.5)
    torch.testing.assert_close(out.float(), ref.float(), atol=TOL[dtype], rtol=TOL[dtype])


def _tables_from(cache: PagedKVCache, seq_lens: list[int], bs: int) -> list[list[int]]:
    # Same deterministic shuffled assignment _setup used (seed 0).
    rng = random.Random(0)
    free = list(range(cache.layout.num_blocks))
    rng.shuffle(free)
    return [[free.pop() for _ in range(-(-s // bs))] for s in seq_lens]


@pytest.mark.skipif(INTERPRET, reason="long contexts are too slow for the interpreter")
@pytest.mark.parametrize("dtype", DTYPES)
def test_long_context_split_kv(dtype) -> None:  # type: ignore[no-untyped-def]
    seq_lens = [8191, 3000, 1]
    cache, meta, q = _setup(seq_lens, 32, 8, 128, 16, dtype)
    ref = paged_attention_ref(q, cache, 0, meta, 128**-0.5)
    for partition in (None, 512, 0):
        out = paged_attention_decode(
            q,
            cache.layer(0),
            meta.block_tables,
            meta.seq_lens,
            128**-0.5,
            max(seq_lens),
            partition_size=partition,
        )
        torch.testing.assert_close(out.float(), ref.float(), atol=TOL[dtype], rtol=TOL[dtype])


def test_choose_partition() -> None:
    assert choose_partition(256, 8, 4096, 108) == 0  # grid already fills the GPU
    assert choose_partition(1, 8, 256, 108) == 0  # too short to split
    p = choose_partition(1, 8, 32768, 108)
    assert p >= 256
    assert p & (p - 1) == 0
    assert choose_partition(1, 8, 32768, 108, block_size=512) % 512 == 0
