"""Physical cache storage: quantization metadata, calibration, block moves."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pagedkv.cache import CacheLayout, PagedKVCache  # noqa: E402
from pagedkv.spec import AttnKind, KVSpec, LayerKV  # noqa: E402

LAYOUT = CacheLayout(num_layers=2, num_blocks=4, block_size=4, num_kv_heads=2, head_dim=8)


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8", "int8"])
def test_nbytes_matches_estimator(kv_dtype: str) -> None:
    cache = PagedKVCache(LAYOUT, kv_dtype)
    layer = LayerKV(AttnKind.FULL, LAYOUT.num_kv_heads, LAYOUT.head_dim, LAYOUT.head_dim)
    spec = KVSpec("t", (layer,) * LAYOUT.num_layers)
    slots = LAYOUT.num_blocks * LAYOUT.block_size
    assert cache.nbytes() == spec.bytes_per_token(kv_dtype) * slots


@pytest.mark.parametrize("kv_dtype", ["fp32", "fp8", "int8"])
def test_write_gather_roundtrip(kv_dtype: str) -> None:
    cache = PagedKVCache(LAYOUT, kv_dtype, compute_dtype=torch.float32)
    k, v = torch.randn(6, 2, 8), torch.randn(6, 2, 8)
    slots = torch.tensor([8, 9, 10, 11, 0, 1])  # block 2 then block 0
    cache.write(1, k, v, slots)
    gk, gv = cache.gather(1, torch.tensor([2, 0]), 6)
    tol = {"fp32": 0.0, "fp8": 0.13, "int8": 0.03}[kv_dtype]
    assert (gk - k).abs().max() <= tol * k.abs().max() + 1e-6
    assert (gv - v).abs().max() <= tol * v.abs().max() + 1e-6


def test_padding_slots_are_skipped_even_while_calibrating() -> None:
    cache = PagedKVCache(LAYOUT, "fp8")
    cache.start_calibration()
    cache.write(0, torch.randn(3, 2, 8), torch.randn(3, 2, 8), torch.tensor([-1, -1, -1]))
    cache.write(0, torch.ones(1, 2, 8) * 3, torch.ones(1, 2, 8), torch.tensor([-1]).clamp(min=0))
    cache.finish_calibration(key_per_channel=True)
    assert cache.k_channel_scales is not None
    assert torch.allclose(cache.k_channel_scales[0], torch.full((2, 8), 3 / 448))


def test_copy_blocks_copies_every_layer_and_scales() -> None:
    cache = PagedKVCache(LAYOUT, "int8", compute_dtype=torch.float32)
    for layer in range(2):
        cache.write(layer, torch.randn(4, 2, 8), torch.randn(4, 2, 8), torch.arange(4))
    cache.copy_blocks([(0, 3)])
    for layer in range(2):
        a = cache.gather(layer, torch.tensor([0]), 4)
        b = cache.gather(layer, torch.tensor([3]), 4)
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])
