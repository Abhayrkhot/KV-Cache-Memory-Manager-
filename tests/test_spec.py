"""Estimator tests against hand-derived values (no transformers needed)."""

from __future__ import annotations

import pytest

from pagedkv.estimate import GIB, CapacityPlan
from pagedkv.presets import load_spec, preset_names, preset_params
from pagedkv.spec import AttnKind, LayerKV, from_hf_config, get_kv_dtype, human_bytes

KIB = 1024


@pytest.mark.parametrize(
    ("preset", "per_token"),
    [
        ("llama-2-7b", 512 * KIB),  # MHA: 2 * 32 layers * 32 heads * 128 * 2 B
        ("llama-2-70b", 320 * KIB),  # GQA, 8 KV heads; the old calculator said 2.5 MiB
        ("llama-3.1-8b", 128 * KIB),
        ("llama-3.1-70b", 320 * KIB),
        ("qwen2.5-7b", 56 * KIB),  # 28 layers * 4 KV heads
        ("deepseek-v3", 61 * 576 * 2),  # MLA latent: (512 + 64) per layer
        ("kimi-k2", 61 * 576 * 2),
        ("jamba-tiny-dev", 2 * 2 * 2 * 64 * 2),  # only 2 of 16 layers are attention
    ],
)
def test_bytes_per_token(preset: str, per_token: int) -> None:
    assert load_spec(preset).bytes_per_token("bf16") == per_token


def test_sliding_window_caps_growth() -> None:
    spec = load_spec("mistral-7b-v0.1")  # all 32 layers slide over 4096 tokens
    assert spec.count(AttnKind.SLIDING) == 32
    assert spec.seq_bytes(32768) == spec.seq_bytes(4096) == 4096 * 128 * KIB


def test_gemma2_alternates_local_and_global() -> None:
    spec = load_spec("gemma-2-9b")
    assert spec.count(AttnKind.SLIDING) == 21
    assert spec.count(AttnKind.FULL) == 21
    per_layer = 8 * 256 * 2 * 2
    assert spec.seq_bytes(8192) == 21 * 8192 * per_layer + 21 * 4096 * per_layer


def test_gpt_oss_layer_types() -> None:
    spec = load_spec("gpt-oss-20b")
    assert spec.count(AttnKind.SLIDING) == 12
    assert {la.window for la in spec.layers if la.kind is AttnKind.SLIDING} == {128}


def test_hybrid_layers_hold_no_per_token_cache() -> None:
    spec = load_spec("granite-4.0-h-tiny")
    assert spec.count(AttnKind.NONE) > spec.count(AttnKind.FULL) > 0
    assert any("recurrent" in n for n in spec.notes)


def test_quantized_dtypes_include_scale_overhead() -> None:
    assert get_kv_dtype("bf16").vector_bytes(128) == 256
    assert get_kv_dtype("fp8").vector_bytes(128) == 128
    assert get_kv_dtype("int8").vector_bytes(128) == 130  # + fp16 scale
    assert get_kv_dtype("int4").vector_bytes(128) == 64 + 4 * 4  # 4 groups of 32, scale + zero
    assert get_kv_dtype("nvfp4").vector_bytes(128) == 64 + 8
    with pytest.raises(ValueError, match="unknown KV dtype"):
        get_kv_dtype("int3")


def test_paged_rounds_up_to_blocks() -> None:
    spec = load_spec("llama-3.1-8b")
    assert spec.paged_seq_bytes(17, 16) == 32 * spec.bytes_per_token()
    assert spec.paged_seq_bytes(16, 16) == spec.seq_bytes(16)


@pytest.mark.parametrize(
    ("seq_len", "blocks"),
    [(0, 0), (3, 1), (8, 2), (9, 3), (12, 2), (13, 3), (20, 2), (21, 3), (24, 2)],
)
def test_sliding_layer_frees_blocks_behind_window(seq_len: int, blocks: int) -> None:
    layer = LayerKV(AttnKind.SLIDING, 1, 8, 8, window=8)
    assert layer.resident_blocks(seq_len, 4) == blocks


def test_tensor_parallel_sharding() -> None:
    spec = load_spec("llama-3.1-70b")
    assert spec.shard(8).bytes_per_token() == spec.bytes_per_token() / 8
    assert spec.shard(16).bytes_per_token() == spec.shard(8).bytes_per_token()  # 1 head replicated
    mla = load_spec("deepseek-v3")
    assert mla.shard(8).bytes_per_token() == mla.bytes_per_token()  # latent is replicated


def test_multimodal_wrapper_uses_text_config() -> None:
    spec = load_spec("gemma-3-27b")
    assert spec.num_layers == 62
    assert spec.count(AttnKind.FULL) == 62 // 6


def test_unknown_layer_type_is_an_error() -> None:
    cfg = {
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 8,
        "layer_types": ["full_attention", "mystery"],
    }
    with pytest.raises(ValueError, match="unknown layer type"):
        from_hf_config(cfg)


def test_capacity_plan() -> None:
    spec = load_spec("llama-3.1-8b")
    params = preset_params("llama-3.1-8b")
    assert params == 8_030_261_248
    plan = CapacityPlan.build(spec, gpu_gib=80, num_params=params, gpu_util=0.9, overhead_gib=2)
    budget = 0.9 * 80 * GIB - params * 2 - 2 * GIB
    assert plan.kv_budget == pytest.approx(budget)
    assert plan.max_seqs_paged(4096) == int(budget // (4096 * 128 * KIB))
    # Reserving the full 128K context per request admits ~3 instead of ~110.
    assert plan.max_seqs_reserved(131072) == int(budget // (131072 * 128 * KIB))
    assert plan.max_seqs_reserved(131072) * 30 < plan.max_seqs_paged(4096)
    fp8 = CapacityPlan.build(spec, gpu_gib=80, num_params=params, kv_dtype="fp8")
    assert fp8.max_seqs_paged(4096) == pytest.approx(2 * plan.max_seqs_paged(4096), abs=1)


def test_capacity_plan_reports_when_weights_do_not_fit() -> None:
    spec = load_spec("llama-3.1-70b")
    plan = CapacityPlan.build(spec, gpu_gib=80, num_params=preset_params("llama-3.1-70b") or 0)
    assert not plan.fits
    assert plan.max_seqs_paged(1024) == 0
    tp4 = CapacityPlan.build(
        spec, gpu_gib=80, num_params=preset_params("llama-3.1-70b") or 0, tensor_parallel=4
    )
    assert tp4.fits


def test_every_preset_parses() -> None:
    for name in preset_names():
        spec = load_spec(name)
        assert spec.num_cache_layers > 0, name


def test_human_bytes() -> None:
    assert human_bytes(512) == "512 B"
    assert human_bytes(320 * KIB) == "320.00 KiB"
    assert human_bytes(3 * GIB) == "3.00 GiB"
