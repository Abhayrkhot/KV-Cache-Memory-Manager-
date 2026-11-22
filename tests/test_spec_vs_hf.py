"""The estimator against what transformers really allocates (meta device: no weights downloaded)."""

from __future__ import annotations

import logging
import warnings

import pytest

pytest.importorskip("transformers")

from pagedkv.hf_meta import measure_hf_cache
from pagedkv.presets import load_preset
from pagedkv.spec import AttnKind, from_hf_config, get_kv_dtype

pytestmark = pytest.mark.hf
BF16 = get_kv_dtype("bf16")


@pytest.fixture(autouse=True)
def _quiet() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)


@pytest.mark.parametrize(
    ("preset", "seq_len"),
    [
        ("llama-3.1-8b", 100),  # GQA
        ("deepseek-v3", 100),  # MLA latent
        ("gemma-2-9b", 4200),  # alternating sliding window, past the window
        ("gpt-oss-20b", 200),  # layer_types, window 128
        ("granite-4.0-h-tiny", 100),  # Mamba hybrid
    ],
)
def test_matches_transformers_layer_by_layer(preset: str, seq_len: int) -> None:
    cfg = load_preset(preset)
    spec = from_hf_config(cfg, preset)
    measured = measure_hf_cache(cfg, seq_len)
    assert len(measured) == spec.num_layers
    for m, layer in zip(measured, spec.layers, strict=True):
        want = layer.resident_tokens(seq_len)
        if layer.kind is AttnKind.SLIDING and layer.window and seq_len >= layer.window:
            want = layer.window - 1  # HF drops the oldest token once the window is full
        assert m.tokens == want, (m, layer)
        assert m.bytes == want * layer.bytes_per_token(BF16), (m, layer)
