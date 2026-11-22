"""Per-layer KV-cache layout of a model, derived from its Hugging Face config.

The textbook formula ``2 * layers * heads * head_dim * bytes`` is wrong for most
current models:

* Grouped-query attention (Llama 2 70B, Llama 3, Mistral, Qwen) caches
  ``num_key_value_heads``, not ``num_attention_heads``. Llama-3.1-70B is
  320 KiB/token, not 2.5 MiB.
* Multi-head latent attention (DeepSeek-V2/V3, Kimi K2) caches one compressed
  latent of ``kv_lora_rank + qk_rope_head_dim`` per layer instead of K and V.
* Sliding-window layers (Mistral v0.1, Gemma 2/3, gpt-oss) keep only the last
  ``sliding_window`` tokens, so their cost stops growing with context.
* Hybrid models (Jamba, Granite 4.0-H, LFM2) have recurrent or convolution
  layers that hold a fixed-size state instead of a per-token cache.
* Quantized KV formats carry scale metadata that the bit width alone hides.

``KVSpec`` captures all of this per layer. ``tests/test_spec_vs_hf.py`` checks it
against the cache tensors Hugging Face transformers actually allocates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class KVDtype:
    """Storage format of cached K/V elements, including quantization metadata."""

    name: str
    bits: int
    description: str
    vector_scale_bytes: int = 0
    """Metadata bytes per (token, head) vector, e.g. one fp16 scale per token per head."""
    group_size: int = 0
    """If non-zero, one scale group per ``group_size`` elements of a head vector."""
    group_scale_bytes: int = 0

    def vector_bytes(self, dim: int) -> float:
        """Bytes to store one head vector of ``dim`` elements."""
        if dim == 0:
            return 0.0
        size = dim * self.bits / 8 + self.vector_scale_bytes
        if self.group_size:
            size += cdiv(dim, self.group_size) * self.group_scale_bytes
        return size


KV_DTYPES: dict[str, KVDtype] = {
    d.name: d
    for d in (
        KVDtype("fp32", 32, "float32"),
        KVDtype("fp16", 16, "float16"),
        KVDtype("bf16", 16, "bfloat16"),
        KVDtype("fp8", 8, "FP8 e4m3/e5m2 with a per-tensor scale (vLLM kv_cache_dtype=fp8)"),
        KVDtype("int8", 8, "INT8, symmetric, one fp16 scale per token per head", 2),
        KVDtype(
            "int4", 4, "INT4, asymmetric, fp16 scale + fp16 zero point per 32 elements", 0, 32, 4
        ),
        KVDtype("nvfp4", 4, "NVFP4 (e2m1), one e4m3 scale per 16 elements", 0, 16, 1),
        KVDtype("mxfp4", 4, "MXFP4 (e2m1), one e8m0 scale per 32 elements", 0, 32, 1),
    )
}
KV_DTYPES["float16"] = KV_DTYPES["fp16"]
KV_DTYPES["bfloat16"] = KV_DTYPES["bf16"]
KV_DTYPES["float32"] = KV_DTYPES["fp32"]


def get_kv_dtype(name: str | KVDtype) -> KVDtype:
    if isinstance(name, KVDtype):
        return name
    try:
        return KV_DTYPES[name.lower()]
    except KeyError:
        known = ", ".join(sorted(KV_DTYPES))
        raise ValueError(f"unknown KV dtype {name!r}; known: {known}") from None


class AttnKind(str, Enum):
    FULL = "full"
    SLIDING = "sliding"
    MLA = "mla"
    NONE = "none"  # recurrent / convolution layer: fixed-size state, no per-token cache


@dataclass(frozen=True)
class LayerKV:
    kind: AttnKind
    num_kv_heads: int = 0
    k_dim: int = 0
    v_dim: int = 0
    window: int | None = None
    """Tokens kept by a sliding-window layer (None = unbounded)."""

    def bytes_per_token(self, dtype: KVDtype) -> float:
        if self.kind is AttnKind.NONE:
            return 0.0
        return self.num_kv_heads * (dtype.vector_bytes(self.k_dim) + dtype.vector_bytes(self.v_dim))

    def resident_tokens(self, seq_len: int) -> int:
        """Tokens whose K/V must stay cached once ``seq_len`` tokens have been processed."""
        if self.kind is AttnKind.NONE:
            return 0
        return seq_len if self.window is None else min(seq_len, self.window)

    def resident_blocks(self, seq_len: int, block_size: int) -> int:
        """Pages a paged cache needs for this layer; windowed layers free pages behind the window."""
        if self.kind is AttnKind.NONE or seq_len == 0:
            return 0
        blocks = cdiv(seq_len, block_size)
        if self.window is None:
            return blocks
        first_live = max(0, seq_len - self.window)
        return blocks - first_live // block_size


@dataclass(frozen=True)
class KVSpec:
    name: str
    layers: tuple[LayerKV, ...]
    max_position_embeddings: int | None = None
    model_type: str = ""
    notes: tuple[str, ...] = field(default=())

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_cache_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.kind is not AttnKind.NONE)

    def count(self, kind: AttnKind) -> int:
        return sum(1 for layer in self.layers if layer.kind is kind)

    def bytes_per_token(self, dtype: str | KVDtype = "bf16") -> float:
        """Marginal cache bytes per token while every window is still filling up."""
        dt = get_kv_dtype(dtype)
        return sum(layer.bytes_per_token(dt) for layer in self.layers)

    def seq_bytes(self, seq_len: int, dtype: str | KVDtype = "bf16") -> float:
        """Exact cache bytes for one sequence of ``seq_len`` tokens (contiguous storage)."""
        _check_len(seq_len)
        dt = get_kv_dtype(dtype)
        return sum(
            layer.resident_tokens(seq_len) * layer.bytes_per_token(dt) for layer in self.layers
        )

    def paged_seq_bytes(
        self, seq_len: int, block_size: int, dtype: str | KVDtype = "bf16"
    ) -> float:
        """Cache bytes for one sequence in a paged cache with ``block_size``-token pages."""
        _check_len(seq_len)
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        dt = get_kv_dtype(dtype)
        return sum(
            layer.resident_blocks(seq_len, block_size) * block_size * layer.bytes_per_token(dt)
            for layer in self.layers
        )

    def shard(self, tensor_parallel: int) -> KVSpec:
        """Per-GPU spec under tensor parallelism.

        KV heads split across ranks; when there are fewer KV heads than ranks each
        rank keeps one replicated head (vLLM's ``max(1, kv_heads // tp)``). The MLA
        latent is not split by head, so every rank stores all of it.
        """
        if tensor_parallel < 1:
            raise ValueError("tensor_parallel must be >= 1")
        if tensor_parallel == 1:
            return self
        layers = tuple(
            layer
            if layer.kind in (AttnKind.MLA, AttnKind.NONE)
            else LayerKV(
                layer.kind,
                max(1, layer.num_kv_heads // tensor_parallel),
                layer.k_dim,
                layer.v_dim,
                layer.window,
            )
            for layer in self.layers
        )
        return KVSpec(
            f"{self.name} (TP={tensor_parallel})",
            layers,
            self.max_position_embeddings,
            self.model_type,
            self.notes,
        )

    def describe(self) -> str:
        parts = []
        for kind in AttnKind:
            n = self.count(kind)
            if n:
                parts.append(f"{n} {kind.value}")
        first = next((la for la in self.layers if la.kind is not AttnKind.NONE), None)
        shape = ""
        if first is not None:
            if first.kind is AttnKind.MLA:
                shape = f", latent dim {first.k_dim}"
            else:
                shape = f", {first.num_kv_heads} KV heads x {first.k_dim}"
        windows = sorted({la.window for la in self.layers if la.window})
        win = f", window {'/'.join(map(str, windows))}" if windows else ""
        return f"{self.name}: {self.num_layers} layers ({', '.join(parts)}){shape}{win}"


def _check_len(seq_len: int) -> None:
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")


_LAYER_TYPE_MAP = {
    "full_attention": AttnKind.FULL,
    "attention": AttnKind.FULL,
    "global_attention": AttnKind.FULL,
    "sliding_attention": AttnKind.SLIDING,
    "chunked_attention": AttnKind.SLIDING,
    "local_attention": AttnKind.SLIDING,
}
_STATEFUL_LAYER_TYPES = {"mamba", "mamba2", "conv", "short_conv", "linear_attention", "recurrent"}


def _layer_kinds(
    cfg: Mapping[str, Any], model_type: str, n: int, notes: list[str]
) -> list[AttnKind]:
    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        if len(layer_types) != n:
            raise ValueError(f"layer_types has {len(layer_types)} entries for {n} layers")
        kinds = []
        for t in layer_types:
            if t in _LAYER_TYPE_MAP:
                kinds.append(_LAYER_TYPE_MAP[t])
            elif t in _STATEFUL_LAYER_TYPES:
                kinds.append(AttnKind.NONE)
            else:
                raise ValueError(f"unknown layer type {t!r}")
        return kinds

    if cfg.get("attn_layer_period"):  # Jamba
        period, offset = int(cfg["attn_layer_period"]), int(cfg.get("attn_layer_offset", 0))
        return [AttnKind.FULL if i % period == offset else AttnKind.NONE for i in range(n)]

    window = cfg.get("sliding_window")
    if model_type == "gemma2":
        return [AttnKind.SLIDING if (i + 1) % 2 else AttnKind.FULL for i in range(n)]
    if model_type in ("gemma3", "gemma3_text"):
        pattern = int(cfg.get("sliding_window_pattern", 6))
        return [AttnKind.SLIDING if (i + 1) % pattern else AttnKind.FULL for i in range(n)]
    if model_type in ("mistral", "mixtral") and window:
        return [AttnKind.SLIDING] * n
    if model_type.startswith(("qwen2", "qwen3")) and cfg.get("use_sliding_window") and window:
        first = int(cfg.get("max_window_layers", 0))
        return [AttnKind.SLIDING if i >= first else AttnKind.FULL for i in range(n)]
    if window and not (model_type.startswith(("qwen2", "qwen3"))):
        notes.append(
            f"config sets sliding_window={window} but {model_type or 'this model'} is not known "
            "to use it; treated as full attention (an upper bound)"
        )
    return [AttnKind.FULL] * n


def from_hf_config(config: Mapping[str, Any], name: str | None = None) -> KVSpec:
    """Build a ``KVSpec`` from a Hugging Face ``config.json`` dictionary."""
    cfg: Mapping[str, Any] = config
    if isinstance(config.get("text_config"), Mapping):  # multimodal wrappers (Gemma 3, Llava, ...)
        cfg = {**config["text_config"]}
    model_type = str(cfg.get("model_type") or config.get("model_type") or "")
    name = name or str(config.get("_name_or_path") or model_type or "model")

    n = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads") or heads)
    head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // heads)
    notes: list[str] = []

    kinds = _layer_kinds(cfg, model_type, n, notes)

    window = cfg.get("sliding_window")
    if model_type == "llama4" or cfg.get("attention_chunk_size"):
        window = cfg.get("attention_chunk_size") or window
        notes.append("chunked attention modeled as a window of attention_chunk_size (upper bound)")
    if any(k is AttnKind.SLIDING for k in kinds) and not window:
        raise ValueError("config has sliding-window layers but no sliding_window size")

    layers: list[LayerKV] = []
    if cfg.get("kv_lora_rank"):
        latent = int(cfg["kv_lora_rank"]) + int(cfg.get("qk_rope_head_dim", 0))
        notes.append(
            f"MLA: one {latent}-dim latent per token per layer (kv_lora_rank + qk_rope_head_dim), "
            "the layout vLLM and SGLang cache"
        )
        mla = LayerKV(AttnKind.MLA, 1, latent, 0)
        layers = [mla if k is not AttnKind.NONE else LayerKV(AttnKind.NONE) for k in kinds]
    else:
        for k in kinds:
            if k is AttnKind.NONE:
                layers.append(LayerKV(AttnKind.NONE))
            else:
                w = int(window) if k is AttnKind.SLIDING and window else None
                layers.append(LayerKV(k, kv_heads, head_dim, head_dim, w))

    if any(k is AttnKind.NONE for k in kinds):
        notes.append(
            f"{sum(k is AttnKind.NONE for k in kinds)} recurrent/conv layers hold a fixed-size "
            "per-sequence state that is not counted here"
        )

    mpe = cfg.get("max_position_embeddings")
    return KVSpec(
        name=name,
        layers=tuple(layers),
        max_position_embeddings=int(mpe) if mpe else None,
        model_type=model_type,
        notes=tuple(notes),
    )


def human_bytes(num: float) -> str:
    """Format a byte count with binary prefixes (KiB = 1024 B)."""
    if num < 0 or math.isnan(num):
        raise ValueError(f"byte count must be non-negative, got {num}")
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if num < 1024 or unit == "TiB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.2f} {unit}"
        num /= 1024
    raise AssertionError("unreachable")
