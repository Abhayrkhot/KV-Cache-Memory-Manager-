"""Measure what Hugging Face transformers actually caches, without downloading weights.

The model is built from its config on PyTorch's ``meta`` device: tensors carry
shapes and dtypes but no storage, so a 671B-parameter model instantiates in about a
second. Running a forward pass on meta tensors fills the real ``DynamicCache``
with correctly shaped (empty) K/V tensors, which we then measure layer by layer.

Feed-forward blocks (including MoE routers, whose data-dependent routing cannot run
on meta tensors) and recurrent mixers are replaced by zero modules: they do not
touch the attention cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

_FFN_ATTRS = (
    "mlp",
    "block_sparse_moe",
    "feed_forward",
    "moe",
    "mamba",
    "mixer",
    "conv",
    "linear_attn",
)


class _Zero(nn.Module):
    def __init__(self, tuple_out: bool) -> None:
        super().__init__()
        self.tuple_out = tuple_out

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        x = args[0] if args else kwargs.get("hidden_states", kwargs.get("x"))
        if not isinstance(x, torch.Tensor):
            raise TypeError("zero module got no input tensor")
        z = torch.zeros_like(x)
        return (z, None) if self.tuple_out else z


@dataclass(frozen=True)
class LayerCache:
    index: int
    kind: str  # class name of the HF cache layer
    tokens: int
    bytes: int


def _build(config: Mapping[str, Any], dtype: torch.dtype) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg_dict = {k: v for k, v in config.items() if not k.startswith("_")}
    cfg_dict.pop("quantization_config", None)  # FP8/MXFP4 checkpoints: measure the bf16 graph
    cfg_dict.pop("auto_map", None)  # never run remote code
    try:
        cfg = AutoConfig.for_model(**cfg_dict)
    except ValueError:
        # Custom model_type (e.g. kimi_k2) whose architecture transformers ships
        # under another name: map it through the architecture class.
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

        arch = (cfg_dict.get("architectures") or [None])[0]
        by_arch = {v: k for k, v in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.items()}
        if arch not in by_arch:
            raise
        cfg_dict["model_type"] = by_arch[arch]
        cfg = AutoConfig.for_model(**cfg_dict)
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model: nn.Module = AutoModelForCausalLM.from_config(cfg, dtype=dtype)  # type: ignore[no-untyped-call]
    return model


def count_params(config: Mapping[str, Any]) -> int:
    """Exact parameter count of the architecture described by ``config``."""
    model = _build(config, torch.bfloat16)
    return sum(p.numel() for p in model.parameters())


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        obj: Any = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError(f"could not find decoder layers in {type(model).__name__}")


def measure_hf_cache(
    config: Mapping[str, Any], seq_len: int, dtype: torch.dtype = torch.bfloat16
) -> list[LayerCache]:
    """Run a meta-device prefill of ``seq_len`` tokens and report each cache layer's size."""
    last_err: Exception | None = None
    for tuple_out in (False, True):
        model = _build(config, dtype)
        for layer in _decoder_layers(model):
            for attr in _FFN_ATTRS:
                if isinstance(getattr(layer, attr, None), nn.Module):
                    setattr(layer, attr, _Zero(tuple_out))
        ids = torch.zeros(1, seq_len, dtype=torch.long, device="meta")
        try:
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            break
        except (TypeError, ValueError, AttributeError) as err:  # FFN returned the wrong arity
            last_err = err
    else:
        raise RuntimeError(f"meta forward failed: {last_err}") from last_err

    cache = getattr(out, "past_key_values", None)
    if cache is None:
        cache = getattr(out, "cache", None)
    if cache is None:
        raise RuntimeError("model returned no cache")
    result = []
    for i, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        size, tokens = 0, 0
        for t in (keys, values):
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                size += t.numel() * t.element_size()
                tokens = max(tokens, t.shape[-2])
        result.append(LayerCache(i, type(layer).__name__, tokens, size))
    return result
