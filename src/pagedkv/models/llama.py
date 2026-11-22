"""Llama-family decoder (Llama 2/3, Mistral, Qwen2/2.5, Qwen3, SmolLM2) on a paged KV cache.

Parameter names mirror Hugging Face, so ``load_state_dict`` takes HF weights as is.
The batch is flattened (``[num_tokens, hidden]``) with per-sequence lengths in
``AttentionMetadata``: one forward can mix prefills, cached-prefix prefills and
decodes. Rotary embeddings come from transformers' own implementation so every
``rope_scaling`` variant (llama3, yarn, ...) matches exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from pagedkv.cache import PagedKVCache
from pagedkv.ops.reference import AttentionMetadata, paged_attention_ref

AttnFn = Callable[
    [torch.Tensor, PagedKVCache, int, AttentionMetadata, float, "int | None"], torch.Tensor
]

SUPPORTED = ("llama", "mistral", "qwen2", "qwen3")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        out: torch.Tensor = self.weight * h.to(dtype)
        return out


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads or cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        bias = bool(getattr(cfg, "attention_bias", False)) or cfg.model_type == "qwen2"
        h = cfg.hidden_size
        self.q_proj = nn.Linear(h, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(h, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, h, bias=cfg.model_type != "qwen2" and bias
        )
        self.qk_norm = cfg.model_type == "qwen3"
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.scale = self.head_dim**-0.5
        self.sliding_window = _layer_window(cfg, layer_idx)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: PagedKVCache,
        meta: AttentionMetadata,
        attn_fn: AttnFn,
    ) -> torch.Tensor:
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        cache.write(self.layer_idx, k, v, meta.slot_mapping)
        o = attn_fn(q, cache, self.layer_idx, meta, self.scale, self.sliding_window)
        out: torch.Tensor = self.o_proj(o.reshape(t, self.num_heads * self.head_dim))
        return out


def _layer_window(cfg: Any, layer_idx: int) -> int | None:
    window = getattr(cfg, "sliding_window", None)
    if not window:
        return None
    layer_types = getattr(cfg, "layer_types", None)
    if layer_types:
        return int(window) if layer_types[layer_idx] == "sliding_attention" else None
    if cfg.model_type == "mistral":
        return int(window)
    if cfg.model_type in ("qwen2", "qwen3") and getattr(cfg, "use_sliding_window", False):
        return int(window) if layer_idx >= getattr(cfg, "max_window_layers", 0) else None
    return None


class MLP(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        h, i = cfg.hidden_size, cfg.intermediate_size
        bias = bool(getattr(cfg, "mlp_bias", False))
        self.gate_proj = nn.Linear(h, i, bias=bias)
        self.up_proj = nn.Linear(h, i, bias=bias)
        self.down_proj = nn.Linear(i, h, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Any, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: PagedKVCache,
        meta: AttentionMetadata,
        attn_fn: AttnFn,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, meta, attn_fn)
        out: torch.Tensor = x + self.mlp(self.post_attention_layernorm(x))
        return out


class _Body(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        self.rotary_emb = LlamaRotaryEmbedding(config=cfg)


class PagedLlama(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        if cfg.model_type not in SUPPORTED:
            raise ValueError(f"model_type {cfg.model_type!r} not supported; supported: {SUPPORTED}")
        self.config = cfg
        self.model = _Body(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if getattr(cfg, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def num_kv_heads(self) -> int:
        return int(self.config.num_key_value_heads or self.config.num_attention_heads)

    @property
    def head_dim(self) -> int:
        cfg = self.config
        return int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedKVCache,
        meta: AttentionMetadata,
        attn_fn: AttnFn = paged_attention_ref,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits for ``logits_indices`` (default: every token)."""
        x = self.model.embed_tokens(input_ids)
        cos, sin = self.model.rotary_emb(x.unsqueeze(0), positions.unsqueeze(0))
        cos, sin = cos[0], sin[0]
        for layer in self.model.layers:
            x = layer(x, cos, sin, cache, meta, attn_fn)
        x = self.model.norm(x)
        if logits_indices is not None:
            x = x[logits_indices]
        logits: torch.Tensor = self.lm_head(x).float()
        return logits

    @classmethod
    def from_hf_model(cls, hf_model: nn.Module) -> PagedLlama:
        """Copy weights from a loaded transformers model (same dtype and device)."""
        p = next(hf_model.parameters())
        with torch.device("meta"):
            model = cls(hf_model.config)
        model = model.to_empty(device=p.device).to(p.dtype)
        state = {k: v for k, v in hf_model.state_dict().items() if "rotary_emb" not in k}
        missing, unexpected = model.load_state_dict(state, strict=False)
        missing = [
            m
            for m in missing
            if "rotary_emb" not in m
            and not (m == "lm_head.weight" and model.config.tie_word_embeddings)
        ]
        if missing or unexpected:
            raise ValueError(f"weight mismatch: missing={missing} unexpected={unexpected}")
        if getattr(model.config, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
        # to_empty() left the (non-persistent) inverse-frequency buffer uninitialized.
        model.model.rotary_emb = type(model.model.rotary_emb)(config=model.config).to(p.device)
        return model.eval()

    @classmethod
    def from_pretrained(
        cls,
        repo_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cpu",
    ) -> PagedLlama:
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(repo_or_path, dtype=dtype)
        model = cls.from_hf_model(hf)
        del hf
        return model.to(device)
