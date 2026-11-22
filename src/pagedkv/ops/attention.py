"""Attention backends with the ``AttnFn`` signature used by the model.

* ``reference``: pure PyTorch loop (``paged_attention_ref``), any device, the oracle.
* ``triton``: decode tokens go through the Triton paged kernel in one launch;
  prefill / chunked-prefill sequences use PyTorch SDPA on gathered K/V.
  Sliding-window layers and int8 caches fall back to the reference path.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from pagedkv.cache import PagedKVCache
from pagedkv.ops.reference import AttentionMetadata, paged_attention_ref


def _prepare(meta: AttentionMetadata, device: torch.device) -> dict[str, Any]:
    cached: dict[str, Any] | None = meta.backend_state.get("triton")
    if cached is not None:
        return cached
    starts = [0]
    for q in meta.query_lens_cpu:
        starts.append(starts[-1] + q)
    dec = [i for i, q in enumerate(meta.query_lens_cpu) if q == 1]
    pre = [i for i, q in enumerate(meta.query_lens_cpu) if q > 1]
    all_decode = len(dec) == meta.num_seqs
    state = {"starts": starts, "prefill": pre, "all_decode": all_decode}
    if dec:
        if all_decode:
            state["bt"], state["sl"] = meta.block_tables, meta.seq_lens
            state["rows"] = None
        else:
            idx = torch.tensor(dec, device=device)
            state["bt"] = meta.block_tables.index_select(0, idx).contiguous()
            state["sl"] = meta.seq_lens.index_select(0, idx).contiguous()
            state["rows"] = torch.tensor([starts[i] for i in dec], device=device)
        state["max_len"] = max(meta.seq_lens_cpu[i] for i in dec)
    meta.backend_state["triton"] = state
    return state


def _sdpa_prefill(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """q ``[n, H, D]`` (the last n positions), k/v ``[s, KVH, D]`` -> ``[n, H, D]``."""
    n, h, _ = q.shape
    s, kvh, _ = k.shape
    if h != kvh:
        k = k.repeat_interleave(h // kvh, dim=1)
        v = v.repeat_interleave(h // kvh, dim=1)
    q_pos = torch.arange(s - n, s, device=q.device)
    mask = torch.arange(s, device=q.device)[None, :] <= q_pos[:, None]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask, scale=scale
    )
    return o.transpose(0, 1)


def triton_attention(
    query: torch.Tensor,
    cache: PagedKVCache,
    layer: int,
    meta: AttentionMetadata,
    scale: float,
    sliding_window: int | None = None,
) -> torch.Tensor:
    if (
        sliding_window is not None
        or cache.kv_dtype not in ("bf16", "fp16", "fp8")
        or cache.k_channel_scales is not None
    ):
        return paged_attention_ref(query, cache, layer, meta, scale, sliding_window)
    from pagedkv.ops.triton_paged import paged_attention_decode

    st = _prepare(meta, query.device)
    k_scale = float(cache.fp8_scales[layer, 0]) if cache.kv_dtype == "fp8" else 1.0
    v_scale = float(cache.fp8_scales[layer, 1]) if cache.kv_dtype == "fp8" else 1.0
    if st["all_decode"]:
        return paged_attention_decode(
            query,
            cache.layer(layer),
            st["bt"],
            st["sl"],
            scale,
            st["max_len"],
            k_scale=k_scale,
            v_scale=v_scale,
        )
    out = torch.empty_like(query)
    if "bt" in st:
        rows = st["rows"]
        out[rows] = paged_attention_decode(
            query.index_select(0, rows).contiguous(),
            cache.layer(layer),
            st["bt"],
            st["sl"],
            scale,
            st["max_len"],
            k_scale=k_scale,
            v_scale=v_scale,
        )
    bs = cache.block_size
    for i in st["prefill"]:
        a, b = st["starts"][i], st["starts"][i + 1]
        s_len = meta.seq_lens_cpu[i]
        ids = torch.tensor(meta.block_tables_cpu[i][: -(-s_len // bs)], device=query.device)
        k, v = cache.gather(layer, ids, s_len)
        out[a:b] = _sdpa_prefill(query[a:b], k, v, scale)
    return out


BACKENDS = {"reference": paged_attention_ref, "triton": triton_attention}
