"""Paged attention in plain PyTorch: the correctness oracle for the Triton kernels.

One call handles any mix of prefill, chunked prefill, prefix-cache hits and
decode. The batch is flattened: sequence ``b`` contributes
``query_lens[b]`` new tokens, which are the *last* ``query_lens[b]`` of its
``seq_lens[b]`` cached tokens, so query ``j`` sits at position
``seq_lens[b] - query_lens[b] + j`` and attends causally to every cached
position up to its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from pagedkv.cache import PagedKVCache


@dataclass
class AttentionMetadata:
    query_start_loc: torch.Tensor
    """[B + 1] cumulative query lengths (int32)."""
    seq_lens: torch.Tensor
    """[B] cached tokens per sequence after this step's writes (int32)."""
    block_tables: torch.Tensor
    """[B, max_blocks] physical block ids (int32, padded with 0)."""
    slot_mapping: torch.Tensor
    """[num_tokens] flat slot for each new token's K/V (int64)."""
    query_lens_cpu: list[int]
    seq_lens_cpu: list[int]
    block_tables_cpu: list[list[int]]
    backend_state: dict[str, Any] = field(default_factory=dict)
    """Per-step scratch space for attention backends (computed once, reused by every layer)."""

    @property
    def num_seqs(self) -> int:
        return len(self.seq_lens_cpu)

    @property
    def max_query_len(self) -> int:
        return max(self.query_lens_cpu) if self.query_lens_cpu else 0

    @property
    def max_seq_len(self) -> int:
        return max(self.seq_lens_cpu) if self.seq_lens_cpu else 0

    @classmethod
    def build(
        cls,
        query_lens: Sequence[int],
        seq_lens: Sequence[int],
        block_tables: Sequence[Sequence[int]],
        block_size: int,
        device: torch.device | str = "cpu",
    ) -> AttentionMetadata:
        if not (len(query_lens) == len(seq_lens) == len(block_tables)):
            raise ValueError(
                "query_lens, seq_lens and block_tables must have one entry per sequence"
            )
        slots: list[int] = []
        for q, s, table in zip(query_lens, seq_lens, block_tables, strict=True):
            if not 0 < q <= s:
                raise ValueError(f"need 0 < query_len <= seq_len, got {q}, {s}")
            if len(table) * block_size < s:
                raise ValueError(f"block table of {len(table)} blocks cannot hold {s} tokens")
            for pos in range(s - q, s):
                slots.append(table[pos // block_size] * block_size + pos % block_size)
        width = max((len(t) for t in block_tables), default=0)
        padded = [list(t) + [0] * (width - len(t)) for t in block_tables]
        starts = [0]
        for q in query_lens:
            starts.append(starts[-1] + q)
        return cls(
            query_start_loc=torch.tensor(starts, dtype=torch.int32, device=device),
            seq_lens=torch.tensor(list(seq_lens), dtype=torch.int32, device=device),
            block_tables=torch.tensor(padded, dtype=torch.int32, device=device).reshape(
                len(padded), width
            ),
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
            query_lens_cpu=list(query_lens),
            seq_lens_cpu=list(seq_lens),
            block_tables_cpu=[list(t) for t in block_tables],
        )


def paged_attention_ref(
    query: torch.Tensor,
    cache: PagedKVCache,
    layer: int,
    meta: AttentionMetadata,
    scale: float,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """``query`` is ``[num_tokens, num_heads, head_dim]``; returns the same shape."""
    num_heads = query.shape[1]
    kv_heads = cache.layout.num_kv_heads
    group = num_heads // kv_heads
    out = torch.empty_like(query)
    bs = cache.block_size
    for b in range(meta.num_seqs):
        qs = int(meta.query_start_loc[b])
        q_len, s_len = meta.query_lens_cpu[b], meta.seq_lens_cpu[b]
        n_blocks = -(-s_len // bs)
        ids = torch.tensor(meta.block_tables_cpu[b][:n_blocks], device=cache.device)
        k, v = cache.gather(layer, ids, s_len)  # [s, kvh, d]
        k = k.repeat_interleave(group, dim=1).float()
        v = v.repeat_interleave(group, dim=1).float()
        q = query[qs : qs + q_len].float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        q_pos = torch.arange(s_len - q_len, s_len, device=query.device)
        k_pos = torch.arange(s_len, device=query.device)
        mask = k_pos[None, :] > q_pos[:, None]
        if sliding_window is not None:
            mask |= k_pos[None, :] <= q_pos[:, None] - sliding_window
        scores = scores.masked_fill(mask[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out[qs : qs + q_len] = torch.einsum("hqk,khd->qhd", probs, v).to(query.dtype)
    return out
