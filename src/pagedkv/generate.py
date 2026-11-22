"""Batched greedy generation and scoring on top of the manager, cache and model.

This is a validation and benchmarking harness, not a server: a static batch is
prefilled together (reusing cached prefixes), then decoded one token per step.
Continuous batching and request scheduling belong in the serving engine that
imports this library.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from pagedkv.cache import CacheLayout, PagedKVCache
from pagedkv.manager import KVCacheManager
from pagedkv.models.llama import AttnFn, PagedLlama
from pagedkv.ops.reference import AttentionMetadata, paged_attention_ref


@dataclass
class StepInput:
    input_ids: torch.Tensor
    positions: torch.Tensor
    meta: AttentionMetadata
    logits_indices: torch.Tensor


class PagedGenerator:
    def __init__(
        self,
        model: PagedLlama,
        num_blocks: int,
        block_size: int = 16,
        kv_dtype: str = "bf16",
        enable_prefix_caching: bool = True,
        attn_fn: AttnFn = paged_attention_ref,
        num_cpu_blocks: int = 0,
    ) -> None:
        p = next(model.parameters())
        self.model = model
        self.device = p.device
        self.attn_fn = attn_fn
        self.kv = KVCacheManager(
            num_blocks,
            block_size,
            enable_prefix_caching=enable_prefix_caching,
            num_cpu_blocks=num_cpu_blocks,
        )
        layout = CacheLayout(
            model.config.num_hidden_layers,
            num_blocks,
            block_size,
            model.num_kv_heads,
            model.head_dim,
        )
        self.cache = PagedKVCache(layout, kv_dtype, compute_dtype=p.dtype, device=self.device)
        self.cpu_cache = (
            PagedKVCache(
                CacheLayout(
                    layout.num_layers,
                    num_cpu_blocks,
                    block_size,
                    layout.num_kv_heads,
                    layout.head_dim,
                ),
                kv_dtype,
                p.dtype,
                "cpu",
                pin_memory=self.device.type == "cuda",
            )
            if num_cpu_blocks
            else None
        )
        self._next_id = 0
        self.tokens: dict[int, list[int]] = {}

    # ------------------------------------------------------------ primitives

    def new_request(
        self, prompt: Sequence[int], cache_salt: bytes | None = None
    ) -> tuple[int, int]:
        """Allocate a request; returns (request id, number of prompt tokens served from cache)."""
        rid = self._next_id
        self._next_id += 1
        hit = self.kv.find_prefix(prompt, cache_salt)
        alloc = self.kv.allocate(
            rid, len(prompt) - hit.num_tokens, prefix=hit, cache_salt=cache_salt
        )
        if alloc is None:
            raise MemoryError("KV cache is full; use more blocks or a smaller batch")
        self.tokens[rid] = list(prompt)
        return rid, hit.num_tokens

    def fork(self, rid: int) -> int:
        child = self._next_id
        self._next_id += 1
        self.kv.fork(rid, child)
        self.tokens[child] = list(self.tokens[rid])
        return child

    def append(self, rid: int, token: int) -> None:
        """Queue ``token`` as the next input of ``rid`` (allocating its slot, copying on write)."""
        alloc = self.kv.allocate(rid, 1)
        if alloc is None:
            raise MemoryError("KV cache is full")
        self.cache.copy_blocks(alloc.copies)
        self.tokens[rid].append(token)

    def free(self, rid: int) -> None:
        self.kv.free(rid)
        del self.tokens[rid]

    def build_step(
        self, rids: Sequence[int], query_lens: Sequence[int], all_logits: bool = False
    ) -> StepInput:
        ids: list[int] = []
        pos: list[int] = []
        seq_lens, tables, last = [], [], []
        for rid, q in zip(rids, query_lens, strict=True):
            n = self.kv.num_tokens(rid)
            ids.extend(self.tokens[rid][n - q : n])
            pos.extend(range(n - q, n))
            seq_lens.append(n)
            tables.append(self.kv.block_table(rid))
            last.append(len(ids) - 1)
        meta = AttentionMetadata.build(
            query_lens, seq_lens, tables, self.kv.block_size, self.device
        )
        dev = self.device
        idx = torch.arange(len(ids), device=dev) if all_logits else torch.tensor(last, device=dev)
        return StepInput(torch.tensor(ids, device=dev), torch.tensor(pos, device=dev), meta, idx)

    @torch.no_grad()
    def run(self, step: StepInput) -> torch.Tensor:
        logits: torch.Tensor = self.model(
            step.input_ids, step.positions, self.cache, step.meta, self.attn_fn, step.logits_indices
        )
        return logits

    # ------------------------------------------------------------- high level

    @torch.no_grad()
    def generate(
        self, prompts: Sequence[Sequence[int]], max_new_tokens: int, keep: bool = False
    ) -> list[list[int]]:
        """Greedy decoding for a static batch. Blocks are freed (but stay cached) unless ``keep``."""
        rids, query_lens = [], []
        for p in prompts:
            rid, hit = self.new_request(p)
            rids.append(rid)
            query_lens.append(len(p) - hit)
        outs: list[list[int]] = [[] for _ in prompts]
        logits = self.run(self.build_step(rids, query_lens))
        for rid in rids:
            self.kv.commit(rid, self.tokens[rid])
        for _ in range(max_new_tokens):
            nxt = logits.argmax(-1).tolist()
            for i, tok in enumerate(nxt):
                outs[i].append(tok)
            if len(outs[0]) == max_new_tokens:
                break
            for rid, tok in zip(rids, nxt, strict=True):
                self.append(rid, tok)
            logits = self.run(self.build_step(rids, [1] * len(rids)))
            for rid in rids:
                self.kv.commit(rid, self.tokens[rid])
        if not keep:
            for rid in rids:
                self.free(rid)
        return outs

    @torch.no_grad()
    def score(self, token_ids: Sequence[int], chunk_size: int) -> torch.Tensor:
        """Per-token negative log-likelihood, prefilling ``chunk_size`` tokens per step.

        Every chunk after the first attends to K/V read back from the paged cache,
        so a quantized ``kv_dtype`` affects the result exactly as it would in serving.
        """
        rid = self._next_id
        self._next_id += 1
        self.tokens[rid] = list(token_ids)
        nll = []
        done = 0
        while done < len(token_ids) - 1:
            q = min(chunk_size, len(token_ids) - 1 - done)
            if self.kv.allocate(rid, q) is None:
                raise MemoryError("KV cache too small for this sequence")
            logits = self.run(self.build_step([rid], [q], all_logits=True))
            targets = torch.tensor(token_ids[done + 1 : done + q + 1], device=self.device)
            nll.append(torch.nn.functional.cross_entropy(logits, targets, reduction="none"))
            done += q
        self.free(rid)
        return torch.cat(nll)
