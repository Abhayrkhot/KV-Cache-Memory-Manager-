"""Per-request block tables on top of a ``BlockPool``.

The manager is engine-agnostic: requests are identified by any hashable id, and
operations return the physical work the executor must do (block copies for
copy-on-write and swapping) instead of doing it. A scheduler drives it like this::

    hit = kv.find_prefix(prompt_ids)                 # cached leading blocks, if any
    alloc = kv.allocate(rid, len(prompt_ids) - hit.num_tokens, prefix=hit, admit=True)
    if alloc is None: ...                            # not enough memory: keep queued
    executor.copy_blocks(alloc.copies)               # copy-on-write after fork()
    run_model(..., block_table=kv.block_table(rid))
    kv.commit(rid, all_token_ids)                    # publish now-full blocks to the cache
    ...
    kv.free(rid)

Token accounting: ``num_tokens(rid)`` counts tokens that have a KV slot, whether
already computed or being computed in the current step. ``allocate`` extends it.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field

from pagedkv.block_pool import BlockPool, OutOfBlocksError
from pagedkv.hashing import BlockHash, hash_blocks
from pagedkv.spec import cdiv

ReqId = Hashable


@dataclass(frozen=True)
class PrefixHit:
    block_ids: tuple[int, ...] = ()
    block_hashes: tuple[BlockHash, ...] = ()
    num_tokens: int = 0
    num_queried_tokens: int = 0


NO_HIT = PrefixHit()


@dataclass(frozen=True)
class Allocation:
    new_blocks: tuple[int, ...]
    copies: tuple[tuple[int, int], ...] = ()
    """(src, dst) block copies the executor must perform before this step writes K/V."""


@dataclass
class _Request:
    table: list[int] = field(default_factory=list)
    num_tokens: int = 0
    hashes: list[BlockHash] = field(default_factory=list)
    """Hashes of leading full blocks, as far as they have been computed."""
    cache_salt: bytes | None = None


@dataclass
class _Swapped:
    cpu_blocks: list[int]
    num_tokens: int
    hashes: list[BlockHash]
    cache_salt: bytes | None


@dataclass(frozen=True)
class KVStats:
    num_blocks: int
    used_blocks: int
    free_blocks: int
    cached_free_blocks: int
    num_requests: int
    num_swapped: int
    logical_tokens: int
    """Sum of per-request token counts; can exceed physical capacity when blocks are shared."""
    allocated_slots: int
    """Slots in physically allocated blocks (used_blocks * block_size)."""
    prefix_queried_tokens: int
    prefix_hit_tokens: int
    evictions: int
    cow_copies: int
    swap_out_blocks: int
    swap_in_blocks: int
    swap_in_cache_hits: int

    @property
    def usage(self) -> float:
        return self.used_blocks / self.num_blocks

    @property
    def prefix_hit_rate(self) -> float:
        return (
            self.prefix_hit_tokens / self.prefix_queried_tokens
            if self.prefix_queried_tokens
            else 0.0
        )


class KVCacheManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        enable_prefix_caching: bool = True,
        num_cpu_blocks: int = 0,
        watermark: float = 0.0,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if not 0 <= watermark < 1:
            raise ValueError("watermark must be in [0, 1)")
        self.block_size = block_size
        self.pool = BlockPool(num_blocks, enable_caching=enable_prefix_caching)
        self.cpu_pool = BlockPool(num_cpu_blocks, enable_caching=False) if num_cpu_blocks else None
        self.watermark_blocks = int(watermark * num_blocks)
        self._reqs: dict[ReqId, _Request] = {}
        self._swapped: dict[ReqId, _Swapped] = {}
        self._prefix_queried = 0
        self._prefix_hit = 0
        self._cow_copies = 0
        self._swap_out = 0
        self._swap_in = 0
        self._swap_in_hits = 0

    # ------------------------------------------------------------------ queries

    @property
    def enable_prefix_caching(self) -> bool:
        return self.pool.enable_caching

    def __contains__(self, req_id: ReqId) -> bool:
        return req_id in self._reqs

    def block_table(self, req_id: ReqId) -> list[int]:
        return list(self._reqs[req_id].table)

    def num_tokens(self, req_id: ReqId) -> int:
        return self._reqs[req_id].num_tokens

    def is_swapped(self, req_id: ReqId) -> bool:
        return req_id in self._swapped

    def slot(self, req_id: ReqId, position: int) -> int:
        """Flat slot index (block_id * block_size + offset) of token ``position``."""
        req = self._reqs[req_id]
        if not 0 <= position < req.num_tokens:
            raise IndexError(f"position {position} outside [0, {req.num_tokens})")
        return req.table[position // self.block_size] * self.block_size + position % self.block_size

    def find_prefix(self, token_ids: Sequence[int], cache_salt: bytes | None = None) -> PrefixHit:
        """Longest run of cached blocks matching the start of ``token_ids``.

        Never covers the last token: the model must run on at least one token to
        produce logits, and that token's K/V goes into a block the request owns.
        """
        if not self.enable_prefix_caching or len(token_ids) < 2:
            return PrefixHit(num_queried_tokens=len(token_ids))
        bs = self.block_size
        hashes = hash_blocks(
            token_ids[: (len(token_ids) - 1) // bs * bs], bs, cache_salt=cache_salt
        )
        ids: list[int] = []
        for h in hashes:
            block_id = self.pool.lookup(h)
            if block_id is None:
                break
            ids.append(block_id)
        return PrefixHit(tuple(ids), tuple(hashes[: len(ids)]), len(ids) * bs, len(token_ids))

    def blocks_needed(
        self, req_id: ReqId | None, num_new_tokens: int, prefix: PrefixHit = NO_HIT
    ) -> int:
        """Free blocks an ``allocate`` call would consume (including revived cached blocks)."""
        req = self._reqs.get(req_id) if req_id is not None else None
        have_tokens = req.num_tokens if req else prefix.num_tokens
        have_blocks = len(req.table) if req else len(prefix.block_ids)
        need = max(0, cdiv(have_tokens + num_new_tokens, self.block_size) - have_blocks)
        if req is not None and num_new_tokens > 0 and self._needs_cow(req):
            need += 1
        revived = sum(1 for b in prefix.block_ids if self.pool.blocks[b].ref_cnt == 0)
        return need + revived

    def can_allocate(
        self,
        req_id: ReqId | None,
        num_new_tokens: int,
        prefix: PrefixHit = NO_HIT,
        *,
        admit: bool = False,
    ) -> bool:
        reserve = self.watermark_blocks if admit else 0
        return self.blocks_needed(req_id, num_new_tokens, prefix) + reserve <= self.pool.num_free

    # ------------------------------------------------------------------ updates

    def allocate(
        self,
        req_id: ReqId,
        num_new_tokens: int,
        *,
        prefix: PrefixHit = NO_HIT,
        cache_salt: bytes | None = None,
        admit: bool = False,
    ) -> Allocation | None:
        """Give ``req_id`` KV slots for ``num_new_tokens`` more tokens.

        On a request's first call, ``prefix`` attaches cached blocks from
        ``find_prefix`` (the request starts with ``prefix.num_tokens`` computed
        tokens). Returns None, with no side effects, if memory is short.
        """
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be non-negative")
        if req_id in self._swapped:
            raise ValueError(f"request {req_id!r} is swapped out; call swap_in first")
        req = self._reqs.get(req_id)
        if req is not None and prefix.block_ids:
            raise ValueError("a cached prefix can only be attached on a request's first allocation")
        for block_id, h in zip(prefix.block_ids, prefix.block_hashes, strict=True):
            if self.pool.blocks[block_id].block_hash != h:
                raise ValueError(
                    "stale prefix hit: a block was evicted since find_prefix(); look it up again"
                )
        if not self.can_allocate(req_id if req else None, num_new_tokens, prefix, admit=admit):
            return None

        if req is None:
            req = _Request(cache_salt=cache_salt)
            self._reqs[req_id] = req
            if self.enable_prefix_caching and prefix.num_queried_tokens:  # came from find_prefix
                self._prefix_queried += prefix.num_queried_tokens
                self._prefix_hit += prefix.num_tokens
            if prefix.block_ids:
                self.pool.touch(prefix.block_ids)
                req.table.extend(prefix.block_ids)
                req.hashes.extend(prefix.block_hashes)
                req.num_tokens = prefix.num_tokens

        copies: list[tuple[int, int]] = []
        if num_new_tokens > 0 and self._needs_cow(req):
            (dst,) = self.pool.allocate(1)
            src = req.table[-1]
            self.pool.release([src])
            req.table[-1] = dst
            copies.append((src, dst))
            self._cow_copies += 1

        n_new = cdiv(req.num_tokens + num_new_tokens, self.block_size) - len(req.table)
        new_blocks = self.pool.allocate(n_new) if n_new > 0 else []
        req.table.extend(new_blocks)
        req.num_tokens += num_new_tokens
        return Allocation(tuple(new_blocks), tuple(copies))

    def _needs_cow(self, req: _Request) -> bool:
        """The next write lands in a partially filled block that another request also references."""
        if req.num_tokens % self.block_size == 0 or not req.table:
            return False
        return self.pool.blocks[req.table[-1]].ref_cnt > 1

    def commit(self, req_id: ReqId, token_ids: Sequence[int]) -> int:
        """Publish blocks that are now full of computed K/V to the prefix cache.

        Call after the forward pass that computed them. ``token_ids`` must hold at
        least ``num_tokens(req_id)`` tokens. Returns the number of newly hashed blocks.
        """
        req = self._reqs[req_id]
        if len(token_ids) < req.num_tokens:
            raise ValueError(f"need {req.num_tokens} token ids, got {len(token_ids)}")
        bs = self.block_size
        num_full = req.num_tokens // bs
        start = len(req.hashes)
        if not self.enable_prefix_caching or num_full <= start:
            return 0
        parent = req.hashes[-1] if req.hashes else None
        new = hash_blocks(
            token_ids[: num_full * bs], bs, start=start, parent=parent, cache_salt=req.cache_salt
        )
        for i, h in enumerate(new, start):
            self.pool.register(req.table[i], h)
        req.hashes.extend(new)
        return len(new)

    def free(self, req_id: ReqId) -> None:
        """Release a finished or preempted (recompute) request. Its full blocks stay cached."""
        if req_id in self._swapped:
            swapped = self._swapped.pop(req_id)
            assert self.cpu_pool is not None
            self.cpu_pool.release(swapped.cpu_blocks)
            return
        req = self._reqs.pop(req_id)
        # Tail first: a request's last blocks are the least likely to be shared, so
        # they should be evicted before its leading (prompt-prefix) blocks.
        self.pool.release(reversed(req.table))

    def fork(self, parent_id: ReqId, child_id: ReqId) -> None:
        """Share all of ``parent_id``'s blocks with a new request (parallel sampling, beam search)."""
        if child_id in self._reqs or child_id in self._swapped:
            raise ValueError(f"request {child_id!r} already exists")
        parent = self._reqs[parent_id]
        self.pool.touch(parent.table)
        self._reqs[child_id] = _Request(
            list(parent.table), parent.num_tokens, list(parent.hashes), parent.cache_salt
        )

    # ------------------------------------------------------------------ swapping

    def swap_out(self, req_id: ReqId) -> list[tuple[int, int]] | None:
        """Move a request's blocks to host memory. Returns (gpu_block, cpu_block) copies.

        Blocks shared with other running requests stay on the GPU for them; this
        request just drops its reference. Full blocks keep their cache entries, so
        if they are still resident at swap-in time they are reused, not copied back.
        """
        if self.cpu_pool is None:
            raise RuntimeError("swapping needs num_cpu_blocks > 0")
        req = self._reqs[req_id]
        if len(req.table) > self.cpu_pool.num_free:
            return None
        cpu_blocks = self.cpu_pool.allocate(len(req.table))
        copies = list(zip(req.table, cpu_blocks, strict=True))
        del self._reqs[req_id]
        self.pool.release(reversed(req.table))
        self._swapped[req_id] = _Swapped(cpu_blocks, req.num_tokens, req.hashes, req.cache_salt)
        self._swap_out += len(copies)
        return copies

    def _swap_in_plan(self, swapped: _Swapped) -> tuple[list[int | None], int]:
        resident: list[int | None] = []
        need = 0
        for i in range(len(swapped.cpu_blocks)):
            block_id = self.pool.lookup(swapped.hashes[i]) if i < len(swapped.hashes) else None
            resident.append(block_id)
            if block_id is None or self.pool.blocks[block_id].ref_cnt == 0:
                need += 1
        return resident, need

    def can_swap_in(self, req_id: ReqId) -> bool:
        _, need = self._swap_in_plan(self._swapped[req_id])
        return need <= self.pool.num_free

    def swap_in(self, req_id: ReqId) -> list[tuple[int, int]] | None:
        """Bring a swapped request back. Returns (cpu_block, gpu_block) copies, or None if short."""
        swapped = self._swapped[req_id]
        resident, need = self._swap_in_plan(swapped)
        if need > self.pool.num_free:
            return None
        hits = [b for b in resident if b is not None]
        self.pool.touch(hits)
        fresh = iter(self.pool.allocate(len(resident) - len(hits)))
        table: list[int] = []
        copies: list[tuple[int, int]] = []
        for i, (cpu_block, block_id) in enumerate(zip(swapped.cpu_blocks, resident, strict=True)):
            if block_id is None:
                block_id = next(fresh)
                copies.append((cpu_block, block_id))
                if i < len(swapped.hashes):
                    self.pool.register(block_id, swapped.hashes[i])
            table.append(block_id)
        assert self.cpu_pool is not None
        self.cpu_pool.release(swapped.cpu_blocks)
        del self._swapped[req_id]
        self._reqs[req_id] = _Request(table, swapped.num_tokens, swapped.hashes, swapped.cache_salt)
        self._swap_in += len(copies)
        self._swap_in_hits += len(hits)
        return copies

    # ------------------------------------------------------------------ stats

    def stats(self) -> KVStats:
        used = self.pool.num_blocks - self.pool.num_free
        return KVStats(
            num_blocks=self.pool.num_blocks,
            used_blocks=used,
            free_blocks=self.pool.num_free,
            cached_free_blocks=self.pool.num_cached_free(),
            num_requests=len(self._reqs),
            num_swapped=len(self._swapped),
            logical_tokens=sum(r.num_tokens for r in self._reqs.values()),
            allocated_slots=used * self.block_size,
            prefix_queried_tokens=self._prefix_queried,
            prefix_hit_tokens=self._prefix_hit,
            evictions=self.pool.num_evictions,
            cow_copies=self._cow_copies,
            swap_out_blocks=self._swap_out,
            swap_in_blocks=self._swap_in,
            swap_in_cache_hits=self._swap_in_hits,
        )

    def check_invariants(self) -> None:
        """Validate ref counts against block tables. O(total blocks); for tests and debugging."""
        self.pool.check_invariants()
        refs = [0] * self.pool.num_blocks
        for rid, req in self._reqs.items():
            if len(req.table) != cdiv(req.num_tokens, self.block_size):
                raise AssertionError(
                    f"{rid!r}: {len(req.table)} blocks for {req.num_tokens} tokens"
                )
            if len(req.hashes) > req.num_tokens // self.block_size:
                raise AssertionError(f"{rid!r}: hashes cover more than its full blocks")
            for b in req.table:
                refs[b] += 1
        for block in self.pool.blocks:
            if block.ref_cnt != refs[block.block_id]:
                raise AssertionError(f"{block}: {refs[block.block_id]} table references")
        if self.cpu_pool is not None:
            self.cpu_pool.check_invariants()
            cpu_refs = sum(len(s.cpu_blocks) for s in self._swapped.values())
            if cpu_refs != self.cpu_pool.num_blocks - self.cpu_pool.num_free:
                raise AssertionError("CPU pool usage disagrees with swapped requests")


__all__ = ["NO_HIT", "Allocation", "KVCacheManager", "KVStats", "OutOfBlocksError", "PrefixHit"]
