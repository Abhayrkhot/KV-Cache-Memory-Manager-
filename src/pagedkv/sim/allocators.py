"""KV allocators compared by the simulator.

``PagedAllocator`` wraps ``KVCacheManager``. ``ReservedAllocator`` is the
pre-PagedAttention design (FasterTransformer, Orca): each sequence gets one
contiguous region sized up front, placed first-fit in a flat address space.
Its three policies follow the vLLM paper (Kwon et al., SOSP 2023, section 6.1):

* ``max``:    reserve ``max_model_len`` tokens (what systems that cannot predict output length do)
* ``pow2``:   reserve the true final length rounded up to a power of two
* ``oracle``: reserve exactly the true final length (an unrealistic lower bound)
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pagedkv.manager import KVCacheManager


@dataclass
class Seq:
    sid: int
    rid: int
    ids: list[int]
    """Prompt plus generated token ids (KV exists for the first ``kv_len`` of them)."""
    kv_len: int
    peak_len: int
    """KV length when this sequence finishes: prompt + output - 1."""
    generated: int = 0
    output_len: int = 0
    admit_step: int = 0
    children: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class MemorySnapshot:
    """Slot accounting for one step; all values in tokens (KV slots)."""

    capacity: int
    token_states: int
    """Slots holding live K/V (shared slots counted once)."""
    reserved: int
    """Slots set aside for tokens a sequence has not produced yet."""
    internal_frag: int
    """Allocated slots that will never hold a token (over-reservation, partial last block)."""
    external_frag: int
    """Free slots that could not be used for the next waiting request (gaps)."""
    cached_free: int = 0
    """Free slots still holding evictable prefix-cache content."""


class Allocator(Protocol):
    name: str
    capacity: int

    def admit(self, seq: Seq) -> bool: ...
    def fork(self, parent: Seq, child: Seq) -> bool: ...
    def grow(self, seq: Seq) -> bool: ...
    def free(self, seq: Seq) -> None: ...
    def snapshot(self, live: list[Seq], blocked: Seq | None) -> MemorySnapshot: ...


class PagedAllocator:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        prefix_caching: bool = True,
        watermark: float = 0.0,
    ) -> None:
        self.kv = KVCacheManager(
            num_blocks, block_size, enable_prefix_caching=prefix_caching, watermark=watermark
        )
        self.block_size = block_size
        self.capacity = num_blocks * block_size
        self.name = f"paged-{block_size}" + ("" if prefix_caching else "-nocache")
        self.prefill_skipped = 0

    def admit(self, seq: Seq) -> bool:
        hit = self.kv.find_prefix(seq.ids[: seq.kv_len])
        if self.kv.allocate(seq.sid, seq.kv_len - hit.num_tokens, prefix=hit, admit=True) is None:
            return False
        self.prefill_skipped += hit.num_tokens
        self.kv.commit(seq.sid, seq.ids)
        return True

    def fork(self, parent: Seq, child: Seq) -> bool:
        self.kv.fork(parent.sid, child.sid)
        return True

    def grow(self, seq: Seq) -> bool:
        if self.kv.allocate(seq.sid, 1) is None:
            return False
        seq.kv_len += 1
        self.kv.commit(seq.sid, seq.ids)
        return True

    def free(self, seq: Seq) -> None:
        self.kv.free(seq.sid)

    def snapshot(self, live: list[Seq], blocked: Seq | None) -> MemorySnapshot:
        bs = self.block_size
        fill: dict[int, int] = {}
        for s in live:
            table = self.kv.block_table(s.sid)
            if table:
                last = table[-1]
                fill[last] = max(fill.get(last, 0), s.kv_len - (len(table) - 1) * bs)
        st = self.kv.stats()
        internal = sum(bs - f for f in fill.values())
        return MemorySnapshot(
            capacity=self.capacity,
            token_states=st.used_blocks * bs - internal,
            reserved=0,
            internal_frag=internal,
            external_frag=0,
            cached_free=st.cached_free_blocks * bs,
        )


Policy = Literal["max", "pow2", "oracle"]


class ReservedAllocator:
    def __init__(self, capacity: int, policy: Policy, max_model_len: int) -> None:
        self.capacity = capacity
        self.policy = policy
        self.max_model_len = max_model_len
        self.name = f"reserve-{policy}"
        self._free: list[tuple[int, int]] = [(0, capacity)]  # (start, length), sorted by start
        self._regions: dict[int, tuple[int, int]] = {}

    def reservation(self, seq: Seq) -> int:
        if self.policy == "max":
            return self.max_model_len
        if self.policy == "pow2":
            return 1 << max(0, seq.peak_len - 1).bit_length()
        return seq.peak_len

    def _take(self, n: int) -> int | None:
        for i, (start, length) in enumerate(self._free):
            if length >= n:
                if length == n:
                    del self._free[i]
                else:
                    self._free[i] = (start + n, length - n)
                return start
        return None

    def _give(self, start: int, n: int) -> None:
        i = bisect.bisect(self._free, (start, n))
        self._free.insert(i, (start, n))
        # merge with neighbours
        if i + 1 < len(self._free) and start + n == self._free[i + 1][0]:
            self._free[i] = (start, n + self._free[i + 1][1])
            del self._free[i + 1]
        if i > 0 and self._free[i - 1][0] + self._free[i - 1][1] == self._free[i][0]:
            self._free[i - 1] = (self._free[i - 1][0], self._free[i - 1][1] + self._free[i][1])
            del self._free[i]

    def admit(self, seq: Seq) -> bool:
        need = self.reservation(seq)
        if need < seq.peak_len:
            raise ValueError(f"sequence of {seq.peak_len} tokens exceeds max_model_len")
        start = self._take(need)
        if start is None:
            return False
        self._regions[seq.sid] = (start, need)
        return True

    def fork(self, parent: Seq, child: Seq) -> bool:
        # Without paging a sample cannot share its parent's prompt: it needs its own copy.
        return self.admit(child)

    def grow(self, seq: Seq) -> bool:
        seq.kv_len += 1
        if seq.kv_len > self._regions[seq.sid][1]:
            raise AssertionError("sequence outgrew its reservation")
        return True

    def free(self, seq: Seq) -> None:
        start, n = self._regions.pop(seq.sid)
        self._give(start, n)

    def free_slots(self) -> int:
        return sum(n for _, n in self._free)

    def snapshot(self, live: list[Seq], blocked: Seq | None) -> MemorySnapshot:
        used = sum(s.kv_len for s in live)
        future = sum(s.peak_len - s.kv_len for s in live)
        internal = sum(self._regions[s.sid][1] - s.peak_len for s in live)
        free = self.free_slots()
        external = 0
        if blocked is not None and free >= self.reservation(blocked):
            external = free  # enough memory in total, but no single gap is large enough
        return MemorySnapshot(self.capacity, used, future, internal, external)
