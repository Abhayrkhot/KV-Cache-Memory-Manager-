"""Block manager tests.

Besides unit tests, a Hypothesis state machine drives random sequences of
admit / decode / fork / free / swap operations against a simulated executor that
stores, for every physical slot, which token's K/V was written there. After every
step each live request must read back exactly its own tokens through its block
table. That one check covers copy-on-write, prefix reuse, eviction and swapping:
any of them going wrong shows up as a request seeing someone else's K/V.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, precondition, rule

from pagedkv.block_pool import BlockPool, OutOfBlocksError
from pagedkv.hashing import hash_blocks
from pagedkv.manager import KVCacheManager

BS = 4


def admit(kv: KVCacheManager, rid: str, tokens: list[int], salt: bytes | None = None) -> int:
    hit = kv.find_prefix(tokens, salt)
    alloc = kv.allocate(rid, len(tokens) - hit.num_tokens, prefix=hit, cache_salt=salt)
    assert alloc is not None
    kv.commit(rid, tokens)
    return hit.num_tokens


# --------------------------------------------------------------------- BlockPool


def test_pool_allocate_release_roundtrip() -> None:
    pool = BlockPool(4)
    ids = pool.allocate(3)
    assert pool.num_free == 1
    pool.release(ids)
    assert pool.num_free == 4
    pool.check_invariants()
    with pytest.raises(OutOfBlocksError):
        pool.allocate(5)


def test_pool_double_release_raises() -> None:
    pool = BlockPool(2)
    (b,) = pool.allocate(1)
    pool.release([b])
    with pytest.raises(AssertionError):
        pool.release([b])


def test_pool_evicts_uncached_before_cached_and_lru_among_cached() -> None:
    pool = BlockPool(4)
    a, b, c, _d = pool.allocate(4)
    pool.register(a, b"A" * 16)
    pool.register(b, b"B" * 16)
    pool.release([a])  # cached, released first -> oldest
    pool.release([c])  # uncached -> front
    pool.release([b])  # cached, newer
    assert pool.allocate(1) == [c]
    assert pool.allocate(1) == [a]  # LRU cached block goes next
    assert pool.lookup(b"A" * 16) is None
    assert pool.lookup(b"B" * 16) == b
    assert pool.num_evictions == 1


def test_pool_register_duplicate_content_keeps_first() -> None:
    pool = BlockPool(2)
    a, b = pool.allocate(2)
    assert pool.register(a, b"H" * 16)
    assert not pool.register(b, b"H" * 16)
    assert pool.lookup(b"H" * 16) == a


# ----------------------------------------------------------------- KVCacheManager


def test_allocates_blocks_lazily() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=BS)
    assert kv.allocate("r", 5) is not None
    assert len(kv.block_table("r")) == 2
    kv.allocate("r", 3)  # fills block 2 exactly
    assert len(kv.block_table("r")) == 2
    kv.allocate("r", 1)
    assert len(kv.block_table("r")) == 3
    kv.check_invariants()


def test_allocate_returns_none_without_side_effects_when_full() -> None:
    kv = KVCacheManager(num_blocks=2, block_size=BS)
    assert kv.allocate("a", 8) is not None
    before = kv.stats()
    assert kv.allocate("b", 1) is None
    assert "b" not in kv
    assert kv.stats() == before


def test_watermark_only_applies_to_admission() -> None:
    kv = KVCacheManager(num_blocks=10, block_size=BS, watermark=0.2)  # 2 blocks reserved
    assert kv.allocate("a", 8 * BS, admit=True) is not None
    assert kv.allocate("b", 1, admit=True) is None
    assert kv.allocate("a", 2 * BS) is not None  # running request may use the reserve


def test_prefix_hit_reuses_blocks_and_never_covers_last_token() -> None:
    kv = KVCacheManager(num_blocks=16, block_size=BS)
    prompt = list(range(12))  # 3 full blocks
    assert admit(kv, "a", prompt) == 0
    kv.free("a")
    # Same prompt: only 2 blocks may be reused, the last token must be recomputed.
    hit = kv.find_prefix(prompt)
    assert hit.num_tokens == 8
    assert admit(kv, "b", [*prompt, 99]) == 12
    assert kv.stats().prefix_hit_tokens == 12
    kv.check_invariants()


def test_prefix_cache_distinguishes_parent_context() -> None:
    kv = KVCacheManager(num_blocks=16, block_size=BS)
    admit(kv, "a", [1, 2, 3, 4, 5, 6, 7, 8, 0])
    # Second block has the same tokens but a different first block: no hit at all.
    assert kv.find_prefix([9, 9, 9, 9, 5, 6, 7, 8, 0]).num_tokens == 0
    assert kv.find_prefix([1, 2, 3, 4, 5, 6, 7, 8, 0]).num_tokens == 8


def test_cache_salt_isolates_tenants() -> None:
    kv = KVCacheManager(num_blocks=16, block_size=BS)
    prompt = list(range(9))
    admit(kv, "a", prompt, salt=b"tenant-a")
    assert kv.find_prefix(prompt, b"tenant-b").num_tokens == 0
    assert kv.find_prefix(prompt).num_tokens == 0
    assert kv.find_prefix(prompt, b"tenant-a").num_tokens == 8


def test_disabled_prefix_caching_never_hits() -> None:
    kv = KVCacheManager(num_blocks=16, block_size=BS, enable_prefix_caching=False)
    admit(kv, "a", list(range(9)))
    kv.free("a")
    assert kv.find_prefix(list(range(9))).num_tokens == 0


def test_revived_cached_blocks_count_against_free_memory() -> None:
    kv = KVCacheManager(num_blocks=3, block_size=BS)
    admit(kv, "a", list(range(9)))  # 3 blocks, 2 full + cached
    kv.free("a")
    hit = kv.find_prefix(list(range(9)))
    assert hit.num_tokens == 8
    # 2 revived + 1 new = 3 blocks: fits exactly.
    assert kv.blocks_needed(None, 1, hit) == 3
    assert kv.allocate("b", 1, prefix=hit) is not None
    assert kv.stats().free_blocks == 0


def test_fork_copy_on_write_copies_once() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=BS)
    kv.allocate("p", 6)  # block 0 full, block 1 has 2 tokens
    kv.fork("p", "c")
    shared = kv.block_table("p")
    a = kv.allocate("p", 1)
    assert a is not None
    assert len(a.copies) == 1
    assert a.copies[0][0] == shared[1]
    assert kv.block_table("p")[0] == shared[0]  # full block still shared
    b = kv.allocate("c", 1)
    assert b is not None
    assert b.copies == ()  # child now owns the original block alone
    assert kv.stats().cow_copies == 1
    kv.check_invariants()


def test_fork_at_block_boundary_needs_no_copy() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=BS)
    kv.allocate("p", 8)
    kv.fork("p", "c")
    a = kv.allocate("p", 1)
    assert a is not None
    assert a.copies == ()


def test_swap_out_and_in_restores_table_and_reuses_cached_blocks() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=BS, num_cpu_blocks=8)
    tokens = list(range(10))
    admit(kv, "a", tokens)
    out = kv.swap_out("a")
    assert out is not None
    assert len(out) == 3
    assert kv.is_swapped("a")
    assert kv.stats().free_blocks == 8
    back = kv.swap_in("a")
    assert back is not None
    # The two full blocks were still cached on the GPU: only the partial one is copied.
    assert len(back) == 1
    assert kv.stats().swap_in_cache_hits == 2
    assert kv.num_tokens("a") == 10
    kv.check_invariants()


def test_swap_in_returns_none_when_gpu_full() -> None:
    kv = KVCacheManager(num_blocks=3, block_size=BS, num_cpu_blocks=4, enable_prefix_caching=False)
    kv.allocate("a", 12)
    assert kv.swap_out("a") is not None
    kv.allocate("b", 12)
    assert not kv.can_swap_in("a")
    assert kv.swap_in("a") is None
    kv.free("b")
    assert kv.swap_in("a") is not None


def test_slot_mapping() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=BS)
    kv.allocate("r", 6)
    t = kv.block_table("r")
    assert kv.slot("r", 0) == t[0] * BS
    assert kv.slot("r", 5) == t[1] * BS + 1
    with pytest.raises(IndexError):
        kv.slot("r", 6)


def test_hash_blocks_incremental_matches_full() -> None:
    toks = list(range(20))
    full = hash_blocks(toks, BS)
    tail = hash_blocks(toks, BS, start=2, parent=full[1])
    assert full[2:] == tail


# ------------------------------------------------------------- stateful property


class ManagerMachine(RuleBasedStateMachine):
    """Random workloads against a simulated KV store; every request must read back its own tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.kv = KVCacheManager(num_blocks=12, block_size=BS, num_cpu_blocks=10)
        self.gpu: dict[int, int] = {}  # slot -> token whose K/V lives there
        self.cpu: dict[int, int] = {}
        self.tokens: dict[str, list[int]] = {}  # every live request's full token sequence
        self.n = 0

    def _copy(self, src_store: dict[int, int], dst_store: dict[int, int], pairs) -> None:  # type: ignore[no-untyped-def]
        for src, dst in pairs:
            for off in range(BS):
                key = src * BS + off
                if key in src_store:
                    dst_store[dst * BS + off] = src_store[key]
                else:
                    dst_store.pop(dst * BS + off, None)

    def _write(self, rid: str, start: int, end: int) -> None:
        for pos in range(start, end):
            self.gpu[self.kv.slot(rid, pos)] = self.tokens[rid][pos]

    def _running(self) -> list[str]:
        return [r for r in self.tokens if r in self.kv]

    def _swapped(self) -> list[str]:
        return [r for r in self.tokens if self.kv.is_swapped(r)]

    @rule(
        prompt=st.lists(st.integers(0, 2), min_size=1, max_size=14),
        salt=st.sampled_from([None, None, b"t1"]),
    )
    def admit(self, prompt: list[int], salt: bytes | None) -> None:
        rid = f"r{self.n}"
        self.n += 1
        hit = self.kv.find_prefix(prompt, salt)
        alloc = self.kv.allocate(
            rid, len(prompt) - hit.num_tokens, prefix=hit, cache_salt=salt, admit=True
        )
        if alloc is None:
            return
        self.tokens[rid] = list(prompt)
        self._copy(self.gpu, self.gpu, alloc.copies)
        self._write(rid, hit.num_tokens, len(prompt))
        self.kv.commit(rid, prompt)

    @precondition(lambda self: bool(self._running()))
    @rule(data=st.data(), tok=st.integers(0, 2))
    def decode(self, data: st.DataObject, tok: int) -> None:
        rid = data.draw(st.sampled_from(self._running()))
        n = self.kv.num_tokens(rid)
        alloc = self.kv.allocate(rid, 1)
        if alloc is None:
            return
        self.tokens[rid].append(tok)
        self._copy(self.gpu, self.gpu, alloc.copies)
        self._write(rid, n, n + 1)
        self.kv.commit(rid, self.tokens[rid])

    @precondition(lambda self: bool(self._running()))
    @rule(data=st.data())
    def fork(self, data: st.DataObject) -> None:
        parent = data.draw(st.sampled_from(self._running()))
        child = f"r{self.n}"
        self.n += 1
        self.kv.fork(parent, child)
        self.tokens[child] = list(self.tokens[parent])

    @precondition(lambda self: bool(self.tokens))
    @rule(data=st.data())
    def finish(self, data: st.DataObject) -> None:
        rid = data.draw(st.sampled_from(sorted(self.tokens)))
        self.kv.free(rid)
        del self.tokens[rid]

    @precondition(lambda self: bool(self._running()))
    @rule(data=st.data())
    def swap_out(self, data: st.DataObject) -> None:
        rid = data.draw(st.sampled_from(self._running()))
        copies = self.kv.swap_out(rid)
        if copies is not None:
            self._copy(self.gpu, self.cpu, copies)

    @precondition(lambda self: bool(self._swapped()))
    @rule(data=st.data())
    def swap_in(self, data: st.DataObject) -> None:
        rid = data.draw(st.sampled_from(self._swapped()))
        copies = self.kv.swap_in(rid)
        if copies is not None:
            self._copy(self.cpu, self.gpu, copies)

    @invariant()
    def consistent(self) -> None:
        self.kv.check_invariants()
        for rid in self._running():
            n = self.kv.num_tokens(rid)
            assert n == len(self.tokens[rid])
            seen = [self.gpu.get(self.kv.slot(rid, p)) for p in range(n)]
            assert seen == self.tokens[rid], f"{rid} reads wrong K/V"


TestManagerMachine = ManagerMachine.TestCase
TestManagerMachine.settings = settings(
    max_examples=300,
    stateful_step_count=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def test_stale_prefix_hit_is_rejected() -> None:
    kv = KVCacheManager(num_blocks=3, block_size=BS)
    prompt = list(range(9))
    admit(kv, "a", prompt)
    kv.free("a")
    hit = kv.find_prefix(prompt)
    assert hit.num_tokens == 8
    kv.allocate("hog", 3 * BS)  # evicts the cached blocks and reuses them
    kv.free("hog")
    with pytest.raises(ValueError, match="stale prefix"):
        kv.allocate("b", 1, prefix=hit)


def test_reset_cache_refuses_while_blocks_are_in_use() -> None:
    kv = KVCacheManager(num_blocks=8, block_size=BS)
    admit(kv, "a", list(range(9)))
    with pytest.raises(RuntimeError, match="in use"):
        kv.pool.reset_cache()
    kv.free("a")
    assert kv.pool.reset_cache() == 2
    assert kv.find_prefix(list(range(9))).num_tokens == 0
    kv.check_invariants()
