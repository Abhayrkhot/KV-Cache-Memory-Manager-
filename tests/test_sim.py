from __future__ import annotations

import random

import pytest

from pagedkv.sim import (
    PagedAllocator,
    Request,
    ReservedAllocator,
    from_lengths,
    shared_prefix,
    simulate,
)
from pagedkv.sim.allocators import Seq


def _pairs(n: int = 400, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(8, 300), rng.randint(4, 300)) for _ in range(n)]


def test_first_fit_merges_free_neighbours() -> None:
    a = ReservedAllocator(100, "oracle", 100)
    seqs = [Seq(i, i, [0] * 10, 10, 20) for i in range(5)]
    for s in seqs:
        assert a.admit(s)
    assert a.free_slots() == 0
    a.free(seqs[1])
    a.free(seqs[3])
    assert not a.admit(Seq(9, 9, [0] * 30, 30, 40))  # 40 free, but in two 20-slot gaps
    a.free(seqs[2])  # 1, 2, 3 merge into one 60-slot gap
    assert a.admit(Seq(9, 9, [0] * 30, 30, 40))


def test_reservation_policies() -> None:
    s = Seq(1, 1, [0] * 100, 100, 299)
    assert ReservedAllocator(10_000, "max", 2048).reservation(s) == 2048
    assert ReservedAllocator(10_000, "pow2", 2048).reservation(s) == 512
    assert ReservedAllocator(10_000, "oracle", 2048).reservation(s) == 299


@pytest.mark.parametrize(
    "make",
    [
        lambda: PagedAllocator(4000 // 16, 16),
        lambda: ReservedAllocator(4000, "max", 600),
        lambda: ReservedAllocator(4000, "pow2", 600),
        lambda: ReservedAllocator(4000, "oracle", 600),
    ],
)
def test_every_request_finishes_with_exact_output(make) -> None:  # type: ignore[no-untyped-def]
    reqs = from_lengths(_pairs(), 150, seed=3)
    r = simulate(reqs, make(), max_num_seqs=64)
    assert r.finished_requests == 150
    assert r.output_tokens == sum(q.output_len for q in reqs)


def test_paged_batches_more_than_any_reservation_policy() -> None:
    results = {}
    for alloc in (
        PagedAllocator(4000 // 16, 16),
        ReservedAllocator(4000, "max", 600),
        ReservedAllocator(4000, "pow2", 600),
        ReservedAllocator(4000, "oracle", 600),
    ):
        results[alloc.name] = simulate(from_lengths(_pairs(), 200, seed=5), alloc, max_num_seqs=256)
    paged = results["paged-16"]
    assert paged.breakdown["external_frag"] == 0
    assert paged.breakdown["reserved"] == 0
    for name, r in results.items():
        if name != "paged-16":
            assert paged.mean_batch_saturated > r.mean_batch_saturated, name
            assert paged.steps < r.steps, name


def test_tight_pool_forces_preemption_but_completes() -> None:
    reqs = [Request(i, 100, 200) for i in range(20)]
    r = simulate(reqs, PagedAllocator(64, 16), max_num_seqs=32)  # 1024 slots, peak need 299 each
    assert r.preemptions > 0
    assert r.recomputed_tokens > 0
    assert r.recomputed_tokens <= r.prefill_tokens - r.prefill_tokens_skipped
    assert r.finished_requests == 20


def test_parallel_sampling_shares_prompt_blocks() -> None:
    reqs = from_lengths([(600, 50)] * 10, 10, n=4)
    paged = simulate(reqs, PagedAllocator(6000 // 16, 16), max_num_seqs=256)
    oracle = simulate(
        from_lengths([(600, 50)] * 10, 10, n=4), ReservedAllocator(6000, "oracle", 1024)
    )
    assert paged.output_tokens == oracle.output_tokens == 10 * 4 * 50
    assert paged.mean_batch_saturated > 2 * oracle.mean_batch_saturated


def test_shared_system_prompt_hits_prefix_cache() -> None:
    reqs = shared_prefix(100, system_len=512, user_lens=[(64, 32)], seed=1)
    cached = PagedAllocator(4000 // 16, 16)
    r = simulate(reqs, cached)
    assert r.prefill_tokens_skipped >= 90 * 512
    nocache = simulate(
        shared_prefix(100, 512, [(64, 32)], seed=1), PagedAllocator(250, 16, prefix_caching=False)
    )
    assert nocache.prefill_tokens_skipped == 0
    assert r.steps <= nocache.steps


def test_oversized_request_is_rejected() -> None:
    with pytest.raises(ValueError, match="whole pool"):
        simulate([Request(0, 1000, 1000)], PagedAllocator(10, 16))
