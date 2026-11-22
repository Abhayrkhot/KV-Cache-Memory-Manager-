"""Allocator-level simulator: paged vs. reserve-up-front KV allocation."""

from pagedkv.sim.allocators import MemorySnapshot, PagedAllocator, ReservedAllocator, Seq
from pagedkv.sim.simulate import SimResult, simulate
from pagedkv.sim.workload import Request, from_lengths, load_lengths, shared_prefix

__all__ = [
    "MemorySnapshot",
    "PagedAllocator",
    "Request",
    "ReservedAllocator",
    "Seq",
    "SimResult",
    "from_lengths",
    "load_lengths",
    "shared_prefix",
    "simulate",
]
