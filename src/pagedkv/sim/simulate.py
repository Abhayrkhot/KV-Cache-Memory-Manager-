"""Iteration-level scheduling simulator for comparing KV allocators.

Each step models one forward pass, scheduled the way vLLM and Orca do it:

1. every running sequence needs one more KV slot (for the token it produced last
   step); if the paged allocator runs out, the most recently admitted sequence is
   preempted (its blocks are freed and it is recomputed later, as in vLLM);
2. waiting requests are admitted FCFS while ``max_num_seqs`` and memory allow; an
   admitted request is prefilled and emits its first token in the same step;
3. sequences that have produced ``output_len`` tokens finish and free memory.

There is no timing model: the simulator measures how many sequences each
allocator lets the engine batch and where the memory goes, which is what limits
throughput when decoding is memory-bound. Wall-clock numbers come from the GPU
benchmarks.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

from pagedkv.sim.allocators import Allocator, MemorySnapshot, Seq
from pagedkv.sim.workload import Request


@dataclass
class SimResult:
    allocator: str
    steps: int
    finished_requests: int
    output_tokens: int
    mean_batch_saturated: float
    """Mean running sequences over steps where memory, not max_num_seqs or the queue, was the limit."""
    mean_batch: float
    peak_batch: int
    breakdown: dict[str, float]
    """Mean share of KV capacity per category over saturated steps."""
    preemptions: int
    recomputed_tokens: int
    prefill_tokens: int
    prefill_tokens_skipped: int
    saturated_steps: int
    timeline: list[tuple[int, int, float]] = field(default_factory=list)
    """(step, running sequences, token-state share) sampled every ``timeline_every`` steps."""

    @property
    def output_tokens_per_step(self) -> float:
        return self.output_tokens / self.steps if self.steps else 0.0


def simulate(
    requests: list[Request],
    alloc: Allocator,
    *,
    max_num_seqs: int = 256,
    seed: int = 0,
    timeline_every: int = 10,
) -> SimResult:
    rng = random.Random(seed)
    next_sid = 0

    def new_seq(rid: int, ids: list[int], output_len: int, generated: int = 0) -> Seq:
        nonlocal next_sid
        next_sid += 1
        prompt_len = len(ids) - generated
        return Seq(
            next_sid, rid, list(ids), len(ids), prompt_len + output_len - 1, generated, output_len
        )

    waiting: deque[tuple[Seq, Request | None]] = deque()
    for r in requests:
        if r.prompt_len + r.output_len - 1 > alloc.capacity:
            raise ValueError(f"request {r.rid} needs more KV than the whole pool")
        waiting.append((new_seq(r.rid, r.prompt_ids or [0] * r.prompt_len, r.output_len), r))

    running: list[Seq] = []
    step = finished = output_tokens = preemptions = recomputed = prefill = 0
    batch_sum = sat_batch_sum = sat_steps = peak = 0
    sums: dict[str, float] = dict.fromkeys(
        ("token_states", "reserved", "internal_frag", "external_frag", "free"), 0.0
    )
    timeline: list[tuple[int, int, float]] = []
    remaining_requests = {r.rid: r.n for r in requests}

    while waiting or running:
        step += 1
        # 1. decode growth, preempting from the back when memory runs out
        i = 0
        while i < len(running):
            seq = running[i]
            if alloc.grow(seq):
                i += 1
                continue
            victim = running.pop()
            alloc.free(victim)
            preemptions += 1
            resumed = new_seq(victim.rid, victim.ids, victim.output_len, victim.generated)
            waiting.appendleft((resumed, None))
            # victim may be `seq` itself, in which case index i now points past the end

        # 2. admission
        blocked: Seq | None = None
        while waiting and len(running) < max_num_seqs:
            seq, req = waiting[0]
            skipped_before = getattr(alloc, "prefill_skipped", 0)
            if not alloc.admit(seq):
                blocked = seq
                break
            hit = getattr(alloc, "prefill_skipped", 0) - skipped_before
            waiting.popleft()
            seq.admit_step = step
            running.append(seq)
            prefill += seq.kv_len
            if seq.generated:  # resumed after preemption; prefix-cache hits are not recomputed
                recomputed += seq.kv_len - hit
            if req is not None and req.n > 1:
                for _ in range(req.n - 1):
                    child = new_seq(seq.rid, seq.ids, seq.output_len)
                    if len(running) < max_num_seqs and alloc.fork(seq, child):
                        running.append(child)
                    else:
                        waiting.appendleft((child, None))
        if not running and blocked is not None:
            raise RuntimeError("a waiting request cannot be admitted into an empty pool")

        # 3. account
        snap: MemorySnapshot = alloc.snapshot(running, blocked)
        batch_sum += len(running)
        peak = max(peak, len(running))
        if blocked is not None:  # memory was the binding constraint this step
            sat_steps += 1
            sat_batch_sum += len(running)
            cap = snap.capacity
            used = snap.token_states + snap.reserved + snap.internal_frag + snap.external_frag
            sums["token_states"] += snap.token_states / cap
            sums["reserved"] += snap.reserved / cap
            sums["internal_frag"] += snap.internal_frag / cap
            sums["external_frag"] += snap.external_frag / cap
            sums["free"] += max(0, cap - used) / cap
        if step % timeline_every == 0:
            timeline.append((step, len(running), snap.token_states / snap.capacity))

        # 4. every running sequence emits one token; finished ones release memory
        still: list[Seq] = []
        for seq in running:
            seq.generated += 1
            seq.ids.append(rng.randrange(1, 32000))
            output_tokens += 1
            if seq.generated >= seq.output_len:
                alloc.free(seq)
                remaining_requests[seq.rid] -= 1
                if remaining_requests[seq.rid] == 0:
                    finished += 1
            else:
                still.append(seq)
        running = still

    skipped = int(getattr(alloc, "prefill_skipped", 0))
    return SimResult(
        allocator=alloc.name,
        steps=step,
        finished_requests=finished,
        output_tokens=output_tokens,
        mean_batch_saturated=sat_batch_sum / sat_steps if sat_steps else 0.0,
        mean_batch=batch_sum / step if step else 0.0,
        peak_batch=peak,
        breakdown={k: v / sat_steps for k, v in sums.items()} if sat_steps else sums,
        preemptions=preemptions,
        recomputed_tokens=recomputed,
        prefill_tokens=prefill,
        prefill_tokens_skipped=skipped,
        saturated_steps=sat_steps,
        timeline=timeline,
    )
