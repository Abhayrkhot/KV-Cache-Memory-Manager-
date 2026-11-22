"""Request streams for the allocator simulator."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Request:
    rid: int
    prompt_len: int
    output_len: int
    prompt_ids: list[int] = field(default_factory=list)
    """Token ids, only needed when the prefix cache is being exercised."""
    n: int = 1
    """Parallel samples sharing this prompt (forked after prefill)."""

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len


def load_lengths(path: str | Path) -> list[tuple[int, int]]:
    data = json.loads(Path(path).read_text())
    return [(int(p), int(o)) for p, o in data["pairs"]]


def from_lengths(
    pairs: Sequence[tuple[int, int]], num: int, seed: int = 0, n: int = 1
) -> list[Request]:
    """Sample ``num`` requests (with replacement only if ``num`` exceeds the pool)."""
    rng = random.Random(seed)
    picked = rng.sample(list(pairs), num) if num <= len(pairs) else rng.choices(list(pairs), k=num)
    reqs = []
    for i, (p, o) in enumerate(picked):
        ids = [rng.randrange(1, 32000) for _ in range(p)]
        reqs.append(Request(i, p, o, ids, n))
    return reqs


def shared_prefix(
    num: int,
    system_len: int,
    user_lens: Sequence[tuple[int, int]],
    num_system_prompts: int = 1,
    seed: int = 0,
) -> list[Request]:
    """Requests that start with one of ``num_system_prompts`` shared system prompts.

    ``user_lens`` supplies (user turn length, output length) pairs, e.g. from ShareGPT.
    """
    rng = random.Random(seed)
    systems = [
        [rng.randrange(1, 32000) for _ in range(system_len)] for _ in range(num_system_prompts)
    ]
    reqs = []
    for i in range(num):
        u, o = user_lens[rng.randrange(len(user_lens))]
        ids = systems[rng.randrange(num_system_prompts)] + [
            rng.randrange(1, 32000) for _ in range(u)
        ]
        reqs.append(Request(i, len(ids), o, ids))
    return reqs
