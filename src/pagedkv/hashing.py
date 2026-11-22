"""Content hashes for prefix caching.

A block's hash covers its own token ids *and* its parent's hash, so equal hashes
mean the whole prefix up to and including that block is identical. This is what
lets a lookup stop at the first miss.

SHA-256 rather than Python's ``hash()``: a 64-bit non-cryptographic hash lets one
tenant craft a prompt whose block collides with another tenant's and read or
poison their cached K/V (vLLM switched its default for this reason). An optional
per-tenant ``cache_salt`` is mixed into the first block, so tenants with
different salts never share cache entries even for identical prompts.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence

BlockHash = bytes
HASH_BYTES = 16
_ROOT = b"\x00" * HASH_BYTES


def hash_block(parent: BlockHash | None, token_ids: Sequence[int], extra: bytes = b"") -> BlockHash:
    h = hashlib.sha256()
    h.update(parent if parent is not None else _ROOT)
    h.update(len(token_ids).to_bytes(4, "little"))
    h.update(array("q", token_ids).tobytes())
    h.update(extra)
    return h.digest()[:HASH_BYTES]


def hash_blocks(
    token_ids: Sequence[int],
    block_size: int,
    *,
    start: int = 0,
    parent: BlockHash | None = None,
    cache_salt: bytes | None = None,
) -> list[BlockHash]:
    """Hashes of the full blocks of ``token_ids`` from block index ``start`` on.

    ``parent`` must be the hash of block ``start - 1`` when ``start > 0``.
    """
    if start > 0 and parent is None:
        raise ValueError("parent hash required when start > 0")
    out: list[BlockHash] = []
    for i in range(start, len(token_ids) // block_size):
        extra = cache_salt if (i == 0 and cache_salt) else b""
        parent = hash_block(parent, token_ids[i * block_size : (i + 1) * block_size], extra)
        out.append(parent)
    return out
