"""Physical KV blocks: reference counts, an LRU free queue, and the prefix-cache index.

A block with ``ref_cnt == 0`` is free, but if it still holds a cached prefix
(``block_hash`` is set) it can be revived by a later request instead of being
recomputed. Free blocks are handed out from the front of the queue:

* blocks with no cached content are moved to the front, since reusing them costs
  nothing;
* cached blocks are appended to the back as they are released, so the least
  recently released cached block is evicted first (LRU).

All operations are O(1) per block.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from pagedkv.hashing import BlockHash


class Block:
    __slots__ = ("block_hash", "block_id", "ref_cnt")

    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash: BlockHash | None = None

    def __repr__(self) -> str:
        cached = "cached" if self.block_hash is not None else "-"
        return f"Block({self.block_id}, ref={self.ref_cnt}, {cached})"


class OutOfBlocksError(RuntimeError):
    pass


class BlockPool:
    def __init__(self, num_blocks: int, enable_caching: bool = True) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.enable_caching = enable_caching
        self.blocks = [Block(i) for i in range(num_blocks)]
        self._free: OrderedDict[int, None] = OrderedDict.fromkeys(range(num_blocks))
        self._cached: dict[BlockHash, int] = {}
        self.num_evictions = 0

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_cached(self) -> int:
        return len(self._cached)

    def num_cached_free(self) -> int:
        """Free blocks that still hold a reusable prefix (evictable cache)."""
        return sum(1 for b in self._free if self.blocks[b].block_hash is not None)

    def lookup(self, block_hash: BlockHash) -> int | None:
        return self._cached.get(block_hash) if self.enable_caching else None

    def allocate(self, n: int) -> list[int]:
        """Take ``n`` blocks off the free queue, evicting cached content if needed."""
        if n > len(self._free):
            raise OutOfBlocksError(f"need {n} blocks, {len(self._free)} free")
        out = []
        for _ in range(n):
            block_id, _ = self._free.popitem(last=False)
            block = self.blocks[block_id]
            if block.block_hash is not None:
                del self._cached[block.block_hash]
                block.block_hash = None
                self.num_evictions += 1
            block.ref_cnt = 1
            out.append(block_id)
        return out

    def touch(self, block_ids: Iterable[int]) -> None:
        """Add a reference to each block, pulling free (cached) ones off the free queue."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ref_cnt == 0:
                del self._free[block_id]
            block.ref_cnt += 1

    def release(self, block_ids: Iterable[int]) -> None:
        """Drop one reference from each block; pass them in the order they should be evicted."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ref_cnt <= 0:
                raise AssertionError(f"block {block_id} released with ref_cnt={block.ref_cnt}")
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._free[block_id] = None
                if block.block_hash is None:
                    self._free.move_to_end(block_id, last=False)

    def register(self, block_id: int, block_hash: BlockHash) -> bool:
        """Publish a full, computed block under ``block_hash``. Returns False if the hash is taken."""
        if not self.enable_caching:
            return False
        block = self.blocks[block_id]
        if block.block_hash is not None:
            return block.block_hash == block_hash
        if block_hash in self._cached:
            return False  # same content already cached in another block; keep that one
        block.block_hash = block_hash
        self._cached[block_hash] = block_id
        return True

    def reset_cache(self) -> int:
        """Drop every cached prefix (e.g. after a weight update). Returns blocks dropped.

        Only allowed when no block is in use: a running request's blocks hold K/V
        computed with the old weights and are still indexed, so a partial reset would
        let new requests hit stale entries (vLLM refuses in the same situation).
        """
        if len(self._free) != len(self.blocks):
            raise RuntimeError("cannot reset the prefix cache while blocks are in use")
        dropped = 0
        for block_id in self._free:
            block = self.blocks[block_id]
            if block.block_hash is not None:
                del self._cached[block.block_hash]
                block.block_hash = None
                dropped += 1
        return dropped

    def check_invariants(self) -> None:
        free = set(self._free)
        if len(free) != len(self._free):
            raise AssertionError("duplicate entries in the free queue")
        for block in self.blocks:
            if block.ref_cnt < 0:
                raise AssertionError(f"{block} has a negative ref count")
            if (block.ref_cnt == 0) != (block.block_id in free):
                raise AssertionError(f"{block} free-queue membership disagrees with ref count")
            if (
                block.block_hash is not None
                and self._cached.get(block.block_hash) != block.block_id
            ):
                raise AssertionError(f"{block} hash is not indexed to it")
        for h, block_id in self._cached.items():
            if self.blocks[block_id].block_hash != h:
                raise AssertionError(f"cache index entry for block {block_id} is stale")
