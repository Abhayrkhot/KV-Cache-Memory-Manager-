"""GPU capacity planning: how many sequences fit once the weights are loaded."""

from __future__ import annotations

from dataclasses import dataclass

from pagedkv.spec import KVDtype, KVSpec, get_kv_dtype

GIB = 1024**3

WEIGHT_BITS = {
    "fp32": 32.0,
    "bf16": 16.0,
    "fp16": 16.0,
    "fp8": 8.0,
    "int8": 8.0,
    "int4": 4.0,
    "mxfp4": 4.25,
}
"""Bits per weight. MXFP4 includes its e8m0 scale (one byte per 32 weights)."""


@dataclass(frozen=True)
class CapacityPlan:
    """Memory plan for one GPU.

    ``kv_budget`` is what a serving engine would hand to its KV-cache pool:
    ``gpu_util`` of the device minus this rank's share of the weights minus
    ``overhead_bytes`` for activations, CUDA graphs and allocator slack (vLLM
    measures the last term with a profiling run; here it is an input).
    """

    spec: KVSpec  # already sharded for this rank
    gpu_bytes: float
    weight_bytes: float
    kv_dtype: KVDtype
    gpu_util: float = 0.90
    overhead_bytes: float = 2 * GIB
    block_size: int = 16
    tensor_parallel: int = 1

    @classmethod
    def build(
        cls,
        spec: KVSpec,
        gpu_gib: float,
        num_params: int,
        weight_dtype: str = "bf16",
        kv_dtype: str | KVDtype = "bf16",
        gpu_util: float = 0.90,
        overhead_gib: float = 2.0,
        block_size: int = 16,
        tensor_parallel: int = 1,
    ) -> CapacityPlan:
        if weight_dtype not in WEIGHT_BITS:
            raise ValueError(
                f"unknown weight dtype {weight_dtype!r}; known: {', '.join(WEIGHT_BITS)}"
            )
        if not 0 < gpu_util <= 1:
            raise ValueError("gpu_util must be in (0, 1]")
        if gpu_gib <= 0 or num_params < 0 or overhead_gib < 0:
            raise ValueError("gpu_gib must be positive; num_params and overhead_gib non-negative")
        return cls(
            spec=spec.shard(tensor_parallel),
            gpu_bytes=gpu_gib * GIB,
            weight_bytes=num_params * WEIGHT_BITS[weight_dtype] / 8 / tensor_parallel,
            kv_dtype=get_kv_dtype(kv_dtype),
            gpu_util=gpu_util,
            overhead_bytes=overhead_gib * GIB,
            block_size=block_size,
            tensor_parallel=tensor_parallel,
        )

    @property
    def kv_budget(self) -> float:
        return max(0.0, self.gpu_bytes * self.gpu_util - self.weight_bytes - self.overhead_bytes)

    @property
    def fits(self) -> bool:
        return self.kv_budget > 0

    def max_seqs_paged(self, seq_len: int) -> int:
        """Sequences of ``seq_len`` tokens that fit when memory is allocated page by page."""
        per_seq = self.spec.paged_seq_bytes(seq_len, self.block_size, self.kv_dtype)
        return int(self.kv_budget // per_seq) if per_seq else 0

    def max_seqs_reserved(self, reserve_len: int) -> int:
        """Sequences that fit when each one reserves ``reserve_len`` tokens up front."""
        per_seq = self.spec.seq_bytes(reserve_len, self.kv_dtype)
        return int(self.kv_budget // per_seq) if per_seq else 0
