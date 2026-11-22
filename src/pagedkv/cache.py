"""The physical KV cache: one preallocated tensor, addressed by block id.

Layout per layer: ``[num_blocks, 2, block_size, num_kv_heads, head_dim]`` (K and V
of one block are adjacent; FlashInfer calls this the NHD paged layout). A block
id from ``KVCacheManager`` indexes dimension 1 of the full
``[num_layers, num_blocks, ...]`` tensor, so copy-on-write and swapping move a
block for every layer with one indexed copy.

Storage formats:

* ``bf16`` / ``fp16`` / ``fp32``: unquantized.
* ``fp8``: float8 e4m3 with one static scale per layer for K and one for V
  (vLLM's ``kv_cache_dtype=fp8``; scale 1.0 unless calibrated).
* ``int8``: symmetric int8 with one fp32 scale per token per head, computed on
  write (no calibration needed).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

_FLOAT_STORAGE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
FP8_MAX = 448.0  # largest finite float8_e4m3fn


@dataclass(frozen=True)
class CacheLayout:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int

    def block_shape(self) -> tuple[int, int, int, int]:
        return (2, self.block_size, self.num_kv_heads, self.head_dim)


class PagedKVCache:
    def __init__(
        self,
        layout: CacheLayout,
        kv_dtype: str = "bf16",
        compute_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        self.layout = layout
        self.kv_dtype = kv_dtype
        self.compute_dtype = compute_dtype
        self.device = torch.device(device)
        shape = (layout.num_layers, layout.num_blocks, *layout.block_shape())
        self.scales: torch.Tensor | None = None
        if kv_dtype in _FLOAT_STORAGE:
            storage = _FLOAT_STORAGE[kv_dtype]
        elif kv_dtype == "fp8":
            storage = torch.float8_e4m3fn
        elif kv_dtype == "int8":
            storage = torch.int8
            # fp16 scales: 2 bytes per token per head, the overhead KVDtype('int8') budgets for.
            self.scales = torch.ones(shape[:-1], dtype=torch.float16, device=self.device)
        else:
            raise ValueError(f"unsupported kv_dtype {kv_dtype!r}")
        self.kv = torch.zeros(shape, dtype=storage, device=self.device, pin_memory=pin_memory)
        # Static per-layer fp8 scales (index 0 = K, 1 = V). Kept on the host: kernels take
        # them as scalar arguments, and 0-dim host tensors broadcast into device math.
        self.fp8_scales = torch.ones(layout.num_layers, 2, dtype=torch.float32)
        # Optional static per-channel key scales [layers, kv_heads, head_dim] (KIVI-style):
        # keys of some models (Qwen2.5) have outlier channels that a per-token scale cannot absorb.
        self.k_channel_scales: torch.Tensor | None = None
        self._amax: torch.Tensor | None = None
        self._k_ch_amax: torch.Tensor | None = None

    # ------------------------------------------------------------- calibration

    def start_calibration(self) -> None:
        """Record K/V absolute maxima (per layer, and per key channel) on subsequent writes."""
        lay = self.layout
        self._amax = torch.zeros(lay.num_layers, 2, dtype=torch.float32, device=self.device)
        self._k_ch_amax = torch.zeros(
            lay.num_layers, lay.num_kv_heads, lay.head_dim, dtype=torch.float32, device=self.device
        )

    def finish_calibration(self, margin: float = 1.0, key_per_channel: bool = False) -> None:
        """Turn observed maxima into static scales.

        Default: per-layer fp8 scales mapping the maxima to ``FP8_MAX / margin``.
        ``key_per_channel``: per-channel key scales (fp8 or int8) instead; values keep
        their per-layer (fp8) or per-token (int8) scales.
        """
        if self._amax is None or self._k_ch_amax is None:
            raise RuntimeError("start_calibration() was not called")
        if key_per_channel:
            if self.kv_dtype not in ("fp8", "int8"):
                raise ValueError("per-channel key scales need an fp8 or int8 cache")
            qmax = FP8_MAX if self.kv_dtype == "fp8" else 127.0
            self.k_channel_scales = (self._k_ch_amax * margin / qmax).clamp(min=1e-8)
        else:
            self.fp8_scales = (self._amax.cpu() * margin / FP8_MAX).clamp(min=1e-8)
        self._amax = self._k_ch_amax = None

    @property
    def block_size(self) -> int:
        return self.layout.block_size

    def nbytes(self) -> int:
        extra = self.scales.numel() * self.scales.element_size() if self.scales is not None else 0
        return self.kv.numel() * self.kv.element_size() + extra

    def layer(self, layer: int) -> torch.Tensor:
        return self.kv[layer]

    # ------------------------------------------------------------------ writes

    def write(
        self, layer: int, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor
    ) -> None:
        """Store ``key``/``value`` ``[num_tokens, kv_heads, head_dim]`` at flat slots.

        ``slot_mapping[i] = block_id * block_size + offset``; negative slots are skipped
        (padding tokens).
        """
        valid = slot_mapping >= 0
        if not bool(valid.all()):
            key, value, slot_mapping = key[valid], value[valid], slot_mapping[valid]
        if slot_mapping.numel() == 0:
            return
        blocks = torch.div(slot_mapping, self.block_size, rounding_mode="floor")
        offsets = slot_mapping % self.block_size
        dst = self.kv[layer]
        for i, x in ((0, key), (1, value)):
            if self._amax is not None:
                self._amax[layer, i] = torch.maximum(
                    self._amax[layer, i], x.detach().abs().max().float()
                )
                if i == 0 and self._k_ch_amax is not None:
                    self._k_ch_amax[layer] = torch.maximum(
                        self._k_ch_amax[layer], x.detach().abs().amax(0).float()
                    )
            if i == 0 and self.k_channel_scales is not None:
                y = x.float() / self.k_channel_scales[layer]
                if self.kv_dtype == "fp8":
                    dst[blocks, i, offsets] = y.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                else:
                    assert self.scales is not None
                    dst[blocks, i, offsets] = torch.round(y).clamp(-127, 127).to(torch.int8)
                    self.scales[layer, blocks, i, offsets] = 1.0
            elif self.kv_dtype == "fp8":
                scaled = (x.float() / self.fp8_scales[layer, i]).clamp(-FP8_MAX, FP8_MAX)
                dst[blocks, i, offsets] = scaled.to(torch.float8_e4m3fn)
            elif self.kv_dtype == "int8":
                assert self.scales is not None
                amax = x.float().abs().amax(dim=-1)
                # Quantize with the fp16-rounded scale that will be stored, so dequantization
                # uses exactly the scale the values were divided by.
                scale = (amax / 127.0).to(torch.float16).float().clamp(min=2**-24)
                q = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127, 127)
                dst[blocks, i, offsets] = q.to(torch.int8)
                self.scales[layer, blocks, i, offsets] = scale.to(self.scales.dtype)
            else:
                dst[blocks, i, offsets] = x.to(dst.dtype)

    # ------------------------------------------------------------------- reads

    def gather(
        self, layer: int, block_ids: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Contiguous, dequantized ``(K, V)`` ``[length, kv_heads, head_dim]`` for one sequence."""
        lay = self.layout
        raw = self.kv[layer, block_ids]  # [n, 2, bs, h, d]
        out = []
        for i in (0, 1):
            x = raw[:, i].reshape(-1, lay.num_kv_heads, lay.head_dim)[:length]
            if i == 0 and self.k_channel_scales is not None:
                x = x.to(torch.float32) * self.k_channel_scales[layer]
            elif self.kv_dtype == "fp8":
                x = x.to(torch.float32) * self.fp8_scales[layer, i]
            elif self.kv_dtype == "int8":
                assert self.scales is not None
                s = self.scales[layer, block_ids, i].reshape(-1, lay.num_kv_heads)[:length]
                x = x.to(torch.float32) * s.unsqueeze(-1)
            out.append(x.to(self.compute_dtype))
        return out[0], out[1]

    # -------------------------------------------------------------- block moves

    def copy_blocks(self, pairs: Sequence[tuple[int, int]]) -> None:
        """Copy-on-write: duplicate whole blocks (all layers) within this cache."""
        if not pairs:
            return
        src = torch.tensor([p[0] for p in pairs], device=self.device)
        dst = torch.tensor([p[1] for p in pairs], device=self.device)
        self.kv[:, dst] = self.kv[:, src]
        if self.scales is not None:
            self.scales[:, dst] = self.scales[:, src]

    def transfer_to(self, other: PagedKVCache, pairs: Sequence[tuple[int, int]]) -> None:
        """Copy blocks into another cache (GPU <-> pinned host for swapping)."""
        if not pairs:
            return
        if (other.layout.block_shape(), other.kv_dtype) != (
            self.layout.block_shape(),
            self.kv_dtype,
        ):
            raise ValueError("caches have different block layouts")
        src = torch.tensor([p[0] for p in pairs], device=self.device)
        dst = torch.tensor([p[1] for p in pairs], device=other.device)
        # Asynchronous only host -> device: the scatter into `other` then runs on the same
        # CUDA stream. A device -> host copy must finish before the CPU-side scatter reads it.
        non_blocking = other.device.type == "cuda"
        other.kv[:, dst] = self.kv[:, src].to(other.device, non_blocking=non_blocking)
        if self.scales is not None and other.scales is not None:
            other.scales[:, dst] = self.scales[:, src].to(other.device, non_blocking=non_blocking)
