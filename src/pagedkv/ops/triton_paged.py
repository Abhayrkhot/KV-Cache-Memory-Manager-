"""Triton paged-attention decode kernel (one query token per sequence).

Grid: (sequence, KV head, KV partition). Each program loads its KV head's query
group (all query heads that share the KV head under GQA, padded to 16 rows so
``tl.dot`` can use tensor cores), then walks the sequence's block table one
page at a time with an online softmax, so K/V are read from HBM exactly once
per KV head.

Split-KV ("flash-decoding"): with few sequences the (sequence, KV head) grid
cannot fill the GPU, so long contexts are cut into partitions processed in
parallel; a second kernel merges the partial results using their log-sum-exp.

Cache layout ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``; FP8 (e4m3)
caches are dequantized in registers with per-layer static scales.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    Out,
    PartOut,
    PartM,
    PartL,
    Q,
    KV,
    BlockTables,
    SeqLens,
    sm_scale,
    k_scale,
    v_scale,
    partition_size,
    stride_qb,
    stride_qh,
    stride_kvb,
    stride_kvkv,
    stride_kvt,
    stride_kvh,
    stride_btb,
    stride_ob,
    stride_oh,
    stride_pb,
    stride_ph,
    stride_pp,
    stride_mb,
    stride_mh,
    QUERY_GROUP: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SPLIT: tl.constexpr,
    KV_FP8: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    part = tl.program_id(2)
    seq_len = tl.load(SeqLens + b)
    start = part * partition_size
    if start >= seq_len:
        return
    end = tl.minimum(start + partition_size, seq_len)

    offs_g = tl.arange(0, GROUP_PAD)
    g_mask = offs_g < QUERY_GROUP
    heads = kvh * QUERY_GROUP + offs_g
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM
    q = tl.load(
        Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
        mask=g_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    m_i = tl.full([GROUP_PAD], float("-inf"), tl.float32)
    l_i = tl.zeros([GROUP_PAD], tl.float32)
    acc = tl.zeros([GROUP_PAD, HEAD_DIM_PAD], tl.float32)
    offs_t = tl.arange(0, BLOCK_SIZE)
    bt_row = BlockTables + b.to(tl.int64) * stride_btb
    for tok0 in range(start, end, BLOCK_SIZE):
        phys = tl.load(bt_row + tok0 // BLOCK_SIZE).to(tl.int64)
        t_mask = (tok0 + offs_t) < end
        kv_mask = t_mask[:, None] & d_mask[None, :]
        k_ptr = (
            KV
            + phys * stride_kvb
            + kvh * stride_kvh
            + offs_t[:, None] * stride_kvt
            + offs_d[None, :]
        )
        k = tl.load(k_ptr, mask=kv_mask, other=0.0)
        v = tl.load(k_ptr + stride_kvkv, mask=kv_mask, other=0.0)
        if KV_FP8:
            k = (k.to(tl.float32) * k_scale).to(q.dtype)
            v = (v.to(tl.float32) * v_scale).to(q.dtype)
        s = tl.dot(q, tl.trans(k)) * sm_scale
        s = tl.where(t_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    o_mask = g_mask[:, None] & d_mask[None, :]
    if SPLIT:
        tl.store(
            PartOut
            + b * stride_pb
            + heads[:, None] * stride_ph
            + part * stride_pp
            + offs_d[None, :],
            out,
            mask=o_mask,
        )
        tl.store(PartM + b * stride_mb + heads * stride_mh + part, m_i, mask=g_mask)
        tl.store(PartL + b * stride_mb + heads * stride_mh + part, l_i, mask=g_mask)
    else:
        tl.store(
            Out + b * stride_ob + heads[:, None] * stride_oh + offs_d[None, :],
            out.to(Out.dtype.element_ty),
            mask=o_mask,
        )


@triton.jit
def _merge_kernel(
    Out,
    PartOut,
    PartM,
    PartL,
    SeqLens,
    partition_size,
    stride_ob,
    stride_oh,
    stride_pb,
    stride_ph,
    stride_pp,
    stride_mb,
    stride_mh,
    NUM_PARTS_PAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PAD: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(SeqLens + b)
    n_parts = tl.cdiv(seq_len, partition_size)
    offs_p = tl.arange(0, NUM_PARTS_PAD)
    p_mask = offs_p < n_parts
    m = tl.load(PartM + b * stride_mb + h * stride_mh + offs_p, mask=p_mask, other=float("-inf"))
    denom = tl.load(PartL + b * stride_mb + h * stride_mh + offs_p, mask=p_mask, other=0.0)
    m_max = tl.max(m, 0)
    w = tl.where(p_mask, denom * tl.exp(m - m_max), 0.0)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM
    o = tl.load(
        PartOut + b * stride_pb + h * stride_ph + offs_p[:, None] * stride_pp + offs_d[None, :],
        mask=p_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    out = tl.sum(o * w[:, None], 0) / tl.sum(w, 0)
    tl.store(
        Out + b * stride_ob + h * stride_oh + offs_d, out.to(Out.dtype.element_ty), mask=d_mask
    )


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


def choose_partition(
    num_seqs: int, num_kv_heads: int, max_seq_len: int, num_sms: int, block_size: int = 16
) -> int:
    """KV tokens per program; 0 means no split.

    Split only when the (sequence, KV head) grid covers less than ~2 waves of SMs
    and there is enough context to divide. Partition sizes are powers of two >= 256
    so each partition streams a reasonable amount of KV, and never smaller than a
    block (both are powers of two, so the partition is a whole number of blocks).
    """
    programs = num_seqs * num_kv_heads
    if programs >= 2 * num_sms or max_seq_len <= 512:
        return 0
    parts = min(-(-4 * num_sms // programs), max_seq_len // 256)
    if parts <= 1:
        return 0
    return max(256, block_size, _next_pow2(-(-max_seq_len // parts)))


def paged_attention_decode(
    query: torch.Tensor,
    kv_layer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    max_seq_len: int,
    *,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    partition_size: int | None = None,
) -> torch.Tensor:
    """Decode attention. ``query`` ``[B, H, D]`` (fp16/bf16) -> ``[B, H, D]``.

    ``kv_layer`` is one layer of the paged cache; ``block_tables`` ``[B, max_blocks]``
    int32; ``seq_lens`` ``[B]`` int32 on the device; ``max_seq_len`` is the host-side
    maximum (avoids a device sync).
    """
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("query must be fp16 or bf16")
    B, H, D = query.shape
    _, two, BS, KVH, D_kv = kv_layer.shape
    if two != 2 or D_kv != D:
        raise ValueError("kv_layer must be [num_blocks, 2, block_size, kv_heads, head_dim]")
    if BS < 16 or BS & (BS - 1):
        raise ValueError("block_size must be a power of two >= 16 (tl.dot tile)")
    if H % KVH:
        raise ValueError("num_heads must be a multiple of num_kv_heads")
    fp8 = kv_layer.dtype == torch.float8_e4m3fn
    if not fp8 and kv_layer.dtype != query.dtype:
        raise TypeError("unquantized cache must match the query dtype")
    group = H // KVH
    out = torch.empty_like(query)
    if B == 0:
        return out
    if query.stride(-1) != 1 or kv_layer.stride(-1) != 1:
        raise ValueError("query and kv_layer must be contiguous in head_dim")
    if block_tables.stride(-1) != 1 or not seq_lens.is_contiguous():
        raise ValueError("block_tables rows and seq_lens must be contiguous")
    # Every sequence must have seq_len >= 1 (a decode token attends at least to itself);
    # not checked here because reading seq_lens would force a device sync.
    if partition_size is None:
        # 108 SMs (A100) when running under TRITON_INTERPRET=1 on the CPU.
        sms = (
            torch.cuda.get_device_properties(query.device).multi_processor_count
            if query.is_cuda
            else 108
        )
        partition_size = choose_partition(B, KVH, max_seq_len, sms, BS)
    split = partition_size > 0
    if split and partition_size % BS:
        raise ValueError("partition_size must be a multiple of block_size")
    num_parts = -(-max_seq_len // partition_size) if split else 1
    if split:
        part_out = torch.empty(B, H, num_parts, D, dtype=torch.float32, device=query.device)
        part_m = torch.empty(B, H, num_parts, dtype=torch.float32, device=query.device)
        part_l = torch.empty_like(part_m)
    else:
        part_out = part_m = part_l = out  # unused
    meta = {
        "QUERY_GROUP": group,
        "GROUP_PAD": max(16, _next_pow2(group)),
        "HEAD_DIM": D,
        "HEAD_DIM_PAD": max(16, _next_pow2(D)),
        "BLOCK_SIZE": BS,
    }
    _decode_kernel[(B, KVH, num_parts)](
        out,
        part_out,
        part_m,
        part_l,
        query,
        kv_layer,
        block_tables,
        seq_lens,
        scale,
        k_scale,
        v_scale,
        partition_size if split else max(max_seq_len, 1),
        query.stride(0),
        query.stride(1),
        kv_layer.stride(0),
        kv_layer.stride(1),
        kv_layer.stride(2),
        kv_layer.stride(3),
        block_tables.stride(0),
        out.stride(0),
        out.stride(1),
        part_out.stride(0) if split else 0,
        part_out.stride(1) if split else 0,
        part_out.stride(2) if split else 0,
        part_m.stride(0) if split else 0,
        part_m.stride(1) if split else 0,
        SPLIT=split,
        KV_FP8=fp8,
        num_warps=4,
        num_stages=2,
        **meta,
    )
    if split:
        _merge_kernel[(B, H)](
            out,
            part_out,
            part_m,
            part_l,
            seq_lens,
            partition_size,
            out.stride(0),
            out.stride(1),
            part_out.stride(0),
            part_out.stride(1),
            part_out.stride(2),
            part_m.stride(0),
            part_m.stride(1),
            NUM_PARTS_PAD=_next_pow2(num_parts),
            HEAD_DIM=D,
            HEAD_DIM_PAD=meta["HEAD_DIM_PAD"],
        )
    return out
