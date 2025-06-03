import math
import torch

import triton
import triton.language as tl

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, num_m_blocks, size_one_kv_head,
                         is_causal_or_local, max_splits):
    """
    Determines the optimal number of splits for maximizing GPU occupancy while balancing memory efficiency.

    Parameters:
    - total_mblocks (int): Total number of m_blocks.
    - num_SMs (int): Number of Streaming Multiprocessors (SMs) in the GPU.
    - num_n_blocks (int): Number of n_blocks.
    - num_m_blocks (int): Number of m_blocks.
    - size_one_kv_head (int): Size of one KV head in bytes.
    - is_causal_or_local (bool): Indicates whether the operation is causal or local.
    - max_splits (int): Maximum number of allowed splits.

    Returns:
    - int: The optimal number of splits.
    """
    # If we have enough m_blocks to almost fill the SMs, prefer 1 split unless memory constraints apply.
    if total_mblocks >= 0.8 * num_SMs:
        size_l2 = 50 * 1024 * 1024  # L2 cache size assumption (50MB)
        # Only split if each KV head is too large for L2 and there are enough m_blocks
        if size_one_kv_head > size_l2 and num_m_blocks >= num_SMs * 2 and not is_causal_or_local:
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        else:
            return 1

    # If num_n_blocks is too small, we don't split
    if num_n_blocks <= 4:
        return 1

    # Limit max_splits to a reasonable range
    max_splits = min(max_splits, num_SMs, num_n_blocks)

    max_efficiency = 0.0
    efficiency = []

    # Compute efficiency for different splits
    for num_splits in range(1, max_splits + 1):
        n_waves = (total_mblocks * num_splits) / num_SMs
        eff = n_waves / math.ceil(n_waves)
        # Track max efficiency
        if eff > max_efficiency:
            max_efficiency = eff

        efficiency.append(eff)

    # Find the smallest number of splits that achieves at least 85% of max efficiency
    for num_splits in range(1, max_splits + 1):
        if efficiency[num_splits - 1] >= 0.95 * max_efficiency:
            return num_splits

    return 1

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['gqa_group_size', 'BLOCK_H', 'BLOCK_N', 'BLOCK_D', 'BLOCK_V'],
)
@triton.jit
def _fwd_kernel_with_kv_cache(
    Q, K, V, Out, L,
    sm_scale,
    cache_seqlens,
    stride_qz, stride_qt, stride_qh, stride_qd,
    stride_kz, stride_kt, stride_kh, stride_kd,
    stride_vz, stride_vt, stride_vh, stride_vd,
    stride_oz, stride_ot, stride_oh, stride_os, stride_od,
    stride_lz, stride_lt, stride_lh, stride_ls,
    num_splits: tl.constexpr,
    seqlen_q: tl.constexpr,
    num_m_blocks: tl.constexpr,
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_sm = tl.program_id(0).to(tl.int64)
    off_split, off_m = off_sm // num_m_blocks, off_sm % num_m_blocks
    off_h_for_kv = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)

    off_h_q = off_h_for_kv * gqa_group_size
    offs_h = tl.arange(0, BLOCK_H)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    mask_h = offs_h < gqa_group_size
    seqlen_k = tl.load(cache_seqlens + off_z)

    Q += off_z * stride_qz + off_h_q * stride_qh
    K += off_z * stride_kz + off_h_for_kv * stride_kh
    V += off_z * stride_vz + off_h_for_kv * stride_vh
    L += off_z * stride_lz + off_h_q * stride_lh + off_split * stride_ls
    Out += off_z * stride_oz + off_h_q * stride_oh + off_split * stride_os
    
    num_kv_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    blocks_per_split = num_kv_blocks // num_splits
    remaining_blocks = num_kv_blocks % num_splits
    loop_range = blocks_per_split + (1 if off_split < remaining_blocks else 0)
    start = blocks_per_split * off_split + min(off_split, remaining_blocks)

    offs_m = tl.arange(0, BLOCK_M) + off_m * BLOCK_M
    mask_q = offs_m < seqlen_q
    mask_qh = mask_q[:, None] & mask_h[None, :]
    q_idx = offs_m[:, None] + tl.zeros([BLOCK_H], dtype=tl.int32) + seqlen_k - seqlen_q
    q_idx = tl.reshape(q_idx, [BLOCK_M * BLOCK_H])

    q = tl.load(Q + offs_m[:, None, None] * stride_qt + offs_h[None, :, None] * stride_qh + offs_d[None, None, :] * stride_qd, mask=mask_qh[:, :, None]) ## padding to min 16
    q = tl.reshape(q, (BLOCK_M * BLOCK_H, BLOCK_D))

    m_i = tl.full([BLOCK_M * BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M * BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M * BLOCK_H, BLOCK_V], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_v[None, :] * stride_vd

    for block_idx in range(start, start + loop_range):
        start_n = block_idx * BLOCK_N
        k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k, cache_modifier=".ca")
        qk = tl.dot(q, k)

        causal_mask = q_idx[:, None] >= start_n + offs_n[None, :]
        qk = tl.where(causal_mask, qk, -1.0e6)
        qk *= sm_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.exp(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k, cache_modifier=".ca")
        p = p.to(v.type.element_ty)

        acc += tl.dot(p, v)
        m_i = m_ij

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    m_i += tl.math.log(l_i)

    l_ptrs = L + offs_m[:, None] * stride_lt + offs_h * stride_lh
    m_i = tl.reshape(m_i, (BLOCK_M, BLOCK_H))
    tl.store(l_ptrs, m_i, mask=mask_qh)
    O_ptrs = Out + offs_m[:, None, None] * stride_ot + offs_h[None, :, None] * stride_oh + offs_v[None, None, :] * stride_od
    acc = tl.reshape(acc, (BLOCK_M, BLOCK_H, BLOCK_V))
    tl.store(O_ptrs, acc, mask=mask_qh[:, :, None])

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BLOCK_V'],
)
@triton.jit
def combine(
    out_partial, out, L,
    stride_op_z, stride_op_t, stride_op_h, stride_op_s, stride_op_d,
    stride_o_z, stride_o_t, stride_o_h, stride_o_d,
    stride_l_z, stride_l_t, stride_l_h, stride_l_s,
    num_splits: tl.constexpr,
    num_splits_pow2: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_h = tl.program_id(0).to(tl.int64)
    off_t = tl.program_id(1).to(tl.int64)
    off_z = tl.program_id(2).to(tl.int64)

    split = tl.arange(0, num_splits_pow2)
    split_mask = split < num_splits

    L += off_z * stride_l_z + off_t * stride_l_t + off_h * stride_l_h
    out_partial += off_z * stride_op_z + off_t * stride_op_t + off_h * stride_op_h
    out += off_z * stride_o_z + off_t * stride_o_t + off_h * stride_o_h

    lse_local = tl.load(L + split * stride_l_s, mask=split_mask, other=float("-inf"))
    lse_max_local = tl.max(lse_local, axis=0)

    lse_logsum_local = tl.sum(tl.exp(lse_local - lse_max_local), axis=0)
    lse_logsum_local = tl.log(lse_logsum_local) + lse_max_local
    
    po_local = tl.load(out_partial + split[:, None] * stride_op_s + tl.arange(0, BLOCK_V) * stride_op_d, mask=split_mask[:, None])
    scale_local = tl.exp(lse_local - lse_logsum_local)
    accum_local = tl.sum(po_local * scale_local[:, None], axis=0)
    tl.store(out + tl.arange(0, BLOCK_V) * stride_o_d, accum_local)

def flash_attention_with_kv_cache(
    q, k, v,
    cache_seqlens,
    sm_scale=None,
):
    # split q to blocks
    batch, seqlen_q, n_heads, key_dim = q.shape
    _, _, n_kv_heads, head_dim = v.shape
    gqa_group_size = n_heads // n_kv_heads
    block_h = triton.next_power_of_2(gqa_group_size)
    block_m = max(256 // block_h, 1)
    block_n = 32

    # assert seqlen_q <= 32, "it seems the performance is not good when seqlen_q > 32"

    assert k.size(0) == v.size(0)
    assert q.size(3) == k.size(3)
    assert k.size(1) == v.size(1)
    assert key_dim in {64, 128, 256}
    assert head_dim in {64, 128, 256}

    props = torch.cuda.get_device_properties(torch.device("cuda:0"))
    num_sm = props.multi_processor_count
    num_m_blocks = triton.cdiv(seqlen_q, block_m)
    num_n_blocks = triton.cdiv(cache_seqlens.max(), block_n)
    size_one_kv_head = cache_seqlens.max() * block_n * (key_dim + head_dim) * 2
    total_mblocks = batch * n_kv_heads
    num_splits = num_splits_heuristic(
        total_mblocks, num_sm, num_n_blocks, num_m_blocks,
        size_one_kv_head, is_causal_or_local=True, max_splits=16
    )

    out_partial = torch.empty((batch, seqlen_q, n_heads, num_splits, head_dim), device=q.device, dtype=torch.float32)
    out = torch.empty((batch, seqlen_q, n_heads, head_dim), device=q.device, dtype=q.dtype)
    L = torch.empty((batch, seqlen_q, n_heads, num_splits), device=q.device, dtype=torch.float32)

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    with torch.cuda.device(q.device.index): 
        grid = lambda META: (num_splits * num_m_blocks, n_kv_heads, batch)
        _fwd_kernel_with_kv_cache[grid](
            q, k, v, out_partial, L,
            sm_scale if sm_scale is not None else key_dim ** -0.5,
            cache_seqlens.contiguous(),
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out_partial.stride(),
            *L.stride(),
            num_splits=num_splits,
            seqlen_q=seqlen_q,
            num_m_blocks=num_m_blocks,
            gqa_group_size=gqa_group_size,
            BLOCK_H = block_h,
            BLOCK_M = block_m,
            BLOCK_N = block_n,
            BLOCK_D = key_dim,
            BLOCK_V = head_dim,
            **extra_kern_args
        )
        grid = lambda META: (n_heads, seqlen_q, batch)
        combine[grid](
            out_partial, out, L,
            *out_partial.stride(),
            *out.stride(),
            *L.stride(),
            num_splits=num_splits,
            num_splits_pow2=triton.next_power_of_2(num_splits),
            BLOCK_V=head_dim,
            **extra_kern_args
        )
    return out

def ref_program_fa(query, key, value, cache_seqlens):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache, flash_attn_func # fa3
    from flash_attn import flash_attn_with_kvcache, flash_attn_func #fa2
    output = flash_attn_with_kvcache(query, key, value, cache_seqlens=cache_seqlens, causal=True)
    return output



def debug(name,expect, actual, atol=1e-3, rtol=1e-3):
    all_close = torch.allclose(expect, actual, atol=atol, rtol=rtol)
    print(name + "  all_close={}".format(all_close))
    if not all_close:
        # print(expect[3, 28])
        # print(actual[3, 28])
        diff = (expect - actual).abs()
        print("all_close={}, max={}, min={}, mean={}".format(all_close, diff.max().item(), diff.min().item(), diff.mean().item()))
        max_indices = torch.nonzero(diff == diff.max().item())
        first_index = tuple(max_indices[0].tolist())
        print(f"Index: {first_index}, expect: {expect[first_index]}, actual: {actual[first_index]}") 

if __name__ == "__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--seqlen_q', type=int, default=128, help='sequence length')
    parser.add_argument('--heads', type=int, default=28, help='heads')
    parser.add_argument('--heads_kv', type=int, default=4, help='heads_kv')
    parser.add_argument('--max_cache_seqlen', type=int, default=65536, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--load_from_file', type=str, default=None, help='load from file')
    args = parser.parse_args()

    batch, seqlen_q, heads, heads_kv, max_cache_seqlen, dim, dim_v = args.batch, args.seqlen_q, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v

    dtype = torch.bfloat16

    Q = torch.randn((batch, seqlen_q, heads, dim), dtype=dtype, device='cuda')
    K = torch.randn((batch, max_cache_seqlen, heads_kv, dim), dtype=dtype, device='cuda')
    V = torch.randn((batch, max_cache_seqlen, heads_kv, dim_v), dtype=dtype, device='cuda')

    cache_seqlens = torch.randint(max_cache_seqlen - 32, max_cache_seqlen, (batch,), device='cuda', dtype=torch.int32)
    print("cache_seqlens: ", cache_seqlens)
    # parity reference
    ref = ref_program_fa(Q, K, V, cache_seqlens)
    # ref = ref_program_triton(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, max_num_blocks, block_size)
    # out = kernel(Q, K, V, block_indices, cache_seqlens, actual_num_blocks, glse, Output_partial)
    # out = sparse_gqa_decode_varlen_indice(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, block_size)
    out = flash_attention_with_kv_cache(Q, K, V, cache_seqlens)
    debug("output", ref, out, atol=1e-3, rtol=1e-3)

    ## latency reference
    for i in range(10):
        ref = ref_program_fa(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        ref = ref_program_fa(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    print("dense time: ", (time.time() - start) / 100*1000)

    for i in range(10):
        out = flash_attention_with_kv_cache(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        out = flash_attention_with_kv_cache(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    print("sparse time: ", (time.time() - start) / 100*1000)