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
def _fwd_kernel_decoding(
    Q, K, V, Out, L,
    sm_scale,
    cache_seqlens,
    block_indices_ptr,
    stride_qz, stride_qh, stride_qd,
    stride_kz, stride_kt, stride_kh, stride_kd,
    stride_vz, stride_vt, stride_vh, stride_vd,
    stride_oz, stride_oh, stride_os, stride_od,
    stride_lz, stride_lh, stride_ls,
    stride_bz, stride_bn, stride_bd,
    max_selected_blocks: tl.constexpr,
    num_splits: tl.constexpr,
    gqa_group_size: tl.constexpr,
    BLOCK_H: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_z = tl.program_id(0).to(tl.int64)
    off_h_for_kv = tl.program_id(1).to(tl.int64)
    off_split = tl.program_id(2).to(tl.int64)

    off_h_q = off_h_for_kv * gqa_group_size
    
    offs_m = tl.arange(0, BLOCK_H) ## head 
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)

    seqlen_k = tl.load(cache_seqlens + off_z)

    Q += off_z * stride_qz + off_h_q * stride_qh
    K += off_z * stride_kz + off_h_for_kv * stride_kh
    V += off_z * stride_vz + off_h_for_kv * stride_vh
    L += off_z * stride_lz + off_h_q * stride_lh + off_split * stride_ls
    Out += off_z * stride_oz + off_h_q * stride_oh + off_split * stride_os
    block_indices_ptr += off_z * stride_bz + off_h_for_kv * stride_bn

    q = tl.load(Q + offs_m[:, None] * stride_qh + offs_d[None, :] * stride_qd,
                mask=(offs_m[:, None] < gqa_group_size)) ## padding to min 16

    blocks_per_split = max_selected_blocks // num_splits
    remaining_blocks = max_selected_blocks % num_splits
    loop_range = blocks_per_split + (1 if off_split < remaining_blocks else 0)
    start = blocks_per_split * off_split + min(off_split, remaining_blocks)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_V], dtype=tl.float32)

    k_ptrs = K + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
    v_ptrs = V + offs_n[:, None] * stride_vt + offs_v[None, :] * stride_vd
    for block_ptr_idx in range(start, start + loop_range):
        block_idx = tl.load(block_indices_ptr + block_ptr_idx * stride_bd)
        if block_idx >= 0:
            start_n = block_idx * BLOCK_N
            k = tl.load(k_ptrs + start_n * stride_kt, mask=offs_n[None, :] + start_n < seqlen_k)
            qk = tl.dot(q, k)
            qk = tl.where(offs_n[None, :] + start_n < seqlen_k, qk, -1e6)
            qk *= sm_scale

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
            p = tl.exp(qk)
            l_ij = tl.sum(p, 1)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v = tl.load(v_ptrs + start_n * stride_vt, mask=offs_n[:, None] + start_n < seqlen_k)
            p = p.to(v.type.element_ty)

            acc += tl.dot(p, v)
            m_i = m_ij

    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    m_i += tl.math.log(l_i)

    l_ptrs = L + offs_m * stride_lh
    tl.store(l_ptrs, m_i, mask=(offs_m < gqa_group_size))
    O_ptrs = Out + offs_m[:, None] * stride_oh + offs_v[None, :] * stride_od
    tl.store(O_ptrs, acc, mask=(offs_m[:, None] < gqa_group_size))

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
    stride_op_z, stride_op_h, stride_op_s, stride_op_d,
    stride_o_z, stride_o_h, stride_o_d,
    stride_l_z, stride_l_h, stride_l_s,
    num_splits: tl.constexpr,
    num_splits_pow2: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    off_z = tl.program_id(0).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    split = tl.arange(0, num_splits_pow2)
    split_mask = split < num_splits

    lse_local = tl.load(L + off_z * stride_l_z + off_h * stride_l_h + split * stride_l_s, mask=split_mask, other=float("-inf"))
    lse_max_local = tl.max(lse_local, axis=0)

    lse_logsum_local = tl.sum(tl.exp(lse_local - lse_max_local), axis=0)
    lse_logsum_local = tl.log(lse_logsum_local) + lse_max_local
    
    po_local = tl.load(out_partial + off_z * stride_op_z + off_h * stride_op_h + split[:, None] * stride_op_s + tl.arange(0, BLOCK_V) * stride_op_d, mask=split_mask[:, None])
    scale_local = tl.exp(lse_local - lse_logsum_local)
    accum_local = tl.sum(po_local * scale_local[:, None], axis=0)
    tl.store(out + off_z * stride_o_z + off_h * stride_o_h + tl.arange(0, BLOCK_V) * stride_o_d, accum_local)

def flash_block_sparse_decoding(
    q, k, v,
    cache_seqlens,
    block_indices,
    sm_scale=None,
    block_size=64,
    num_splits=None
):
    # split q to blocks
    batch, n_heads, key_dim = q.shape
    _, _, n_kv_heads, head_dim = v.shape
    gqa_group_size = n_heads // n_kv_heads
    max_selected_blocks = block_indices.shape[-1]
    block_h = max(triton.next_power_of_2(gqa_group_size), 16)

    assert k.size(0) == v.size(0)
    assert q.size(2) == k.size(3)
    assert k.size(1) == v.size(1)
    assert key_dim in {64, 128, 256}
    assert head_dim in {64, 128, 256}
    assert triton.next_power_of_2(block_size) == block_size, "block size must be power of 2"

    props = torch.cuda.get_device_properties(torch.device("cuda:0"))
    num_sm = props.multi_processor_count
    num_m_blocks = 1
    num_n_blocks = max_selected_blocks
    size_one_kv_head = max_selected_blocks * block_size * (key_dim + head_dim) * 2
    total_mblocks = batch * n_kv_heads * num_m_blocks
    if num_splits is None:
        num_splits = num_splits_heuristic(
            total_mblocks, num_sm, num_n_blocks, num_m_blocks,
            size_one_kv_head, is_causal_or_local=True, max_splits=8)

    out_partial = torch.empty((batch, n_heads, num_splits, head_dim), device=q.device, dtype=torch.float32)
    out = torch.empty((batch, n_heads, head_dim), device=q.device, dtype=q.dtype)
    L = torch.empty((batch, n_heads, num_splits), device=q.device, dtype=torch.float32)

    if is_hip():
        extra_kern_args = {"waves_per_eu": 1}
    else:
        extra_kern_args = {}

    with torch.cuda.device(q.device.index): 
        grid = lambda META: (batch, n_kv_heads, num_splits)
        _fwd_kernel_decoding[grid](
            q, k, v, out_partial, L,
            sm_scale if sm_scale is not None else key_dim ** -0.5,
            cache_seqlens.contiguous(),
            block_indices.contiguous(),
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *out_partial.stride(),
            *L.stride(),
            *block_indices.stride(),
            max_selected_blocks=max_selected_blocks,
            num_splits=num_splits,
            gqa_group_size=gqa_group_size,
            BLOCK_H = block_h,
            BLOCK_N = block_size,
            BLOCK_D = key_dim,
            BLOCK_V = head_dim,
            **extra_kern_args
        )
        grid = lambda META: (batch, n_heads)
        combine[grid](
            out_partial, out, L,
            *out_partial.stride(),
            *out.stride(),
            *L.stride(),
            num_splits=num_splits,
            num_splits_pow2=triton.next_power_of_2(num_splits),
            BLOCK_V = head_dim,
            **extra_kern_args
        )
    return out

def main():
    from torch.nn import functional as F
    import time
    torch.cuda.manual_seed(0)
    bsz, n_head, key_dim = 4, 2, 128
    n_kv_seq = 8192
    head_dim = 128
    gqa_size = 6
    block_size = 16
    dtype = torch.float16
    xq = torch.randn((bsz, n_head * gqa_size, key_dim), device='cuda', dtype=dtype)
    xk = torch.randn((bsz, n_kv_seq, n_head, key_dim), device='cuda', dtype=dtype)
    xv = torch.randn((bsz, n_kv_seq, n_head, head_dim), device='cuda', dtype=dtype)
    cache_seqlens = torch.randint(100, n_kv_seq, (bsz,), device='cuda', dtype=torch.int32)
    sparse_mask = torch.rand((bsz, n_head, (n_kv_seq + block_size - 1) // block_size), device='cuda') > 0.9
    max_selected_blocks = sparse_mask.sum(dim=-1).max()
    print("max_selected_blocks", max_selected_blocks)
    sparse_indices = torch.full((bsz, n_head, max_selected_blocks), -1, device='cuda', dtype=torch.int32)
    for i in range(bsz):
        for j in range(n_head):
            valid_blocks = torch.where(sparse_mask[i, j])[0]
            sparse_indices[i, j, :len(valid_blocks)] = valid_blocks
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        triton_output = flash_block_sparse_decoding(xq, xk, xv, cache_seqlens, sparse_indices, block_size=block_size)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Triton Time taken: {end_time - start_time} seconds")
    
    naive_mask = torch.zeros((bsz, n_head, 1, n_kv_seq), device=xq.device, dtype=torch.bool)
    for i in range(bsz):
        block_mask = sparse_mask[i].repeat_interleave(block_size, dim=-1)
        block_mask = torch.masked_fill(block_mask, torch.arange(n_kv_seq, device=xq.device) >= cache_seqlens[i], False)
        naive_mask[i] = block_mask.unsqueeze(1)
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(100):
        output = F.scaled_dot_product_attention(xq.unsqueeze(2), xk.transpose(1, 2), xv.transpose(1, 2), attn_mask=naive_mask.repeat_interleave(gqa_size, dim=1), enable_gqa=True)
        output = output.view(bsz, n_head * gqa_size, head_dim)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"Torch SDPA Time taken: {end_time - start_time} seconds")
    print(output.shape, triton_output.shape)
    print((output - triton_output).abs().max(), (output - triton_output).abs().mean())
    
if __name__ == "__main__":
    main()