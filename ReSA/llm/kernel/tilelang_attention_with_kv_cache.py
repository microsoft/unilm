# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
import time
import math

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

def flashattn(heads, heads_kv, dim, dim_v):
    scale = (1.0 / dim)**0.5 * 1.44269504  # log2(e)
    dtype = "bfloat16"
    accum_dtype = "float"
    kv_group_num = heads // heads_kv


    def kernel_func(batch, block_N, block_H, block_M, num_split, num_stages, threads, seqlen_q, max_cache_seqlen):
        shape_q = [batch, seqlen_q, heads, dim]
        shape_k = [batch, max_cache_seqlen, heads_kv, dim]
        shape_v = [batch, max_cache_seqlen, heads_kv, dim_v]
        shape_o = [batch, seqlen_q, heads, dim_v]
        part_shape = [batch, seqlen_q, heads, num_split, dim_v]
        num_block_M = (seqlen_q + block_M - 1) // block_M

        @T.macro
        def flash_attn_split(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlens: T.Tensor([batch], "int32"),
                glse: T.Tensor([batch, seqlen_q, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
        ):
            with T.Kernel(
                    batch * num_block_M, heads_kv, num_split, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M * block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim_v], dtype)
                acc_s = T.alloc_fragment([block_M * block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M * block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_M * block_H, dim_v], accum_dtype)

                scores_max = T.alloc_fragment([block_M * block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M * block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_M * block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_M * block_H], accum_dtype)
                logsum = T.alloc_fragment([block_M * block_H], accum_dtype)

                bid = T.floordiv(bx, num_block_M)
                mid = T.floormod(bx, num_block_M) * block_M
                hid = by
                sid = bz

                for i, d in T.Parallel(block_M * block_H, dim):
                    i_m = T.floordiv(i, block_H)
                    i_h = T.floormod(i, block_H)
                    if i_h < kv_group_num:
                        Q_shared[i, d] = Q[bid, mid + i_m, hid * kv_group_num + i_h, d]

                # T.copy(Q[bid, mid:(mid + block_M), hid * kv_group_num : (hid + 1) * kv_group_num, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                # num_blocks = actual_num_blocks[bid]
                num_blocks = (cache_seqlens[bid] + block_N - 1) // block_N
                blocks_per_split = T.floordiv(num_blocks, num_split)
                remaining_blocks = T.floormod(num_blocks, num_split)
                loop_range = (blocks_per_split + T.if_then_else(sid < remaining_blocks, 1, 0))
                start = blocks_per_split * sid + T.min(sid, remaining_blocks)
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    i_s = start + k
                    T.copy(
                        K[bid, i_s * block_N: (i_s + 1) * block_N, hid, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_M * block_H, block_N):
                        i_m = T.floordiv(i, block_H)
                        i_h = T.floormod(i, block_H)
                        acc_s[i_m * block_H + i_h, j] = T.if_then_else(i_s * block_N + j > cache_seqlens[bid] - seqlen_q + (mid + i_m), -T.infinity(accum_dtype), acc_s[i_m * block_H + i_h, j])
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M * block_H):
                        scores_max[i] = T.if_then_else(scores_max[i] > scores_max_prev[i], scores_max[i], scores_max_prev[i])
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M * block_H, block_N):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M * block_H):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_M * block_H, dim_v):
                        acc_o[i, j] *= scores_scale[i]
                    T.copy(
                        V[bid, i_s * block_N: (i_s + 1) * block_N, hid, :], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M * block_H, dim_v):
                    acc_o[i, j] /= logsum[i]
                
                for i in T.Parallel(block_M * block_H):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                
                for i in T.Parallel(block_M * block_H):
                    i_m = T.floordiv(i, block_H)
                    i_h = T.floormod(i, block_H)
                    if i_h < kv_group_num:
                        glse[bid, mid + i_m, hid * kv_group_num + i_h, sid] = logsum[i]

                for i, v in T.Parallel(block_M * block_H, dim_v):
                    i_m = T.floordiv(i, block_H)
                    i_h = T.floormod(i, block_H)
                    if i_h < kv_group_num:
                        Output_partial[bid, mid + i_m, hid * kv_group_num + i_h, sid, v] = acc_o[i, v]
                              
        @T.macro
        def combine(
                glse: T.Tensor([batch, seqlen_q, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            with T.Kernel(heads, seqlen_q, batch, threads=128) as (bx, by, bz):
                po_local = T.alloc_fragment([dim_v], accum_dtype)
                o_accum_local = T.alloc_fragment([dim_v], accum_dtype)
                lse_local_split = T.alloc_local([1], accum_dtype)
                lse_logsum_local = T.alloc_local([1], accum_dtype)
                lse_max_local = T.alloc_local([1], accum_dtype)
                scale_local = T.alloc_local([1], accum_dtype)
                max_split = T.alloc_local([1], "int32")

                T.annotate_layout({
                    lse_logsum_local: T.Fragment(lse_logsum_local.shape, forward_thread_fn=lambda i: i),
                })

                T.clear(lse_logsum_local)
                T.clear(o_accum_local)
                lse_max_local[0] = -T.infinity(accum_dtype)
                for k in T.serial(num_split):
                    lse_local_split[0] = glse[bz, by, bx, k]
                    if (lse_local_split[0] != 0):
                        max_split[0] = k
                        lse_max_local[0] = T.max(lse_max_local[0], glse[bz, by, bx, k])

                for k in T.Pipelined(num_split, num_stages=1):
                    if k <= max_split[0]:
                        lse_local_split[0] = glse[bz, by, bx, k]
                        lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
                lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
                for k in T.serial(num_split):
                    if k <= max_split[0]:
                        for i in T.Parallel(dim_v):
                            po_local[i] = Output_partial[bz, by, bx, k, i]
                        lse_local_split[0] = glse[bz, by, bx, k]
                        scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                        for i in T.Parallel(dim_v):
                            o_accum_local[i] += po_local[i] * scale_local[0]
                for i in T.Parallel(dim_v):
                    Output[bz, by, bx, i] = o_accum_local[i]

        
        @T.prim_func
        def main(
                Q: T.Tensor(shape_q, dtype),
                K: T.Tensor(shape_k, dtype),
                V: T.Tensor(shape_v, dtype),
                cache_seqlens: T.Tensor([batch], "int32"),
                glse: T.Tensor([batch, seqlen_q, heads, num_split], accum_dtype),
                Output_partial: T.Tensor(part_shape, accum_dtype),
                Output: T.Tensor(shape_o, dtype),
        ):
            flash_attn_split(Q, K, V, cache_seqlens, glse, Output_partial)
            combine(glse, Output_partial, Output)

        return main

    return kernel_func


class AttentionWithKVCache(torch.nn.Module):
    def __init__(self, heads, heads_kv, dim, dim_v, seqlen_q):
        super(AttentionWithKVCache, self).__init__()
        self.heads = heads
        self.heads_kv = heads_kv
        self.dim = dim
        self.dim_v = dim_v
        self.block_N = 32
        self.block_H = tilelang.next_power_of_2(heads // heads_kv)
        self.block_M = seqlen_q

        program = flashattn(heads, heads_kv, dim, dim_v)(
            batch=T.symbolic("batch"),
            block_N=self.block_N,
            block_H=self.block_H,
            block_M=self.block_M,
            num_split=T.symbolic("num_split"),
            num_stages=2,
            threads=128,
            seqlen_q=seqlen_q,
            max_cache_seqlen=T.symbolic("max_cache_seqlen"),
        )

        self.kernel = tilelang.compile(
            program,
            out_idx=-1,
            target='cuda',
            execution_backend="cython"
        )

        props = torch.cuda.get_device_properties(torch.device("cuda:0"))
        self.num_sm = props.multi_processor_count

    def forward(self, query, key, value, cache_seqlens):
        batch = query.shape[0]
        seqlen_q = query.shape[1]
        heads = self.heads
        heads_kv = self.heads_kv
        dim = self.dim
        dim_v = self.dim_v

        # Compute static scheduling parameters
        num_m_blocks = (seqlen_q + self.block_M - 1) // self.block_M
        num_n_blocks = (cache_seqlens.max().item() + self.block_N - 1) // self.block_N
        size_one_kv_head = num_n_blocks * self.block_N * (dim + dim_v) * 2
        total_mblocks = batch * heads_kv * num_m_blocks
        # num_sm = 132
        num_sm = self.num_sm

        num_split = num_splits_heuristic(
            total_mblocks, num_sm, num_n_blocks, num_m_blocks,
            size_one_kv_head, is_causal_or_local=True, max_splits=16
        )

        glse = torch.empty((batch, seqlen_q, heads, num_split), dtype=torch.float32, device='cuda')
        output_partial = torch.empty((batch, seqlen_q, heads, num_split, dim_v), dtype=torch.float32, device='cuda')

        output = self.kernel(
            query, key, value, cache_seqlens,
            glse, output_partial
        )
        return output

def ref_program_fa(query, key, value, cache_seqlens):
    # latency reference
    # from flash_attn_interface import flash_attn_with_kvcache, flash_attn_func # fa3
    from flash_attn import flash_attn_with_kvcache, flash_attn_func #fa2
    output = flash_attn_with_kvcache(query, key, value, cache_seqlens=cache_seqlens)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--seqlen_q', type=int, default=32, help='sequence length')
    parser.add_argument('--heads', type=int, default=28, help='heads')
    parser.add_argument('--heads_kv', type=int, default=4, help='heads_kv')
    parser.add_argument('--max_cache_seqlen', type=int, default=65536, help='kvcache sequence length')
    parser.add_argument('--dim', type=int, default=128, help='dim')
    parser.add_argument('--dim_v', type=int, default=128, help='dim_v')
    parser.add_argument('--block_size', type=int, default=32, help='block_size')
    parser.add_argument('--load_from_file', type=str, default=None, help='load from file')
    args = parser.parse_args()

    batch, seqlen_q, heads, heads_kv, max_cache_seqlen, dim, dim_v = args.batch, args.seqlen_q, args.heads, args.heads_kv, args.max_cache_seqlen, args.dim, args.dim_v
    block_size = args.block_size

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
    sparse_kernel = AttentionWithKVCache(heads, heads_kv, dim, dim_v, seqlen_q)
    out = sparse_kernel(Q, K, V, cache_seqlens)
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
        # out = sparse_gqa_decode_varlen_indice(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, block_size)
        out = sparse_kernel(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        # out = sparse_gqa_decode_varlen_indice(Q, K, V, block_indices, cache_seqlens, max_cache_seqlen, block_size)
        out = sparse_kernel(Q, K, V, cache_seqlens)
    torch.cuda.synchronize()
    print("sparse time: ", (time.time() - start) / 100*1000)



