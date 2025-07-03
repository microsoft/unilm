import torch
import torch.nn.functional as F
import triton
import triton.language as tl
       
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": BLOCK_N}, num_warps=num_warps, num_stages=1)
        for num_warps in [1, 2, 4, 8]
        for BLOCK_N in [32, 64]
    ],
    key=['local_block_num', 'head_dim'],
)
@triton.jit
def block_attn_decoding_kernel(
    q,                             # pointer to float32, shape [batch_size, n_head * gqa_size, head_dim]
    k_min,                         # pointer to float32, shape [batch_size, num_blocks, n_head, head_dim]
    k_max,                         # pointer to float32, shape [batch_size, num_blocks, n_head, head_dim]
    num_blocks,                    # pointer to int32, shape [batch_size]
    local_block_num: tl.constexpr, # int32, number of local blocks per query
    head_dim: tl.constexpr,        # int32, dimension per head
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_ab, stride_ah, stride_at,
    attn_score,                    # pointer to int32, shape [batch_size, n_head, num_blocks]
    BLOCK_N: tl.constexpr,         # int32, number of queries per block
):
    batch_id = tl.program_id(0)
    head_idx = tl.program_id(1)
    start_block_idx = tl.program_id(2) * BLOCK_N
    base_q_ptr = tl.make_block_ptr(q + batch_id * stride_qb + head_idx * stride_qh, shape=(head_dim,), block_shape=(head_dim,), strides=(stride_qd,), offsets=(0,), order=(0,))
    mean_q = tl.load(base_q_ptr).to(tl.float32)
    POS_INF = 1e38

    total_n_blocks = tl.load(num_blocks + batch_id)
    block_block_idx = start_block_idx + tl.arange(0, BLOCK_N)
    k_min_ptr = tl.make_block_ptr(k_min + batch_id * stride_kb + head_idx * stride_kh, shape=(total_n_blocks, head_dim), block_shape=(BLOCK_N, head_dim), strides=(stride_kt, stride_kd), offsets=(start_block_idx, 0), order=(0, 1))
    k_max_ptr = tl.make_block_ptr(k_max + batch_id * stride_kb + head_idx * stride_kh, shape=(total_n_blocks, head_dim), block_shape=(BLOCK_N, head_dim), strides=(stride_kt, stride_kd), offsets=(start_block_idx, 0), order=(0, 1))
    k_min_val = tl.load(k_min_ptr, boundary_check=(0, 1)).to(tl.float32)
    k_max_val = tl.load(k_max_ptr, boundary_check=(0, 1)).to(tl.float32)
    score = tl.maximum(k_min_val * mean_q, k_max_val * mean_q)
    score = tl.sum(score, axis=1)
    score = tl.where(block_block_idx >= total_n_blocks - local_block_num, POS_INF, score)
    attn_score_ptr = tl.make_block_ptr(attn_score + batch_id * stride_ab + head_idx * stride_ah, shape=(total_n_blocks,), block_shape=(BLOCK_N,), strides=(stride_at,), offsets=(start_block_idx,), order=(0,))
    tl.store(attn_score_ptr, score, boundary_check=(0,))

def block_attn_decoding(q, k_min, k_max, num_blocks, local_block_num):
    assert num_blocks.max().item() <= k_min.shape[1] , "num_blocks should be less than or equal to k_min.shape[1]"
    min_val = torch.finfo(torch.float32).min
    batch, n_kv_heads, head_dim = q.shape[0], k_min.shape[2], k_min.shape[3]
    attn_score = torch.full((batch, n_kv_heads, num_blocks.max().item()), fill_value=min_val, device=q.device, dtype=torch.float32)
    grid = lambda META: (batch, n_kv_heads, triton.cdiv(num_blocks.max().item(), META['BLOCK_N']))
    with torch.cuda.device(q.device.index): 
        block_attn_decoding_kernel[grid](q, k_min, k_max, 
            num_blocks, local_block_num, head_dim,
            *q.stride(),
            *k_min.stride(),
            *attn_score.stride(),
            attn_score,
        )
    return attn_score

@torch.compile(fullgraph=True)
def get_topk_indices(block_attn_score, block_index_mask, max_num_selected_blocks):
    block_index = torch.topk(block_attn_score, k=max_num_selected_blocks, dim=-1, sorted=True).indices
    topk_indices = block_index.masked_fill_(~block_index_mask, -1)
    topk_indices = torch.sort(topk_indices, dim=-1, descending=False).values
    return topk_indices.to(torch.int32)

@torch.compile(fullgraph=True)
def get_num_blocks(cache_seqlens, block_size, sparse_ratio, min_block_num):
    num_blocks = (cache_seqlens + (block_size - 1)) // block_size
    num_selected_blocks = torch.ceil(num_blocks * sparse_ratio)
    num_selected_blocks = torch.maximum(num_selected_blocks, num_blocks.clamp(max=min_block_num)).long()
    return num_blocks, num_selected_blocks

class KVManager:
    def __init__(self, num_heads, block_size, sparse_ratio, local_block_num, min_block_num):
        self.num_heads = num_heads
        self.block_size = block_size
        self.sparse_ratio = sparse_ratio
        self.local_block_num = local_block_num
        self.min_block_num = min_block_num
        self.block_max_key = None
        self.block_min_key = None
        self.num_elements = None
        assert self.local_block_num <= self.min_block_num, "local_block_num should be less than or equal to min_block_num"

    def init_centeroids(self, key, cache_seqlens):
        bsz, seqlen, num_heads, head_dim = key.shape
        max_val = torch.finfo(key.dtype).max
        min_val = torch.finfo(key.dtype).min
        self.num_elements = seqlen
        key_block = key.reshape(bsz, -1, self.block_size, num_heads, head_dim)
        self.block_max_key = key_block.max(dim=2).values
        self.block_min_key = key_block.min(dim=2).values
        num_blocks = (cache_seqlens + self.block_size - 1) // self.block_size
        key_last_block = key_block[torch.arange(bsz, device=key.device), num_blocks - 1]
        valid_mask = torch.arange(self.block_size, device=key.device) + (num_blocks[:, None] - 1) * self.block_size < cache_seqlens[:, None]
        last_min_key = torch.masked_fill(key_last_block, ~valid_mask[:, :, None, None], max_val).min(dim=1).values
        last_max_key = torch.masked_fill(key_last_block, ~valid_mask[:, :, None, None], min_val).max(dim=1).values
        self.block_max_key[torch.arange(bsz, device=key.device), num_blocks - 1] = last_max_key
        self.block_min_key[torch.arange(bsz, device=key.device), num_blocks - 1] = last_min_key
        causal_mask = torch.arange(self.block_max_key.shape[1], device=key.device) < num_blocks[:, None]
        self.block_max_key.masked_fill_(~causal_mask[:, :, None, None], min_val)
        self.block_min_key.masked_fill_(~causal_mask[:, :, None, None], max_val)
        
    @torch.compile(fullgraph=True)
    def update_centeroids(self, key, cache_seqlens):
        num_blocks = (cache_seqlens + self.block_size - 1) // self.block_size
        batch_index = torch.arange(num_blocks.shape[0], device=key.device)
        self.block_max_key[batch_index, num_blocks - 1] = torch.maximum(self.block_max_key[batch_index, num_blocks - 1], key)
        self.block_min_key[batch_index, num_blocks - 1] = torch.minimum(self.block_min_key[batch_index, num_blocks - 1], key)

    def clear_centeroids(self):
        self.block_max_key = None
        self.block_min_key = None
        self.num_elements = None

    def get_kv_cache_indices(self, query, cache_seqlens):
        bsz, num_heads, head_dim = query.shape
        max_val = torch.finfo(torch.float32).max
        query = query.view(bsz, 1, self.num_heads, num_heads // self.num_heads, head_dim).mean(dim=3).float() * (head_dim ** -0.5)
        attn_score = torch.maximum(query * self.block_max_key.float(), query * self.block_min_key.float()).sum(dim=-1)
        topk_indices = torch.full((bsz, self.num_heads, self.block_max_key.shape[1]), device=query.device, dtype=torch.int32, fill_value=-1)
        num_blocks = (cache_seqlens + self.block_size - 1) // self.block_size
        num_selected_blocks = torch.ceil(num_blocks * self.sparse_ratio)[:, None].repeat(1, self.num_heads)

        num_selected_blocks = torch.maximum(num_selected_blocks, num_blocks.clamp(max=self.min_block_num)[:, None]).long()
        for b in range(bsz):
            local_attn_score = attn_score[b, :num_blocks[b]]
            local_attn_score[-self.local_block_num:] = max_val
            block_index = torch.sort(local_attn_score, dim=0, descending=True).indices.transpose(0, 1)
            block_index_mask = torch.arange(num_blocks[b], device=query.device) < num_selected_blocks[b][:, None]
            topk_indices[b, :, :num_blocks[b]] = block_index.masked_fill(~block_index_mask, -1).to(torch.int32)
        topk_indices = torch.sort(topk_indices, dim=-1, descending=True).values
        return topk_indices

    def get_kv_cache_indices_fast(self, query, cache_seqlens):
        bsz, num_heads, head_dim = query.shape
        num_blocks, num_selected_blocks = get_num_blocks(cache_seqlens, self.block_size, self.sparse_ratio, self.min_block_num)
        max_num_selected_blocks = num_selected_blocks.max().item()
        query = query.view(bsz, self.num_heads, num_heads // self.num_heads, head_dim).mean(dim=2) * (head_dim ** -0.5)
        block_attn_score = block_attn_decoding(query, self.block_max_key, self.block_min_key, num_blocks, self.local_block_num)
        block_index_mask = torch.arange(max_num_selected_blocks, device=query.device) < num_selected_blocks[:, None, None]
        topk_indices = get_topk_indices(block_attn_score, block_index_mask, max_num_selected_blocks)
        return topk_indices

        
