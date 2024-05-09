import torch
import torch.nn.functional as F
from torch import nn

from fairseq.model_parallel.megatron.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
)

from .model_parallel_init import init_method
from .kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func

class CrossAttention(nn.Module):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.dim
        self.num_heads = args.n_attn_heads // args.model_parallel_size
        self.num_kv_heads = args.n_attn_kv_heads // args.model_parallel_size
        
        self.head_dim = args.dim // args.n_attn_heads
        self.q_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=init_method)
        self.out_proj = RowParallelLinear(args.dim, args.dim, bias=False, input_is_parallel=True, init_method=init_method)

    def forward(
        self,
        x,
        key,
        value,
        rel_pos
    ):
        bsz, tgt_len, _ = x.size()
        
        q = self.q_proj(x)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        q = apply_rotary_emb(q, *rel_pos, interleaved=True)

        attn = flash_attn_func(q, key, value, causal=True)
        attn = attn.view(bsz, tgt_len, self.head_dim * self.num_heads)

        attn = self.out_proj(attn)
        return attn