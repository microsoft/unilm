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

class SlidingWindowAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.dim
        self.num_heads = args.n_self_heads // args.model_parallel_size
        self.window_size = args.sliding_window
        
        self.head_dim = args.dim // args.n_self_heads

        self.q_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=init_method)
        self.k_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=init_method)
        self.v_proj = ColumnParallelLinear(args.dim, args.dim, bias=False, gather_output=False, init_method=init_method)
        self.out_proj = RowParallelLinear(args.dim, args.dim, bias=False, input_is_parallel=True, init_method=init_method)
    
    def forward(
        self,
        x,
        rel_pos,
        start_pos=0,
        incremental_state=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)
        if incremental_state is not None:
            if "prev_key" not in incremental_state:
                incremental_state["prev_key"] = torch.empty(self.args.max_batch_size, self.window_size, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
                incremental_state["prev_value"] = torch.empty(self.args.max_batch_size, self.window_size, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)

            key = torch.cat([incremental_state["prev_key"][:bsz, :start_pos], k], dim=1)
            value = torch.cat([incremental_state["prev_value"][:bsz, :start_pos], v], dim=1)
            if key.shape[1] > self.window_size:
                incremental_state["prev_key"][:bsz] = key[:, -self.window_size:]
                incremental_state["prev_value"][:bsz] = value[:, -self.window_size:]
            else:
                incremental_state["prev_key"][:bsz, start_pos : start_pos + tgt_len] = k
                incremental_state["prev_value"][:bsz, start_pos : start_pos + tgt_len] = v
        else:
            key, value = k, v
        attn = flash_attn_func(q, key, value, causal=True, window_size=(self.window_size - 1, 0)) 
        attn = attn.reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        attn = self.out_proj(attn)
        return attn