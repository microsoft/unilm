import math
import torch
import torch.nn.functional as F
from torch import nn

from kernel.rotary import apply_rotary_emb
from flex_head_fa import flash_attn_func
try:
    from apex.normalization import FusedRMSNorm as RMSNorm 
except ModuleNotFoundError:
    print("No fused RMSNorm")
    from rms_norm import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadFlashDiff1(nn.Module):
    """
    (Recommended)
    DiffAttn implemented with FlashAttention, for packages that support different qk/v dimensions
    e.g., our customized flex_head_fa (https://aka.ms/flash-diff) and xformers (https://github.com/facebookresearch/xformers)
    """
    def __init__(
        self,
        embed_dim,
        depth, # current layer index
        num_heads,
        num_kv_heads=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # arg num_heads set to half of baseline Transformer's num_heads
        # for e.g., to compare with a baseline Transformer with 16 heads, pass in num_heads=8 for DIFF Transformer
        self.num_heads = num_heads
        
        # arg num_kv_heads set to half of baseline Transformer's num_kv_heads if use GQA
        # for e.g., to compare with a baseline Transformer with 16 heads and 8 kv_heads, 
        # pass in num_heads=8, num_kv_heads=4 for DIFF Transformer
        # if use MHA, pass in num_kv_heads=None
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim // self.n_rep, bias=False)
        self.out_proj = nn.Linear(self.num_heads * 2 * self.head_dim, embed_dim, bias=False)

        # depth means current layer index
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
    
    def forward(
        self,
        x,
        rel_pos,
        attn_mask=None,
    ):
        bsz, tgt_len, embed_dim = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        offset = src_len - tgt_len
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        attn1 = flash_attn_func(q1, k1, v, causal=True)
        attn2 = flash_attn_func(q2, k2, v, causal=True)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2

        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        
        attn = self.out_proj(attn)
        return attn
