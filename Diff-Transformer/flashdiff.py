# modified from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py
import math

import torch
import torch.nn as nn
from einops import rearrange

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func, flash_attn_varlen_func = None, None


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def split_heads(x):
    # split by num_heads, the stripe pattern is friendly to tensor parallel.
    x = rearrange(x, "... (H two) D -> ... H two D", two=2)
    x1 = x[..., 0, :]
    x2 = x[..., 1, :]
    return x1, x2


class FlashDiffAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        head_dim: The dimension of the heads.
        depth: The layer id, starting from 0.
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(
        self,
        head_dim,
        depth,
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
    ):
        super().__init__()
        assert flash_attn_varlen_func is not None, "FlashAttention is not installed"
        assert flash_attn_func is not None, "FlashAttention is not installed"
        self.head_dim = head_dim
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.drop = nn.Dropout(attention_dropout)
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)
        self.window_size = window_size
        self.deterministic = deterministic

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    def forward(
        self,
        q,
        k,
        v,
        causal=None,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensors containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then each has shape (B, S, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then each has shape
                (total, H, D), where total is the sum of the sequence lengths in the batch.
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and k.is_cuda and v.is_cuda
        causal = self.causal if causal is None else causal
        unpadded = cu_seqlens is not None
        if self.alibi_slopes is not None:
            self.alibi_slopes = self.alibi_slopes.to(torch.float32)

        q1, q2 = split_heads(q)
        k1, k2 = split_heads(k)
        v1, v2 = split_heads(v)

        kwargs = {
            "dropout_p": self.drop.p if self.training else 0.0,
            "softmax_scale": self.softmax_scale,
            "causal": causal,
            "alibi_slopes": self.alibi_slopes,
            "window_size": self.window_size,
            "deterministic": self.deterministic,
        }

        if unpadded:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen_k, int)

            kwargs.update({
                "cu_seqlens_q": cu_seqlens,
                "max_seqlen_q": max_seqlen,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_k": max_seqlen_k,
            })
            attn_func = flash_attn_varlen_func
        else:
            attn_func = flash_attn_func
        
        attn11 = attn_func(q1, k1, v1, **kwargs)
        attn12 = attn_func(q1, k1, v2, **kwargs)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        attn21 = attn_func(q2, k2, v1, **kwargs)
        attn22 = attn_func(q2, k2, v2, **kwargs)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn = attn1 - lambda_full * attn2
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        # reshape back to 2 * num_head
        attn = rearrange(attn, "... H (two D) -> ... (H two) D", two=2)
        return attn
