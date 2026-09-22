import torch
from torch import nn
from typing import Optional, Tuple
from kernel.rotary import apply_rotary_emb
from flash_attn import flash_attn_func


@torch.compile
def diff_func(attn1: torch.Tensor, attn2: torch.Tensor, lambda_val: torch.Tensor) -> torch.Tensor:
    return attn1 - torch.sigmoid(lambda_val).unsqueeze(-1) * attn2


class MultiheadFlashDiffV2(nn.Module):
    """
    Differential Attention Version 2 (DiffAttnV2) implementation using Flash Attention.
    """
    def __init__(
        self,
        use_diff_v2: bool, # If False, acts as a baseline Transformer attention
        d_model: int,      # Model dimension
        num_heads: int,    # Number of output heads
        num_kv_heads: Optional[int], # Number of KV heads for GQA
        head_dim: int,     # Dimension per head
    ):
        super().__init__()
        self.use_diff_v2 = use_diff_v2
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = head_dim
        
        self.num_q_heads = 2 * self.num_heads if self.use_diff_v2 else self.num_heads
        self.q_proj = nn.Linear(self.d_model, self.num_q_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model, bias=False)
        self.lambda_proj = nn.Linear(self.d_model, self.num_heads, bias=False) if self.use_diff_v2 else None
    
    def forward(
        self,
        x: torch.Tensor,               # Input tensor [bsz, seq_len, d_model]
        rel_pos: Tuple[torch.Tensor, torch.Tensor], # Rotary embedding (cos, sin)
    ) -> torch.Tensor:
        """
        Forward pass for MultiheadFlashDiffV2.
        
        Args:
            x: Input hidden states of shape [batch, length, d_model]
            rel_pos: Tuple of (cos, sin) tensors for rotary positional embeddings
            
        Returns:
            Output tensor of shape [batch, length, d_model]
        """
        bsz, tgt_len, _ = x.size()
        src_len = tgt_len

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, tgt_len, self.num_q_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)

        attn = flash_attn_func(q, k, v, causal=True)
        if self.use_diff_v2:
            lambda_val = self.lambda_proj(x)
            attn1, attn2 = attn[:, :, 0::2], attn[:, :, 1::2]
            attn = diff_func(attn1, attn2, lambda_val)
        
        attn = attn.reshape(bsz, tgt_len, self.num_heads * self.head_dim)
        output = self.o_proj(attn)
        return output
