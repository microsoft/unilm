import random
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List
from dataclasses import dataclass
from apex.normalization.fused_layer_norm import fused_rms_norm_affine
from kernel.flash_sparse_decoding import flash_block_sparse_decoding
from kernel.flash_attention_with_kv_cache import flash_attention_with_kv_cache
from kernel.rotary import apply_rotary_emb

from flash_attn import flash_attn_with_kvcache
import math
import logging
logger = logging.getLogger(__name__)

from .context_manager import KVManager

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    vocab_size: Optional[int] = None

    max_batch_size: int = 0
    max_seq_len: int = -1
    model_parallel_size: int = 1
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    save_feature: str = None
    # decoding config
    temperature: float = 0.6
    top_p: float = 0.9
    # SPA config
    resa_rec_freq: int = 32
    resa_block_size: int = 16
    resa_sparse_ratio: float = 0.1
    resa_local_block_num: int = 1
    resa_min_block_num: int = 16

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.normalized_shape = torch.Size((dim,))
    def forward(self, x):
        return fused_rms_norm_affine(x, self.weight, self.normalized_shape, self.eps)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return freqs

class Attention(nn.Module):
    def __init__(self, index: int, args: ModelArgs):
        super(Attention, self).__init__()
        self.args = args
        self.layer_index = index
        self.head_dim = args.dim // args.n_heads
        self.kv_head = args.n_kv_heads
        self.head = args.n_heads
        self.qkv_proj = nn.Linear(self.args.dim, (self.head + self.kv_head * 2) * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.head * self.head_dim, self.args.dim, bias=False)
        # from kernel.tilelang_sparse_decoding import SparseFlashAttn
        # self.sparse_kernel = SparseFlashAttn(self.head, self.kv_head, self.head_dim, self.head_dim, self.args.resa_block_size)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        kv_cache_index: Tuple[torch.Tensor, torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        bsz, seqlen = x.shape[0], x.shape[1]
        seq_bsz = cu_seqlens_q.shape[0] - 1
        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        
        xqkv = self.qkv_proj(x)
        xqkv = rearrange(xqkv, 'b n (h d) -> (b n) h d', d=self.head_dim)
        xqk, xv = xqkv.split([self.head + self.kv_head, self.kv_head], dim=-2)
        xqk = apply_rotary_emb(xqk.unsqueeze(0), *rel_pos, inplace=True).squeeze(0)
        xq, xk = xqk.split([self.head, self.kv_head], dim=-2)
        if kv_cache is not None:
            batch_index, seq_index = kv_cache_index
            kv_cache = kv_cache[self.layer_index]
            kv_cache[0][batch_index, seq_index] = xk
            kv_cache[1][batch_index, seq_index] = xv
        if self.args.resa_sparse_ratio < 1.0 and seqlen == 1:
            assert kv_cache is not None, "kv_cache should not be None for generation"
            kv_manager = kv_cache[2]
            if kv_manager.num_elements is None:
                kv_manager.init_centeroids(kv_cache[0][:seq_bsz], seqlens_k)
            else:
                kv_manager.update_centeroids(xk, seqlens_k)
            sparse_indices = kv_manager.get_kv_cache_indices_fast(xq, seqlens_k)
            output = flash_block_sparse_decoding(xq, kv_cache[0], kv_cache[1], seqlens_k, sparse_indices, block_size=self.args.resa_block_size)
        else:
            max_seqlen_q = seqlens_q.max().item()
            xq_pad = torch.zeros((seq_bsz, max_seqlen_q, self.head, self.head_dim), device=xq.device, dtype=xq.dtype)
            xq_pad_mask = torch.arange(max_seqlen_q, device=xq.device)[None, :] >= max_seqlen_q - seqlens_q[:, None]
            xq_pad[xq_pad_mask] = xq
            if max_seqlen_q <= 32:
                output = flash_attention_with_kv_cache(xq_pad, kv_cache[0], kv_cache[1], cache_seqlens=seqlens_k)
            else:
                output = flash_attn_with_kvcache(xq_pad, kv_cache[0], kv_cache[1], cache_seqlens=seqlens_k, causal=True)
            output = output[xq_pad_mask]
        output = rearrange(output, '(b n) h d -> b n (h d)', b=bsz)
        output = self.o_proj(output)
        return output
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.up_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate_proj = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):

    def __init__(self, index: int, args: ModelArgs):
        super(Block, self).__init__()
        self.args = args
        self.self_attn = Attention(index, args)
        self.mlp = FeedForwardNetwork(args.dim, args.hidden_dim)
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        kv_cache_index: Tuple[torch.Tensor, torch.Tensor],
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), rel_pos=rel_pos, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, kv_cache_index=kv_cache_index, kv_cache=kv_cache)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super(Model, self).__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        nn.init.normal_(self.tok_embeddings.weight, mean=0, std=args.dim ** -0.5)
        self.layers = nn.ModuleList()
        for index in range(args.n_layers):
            block = Block(index,args)
            self.layers.append(block)
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.tie_word_embeddings:
            self.output.weight = self.tok_embeddings.weight
        
        self._precompute_freqs_cis(args.max_seq_len)
        
    def _precompute_freqs_cis(self, max_seqlen):
        freqs_cis = precompute_freqs_cis(self.args.dim // self.args.n_heads, max_seqlen, theta=self.args.rope_theta)
        self.cos = freqs_cis.cos()
        self.sin = freqs_cis.sin()

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        last_hidden_only: bool = False,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache should not be None for decoding only code"
        seqlens_q = cu_seqlens[1:] - cu_seqlens[:-1]
        seqlens_k = start_pos + seqlens_q
        cu_seqlens_q = torch.cat((torch.tensor([0], device=tokens.device), seqlens_q.cumsum(dim=0)), dim=0).to(torch.int32)
        cu_seqlens_k = torch.cat((torch.tensor([0], device=tokens.device), seqlens_k.cumsum(dim=0)), dim=0).to(torch.int32)
        h = self.tok_embeddings(tokens)
        batch_index = torch.cat([torch.full((seqlens_q[i],), fill_value=i) for i in range(len(seqlens_q))])
        seq_index = torch.cat([torch.arange(start_pos[i], seqlens_k[i], device=h.device) for i in range(len(seqlens_q))])
        self.cos, self.sin = self.cos.to(h), self.sin.to(h)
        rel_pos = (self.cos[seq_index], self.sin[seq_index])
        for i, layer in enumerate(self.layers):
            h = layer(h, rel_pos=rel_pos, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, kv_cache_index=(batch_index, seq_index), kv_cache=kv_cache)

        h = self.norm(h)
        if last_hidden_only:
            return h
        else:
            logits = self.output(h)
        return logits


def create_kv_cache(args: ModelArgs, batch_size: int, dtype: torch.dtype = torch.float16, device: torch.device = torch.device('cuda')) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    kv_cache = []
    for _ in range(args.n_layers):
        k_cache = torch.zeros(batch_size, args.max_seq_len, 
                        args.n_kv_heads, args.dim // args.n_heads, dtype=dtype, device=device)
        v_cache = torch.zeros(batch_size, args.max_seq_len, 
                        args.n_kv_heads, args.dim // args.n_heads, dtype=dtype, device=device)
        kv_manager = KVManager(args.n_kv_heads, args.resa_block_size, args.resa_sparse_ratio, args.resa_local_block_num, args.resa_min_block_num)
        kv_cache.append([k_cache, v_cache, kv_manager])
    return kv_cache
