import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from flash_attn import flash_attn_func

from fairseq.model_parallel.megatron.mpu import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_model_parallel_region,
    VocabParallelEmbedding
)

from fairscale.nn import checkpoint_wrapper

from .rms_norm import RMSNorm
from .kernel.rotary import apply_rotary_emb
from .model_parallel_init import init_method, vocab_init_method


def precompute_freqs_cis(dim: int, end: int, theta: float) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return freqs


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0
    max_seq_len: int = -1
    model_parallel_size: int = 1
    load_checkpoint: bool = False
    rope_theta: float = 10000.0
    sliding_window: Optional[int] = None


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.dim = args.dim
        self.head_dim = args.head_dim
        self.hidden_dim = args.n_heads * args.head_dim
        self.key_value_dim = args.n_kv_heads * args.head_dim
        self.n_heads = args.n_heads // args.model_parallel_size
        self.n_kv_heads = args.n_kv_heads // args.model_parallel_size
        self.activate_sliding_window = args.sliding_window is not None
        self.cache_len = args.sliding_window - 1 if self.activate_sliding_window else args.max_seq_len

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.args.head_dim**-0.5

        self.wq = ColumnParallelLinear(self.dim, self.hidden_dim, bias=False, gather_output=False, init_method=init_method)
        self.wk = ColumnParallelLinear(self.dim, self.key_value_dim, bias=False, gather_output=False, init_method=init_method)
        self.wv = ColumnParallelLinear(self.dim, self.key_value_dim, bias=False, gather_output=False, init_method=init_method)
        self.wo = RowParallelLinear(self.hidden_dim, self.dim, bias=False, input_is_parallel=True, init_method=init_method)

    def forward(
        self,
        x: torch.Tensor,
        rel_pos: Tuple[torch.Tensor, torch.Tensor],
        start_pos: int,
        incremental_state = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xq = apply_rotary_emb(xq, *rel_pos)
        xk = apply_rotary_emb(xk, *rel_pos)
        if incremental_state is not None:
            if "cache_k" not in incremental_state:
                incremental_state["cache_k"] = torch.zeros(
                    (
                        self.args.max_batch_size,
                        self.cache_len,
                        self.n_kv_heads,
                        self.head_dim,
                    )
                ).to(xk)
                incremental_state["cache_v"] = torch.zeros(
                    (
                        self.args.max_batch_size,
                        self.cache_len,
                        self.n_kv_heads,
                        self.head_dim,
                    )
                ).to(xv)
            key = torch.cat([incremental_state["cache_k"][:, :start_pos], xk], dim=1)
            value = torch.cat([incremental_state["cache_v"][:, :start_pos], xv], dim=1)
            if key.shape[1] > self.cache_len:
                incremental_state["cache_k"][:bsz] = key[:, -self.cache_len:]
                incremental_state["cache_v"][:bsz] = value[:, -self.cache_len:]
            else:
                incremental_state["cache_k"][:bsz, start_pos : start_pos + seqlen] = xk
                incremental_state["cache_v"][:bsz, start_pos : start_pos + seqlen] = xv
        else:
            key, value = xk, xv

        output = flash_attn_func(xq, key, value, causal=True, window_size=(self.args.sliding_window - 1, 0) if self.activate_sliding_window else (-1, -1))

        return self.wo(output.view(bsz, seqlen, self.n_heads * self.head_dim))


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = ColumnParallelLinear(args.dim, args.hidden_dim, bias=False, gather_output=False, init_method=init_method)
        self.w2 = RowParallelLinear(args.hidden_dim, args.dim, bias=False, input_is_parallel=True, init_method=init_method)
        self.w3 = ColumnParallelLinear(args.dim, args.hidden_dim, bias=False, gather_output=False, init_method=init_method)

    def forward(self, x) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.attention = Attention(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.args = args

        self.feed_forward: nn.Module
        self.feed_forward = FeedForward(args=args)

    def forward(
        self, x: torch.Tensor, rel_pos: Tuple[torch.Tensor, torch.Tensor], start_pos: int, incremental_state = None
    ) -> torch.Tensor:
        r = self.attention.forward(self.attention_norm(x), rel_pos, start_pos, incremental_state)
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        mp_rank: int = 0,
        checkpoint_activations: bool = False
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        self._window_precomputed_freqs_cis: Optional[torch.Tensor] = None
        self._global_precomputed_freqs_cis: Optional[torch.Tensor] = None
        assert self.vocab_size > 0
        self.mp_rank = mp_rank
        self.checkpoint_activations = checkpoint_activations
        self.tok_embeddings = VocabParallelEmbedding(
            args.vocab_size, args.dim, -1, init_method=vocab_init_method
        )
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size // args.model_parallel_size, bias=False)
        # Initialize all layers but slice off those not of this rank.
        layers = [TransformerBlock(args=args) for idx in range(args.n_layers)]
        if checkpoint_activations:
            layers = [checkpoint_wrapper(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)
        self.n_local_layers = len(self.layers)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            theta = self.args.rope_theta
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, self.args.max_seq_len, theta
            )
        if self._precomputed_freqs_cis.device != self.device:
            self._precomputed_freqs_cis = self._precomputed_freqs_cis.to(
                device=self.device
            )
        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def forward_partial(
        self,
        input_ids: torch.Tensor,
        start_pos: Optional[int] = 0,
        incremental_state = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(input_ids)
        rel_pos = self.build_rel_pos(h, start_pos)
        for local_layer_id, layer in enumerate(self.layers):
            if incremental_state is not None:
                if local_layer_id not in incremental_state:
                    incremental_state[local_layer_id] = {}
            h = layer(h, rel_pos, start_pos, incremental_state=incremental_state[local_layer_id] if incremental_state is not None else None)

        return self.norm(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: Optional[int] = 0,
        incremental_state = None,
    ) -> torch.Tensor:
        h = self.forward_partial(input_ids, start_pos, incremental_state)
        if self.args.model_parallel_size > 1:
            h = copy_to_model_parallel_region(h)
        outs = self.output(h)
        return outs.float(), None

    def load_state_dict(self, state_dict, strict=False, assign=False):
        state_to_load = {}
        for k, v in state_dict.items():
            if k.startswith("tok_embeddings") or k.startswith("output"):
                state_to_load[k] = v.view(self.args.model_parallel_size, self.vocab_size // self.args.model_parallel_size, self.args.dim)[self.mp_rank]
            elif "wq" in k or "wk" in k or "wv" in k or "w1" in k or "w3" in k:
                state_to_load[k] = v.view(self.args.model_parallel_size, -1, v.shape[1])[self.mp_rank]
            elif "wo" in k or "w2" in k:
                state_to_load[k] = v.view(v.shape[0], self.args.model_parallel_size, -1)[:, self.mp_rank]
            else:
                state_to_load[k] = v
        super().load_state_dict(state_to_load, strict=False, assign=assign)
        print("Loaded state dict from checkpoint.")
