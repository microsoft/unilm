import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn import checkpoint_wrapper

from fairseq.model_parallel.megatron.mpu import (
    ColumnParallelLinear,
    copy_to_model_parallel_region,
    VocabParallelEmbedding
)

from .gate_retention import GateRetention
from .sliding_window_attention import SlidingWindowAttention
from .cross_attention import CrossAttention
from .feedforward_network import FeedForwardNetwork, init_method
from .rms_norm import RMSNorm

from .kernel.rotary import apply_rotary_emb
from .model_parallel_init import vocab_init_method, init_method


@dataclass
class YOCOArgs:
    dim: int
    n_layers: int
    hidden_dim: int
    n_self_heads: int
    n_attn_heads: int
    n_attn_kv_heads: int
    vocab_size: int

    max_batch_size: int = 0
    max_seq_len: int = -1
    model_parallel_size: int = 1
    load_checkpoint: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    sliding_window: Optional[int] = None

class DecoderLayer(nn.Module):
    def __init__(
        self,
        args: YOCOArgs,
        is_cross_layer=False
    ):
        super().__init__()
        self.args = args
        self.is_cross_layer = is_cross_layer

        if is_cross_layer:
            self.mixer = CrossAttention(args)
        elif args.sliding_window is not None:
            self.mixer = SlidingWindowAttention(args)
        else:
            self.mixer = GateRetention(args) 

        self.mixer_layer_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.ffn =  FeedForwardNetwork(
            args.dim,
            args.hidden_dim,
            args.load_checkpoint
        )

        self.final_layer_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x,
        start_pos=0,
        key=None,
        value=None,
        rel_pos=None,
        incremental_state=None,
        is_prefilling=False,
    ):
        residual = x
        x = self.mixer_layer_norm(x)

        if self.is_cross_layer:
            x = self.mixer(
                x,
                key,
                value,
                rel_pos=rel_pos,
            )
        elif self.args.sliding_window is not None:
            x = self.mixer(
                x,
                rel_pos=rel_pos,
                start_pos=start_pos,
                incremental_state=incremental_state,
            )
        else:
            x = self.mixer(
                x,
                rel_pos=rel_pos,
                incremental_state=incremental_state,
                is_prefilling=is_prefilling,)

        x = x + residual
        residual = x
        x = self.final_layer_norm(x)

        x = self.ffn(x)

        x = x + residual
        return x

class SelfDecoder(nn.Module):
    def __init__(
        self,
        args: YOCOArgs,
        checkpoint_activations: bool = False
    ):
        super().__init__()
        self.args = args 
        layers = [DecoderLayer(args, is_cross_layer=False,) for idx in range(args.n_layers // 2)]
        if checkpoint_activations:
            layers = [checkpoint_wrapper(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)
        self.head_dim = args.dim // args.n_self_heads
        self.block_size = 256
        self._precomputed_freqs_cis = None

    def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / (self.args.rope_theta ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(self.args.max_seq_len).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def get_index_mask(self, x, length, pad_length):
        return torch.arange(pad_length, device=x.device) >= length

    def forward(
        self,
        x,
        incremental_state=None,
        is_prefilling=False,
        start_pos=0
    ):
        if is_prefilling and x.size(1) % self.block_size != 0 and self.args.sliding_window is None:
            padding_len = self.block_size - x.size(1) % self.block_size
            x = F.pad(x, (0, 0, 0, padding_len), value=0)
        else:
            padding_len = 0

        if incremental_state is not None and is_prefilling:
            index_mask = self.get_index_mask(x, x.size(1) - padding_len, x.size(1))

        rel_pos = self.build_rel_pos(x, start_pos)
        for idx, layer in enumerate(self.layers):
            if incremental_state is not None:
                if idx not in incremental_state:
                    incremental_state[idx] = {}
                if is_prefilling:
                    incremental_state[idx]["index_mask"] = index_mask
            x = layer(
                x,
                start_pos=start_pos,
                rel_pos=rel_pos,
                incremental_state=incremental_state[idx] if incremental_state is not None else None,
                is_prefilling=is_prefilling,)

        x = x[:, :x.size(1) - padding_len, :]
        return x

class CrossDecoder(nn.Module):
    def __init__(
        self,
        args: YOCOArgs,
        checkpoint_activations: bool = False
    ):
        super().__init__()
        self.args = args
        self.num_heads = args.n_attn_kv_heads
        self.head_dim = args.dim // args.n_attn_heads
        self.k_proj = ColumnParallelLinear(args.dim, self.head_dim * args.n_attn_kv_heads, bias=False, gather_output=False, init_method=init_method)
        self.v_proj = ColumnParallelLinear(args.dim, self.head_dim * args.n_attn_kv_heads, bias=False, gather_output=False, init_method=init_method)
        self.kv_layer_norm = RMSNorm(args.dim, eps=args.norm_eps)
        layers = [DecoderLayer(args, is_cross_layer=True) for idx in range(args.n_layers // 2)]
        if checkpoint_activations:
            layers = [checkpoint_wrapper(layer) for layer in layers]
        self.layers = nn.ModuleList(layers)
        self._precomputed_freqs_cis = None

    def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / (self.args.rope_theta ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(self.args.max_seq_len).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos

    def forward(
        self,
        x,
        incremental_state=None,
        start_pos=0,
        skip_cross_decoder=False,
    ):
        bsz, seqlen, embed_dim = x.size()
        x_norm = self.kv_layer_norm(x)
        key, value = self.k_proj(x_norm), self.v_proj(x_norm)
        key = key.view(bsz, seqlen, self.num_heads, self.head_dim)
        value = value.view(bsz, seqlen, self.num_heads, self.head_dim)
        rel_pos = self.build_rel_pos(x, start_pos)
        key = apply_rotary_emb(key, *rel_pos, interleaved=True)
        if incremental_state is not None:
            if "prev_key" not in incremental_state:
                incremental_state["prev_key"] = torch.empty(bsz, self.args.max_seq_len, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
                incremental_state["prev_value"] = torch.empty(bsz, self.args.max_seq_len, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
            incremental_state["prev_key"][:, start_pos : start_pos + seqlen] = key
            incremental_state["prev_value"][:, start_pos : start_pos + seqlen] = value
            key = incremental_state["prev_key"][:, : start_pos + seqlen]
            value = incremental_state["prev_value"][:, : start_pos + seqlen]
        
        if skip_cross_decoder:
            return torch.zeros(bsz, 1, embed_dim, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x = layer(
                x,
                key=key,
                value=value,
                rel_pos=rel_pos)

        return x

class YOCO(nn.Module):
    def __init__(
        self,
        args: YOCOArgs,
        checkpoint_activations: bool = False,
        share_input_output_embed: bool = False,
    ):
        super().__init__()
        self.args = args
        self.embed_scale = math.sqrt(args.dim)
        self.embed_tokens = VocabParallelEmbedding(
            args.vocab_size, args.dim, -1, init_method=vocab_init_method
        )
        self.output_projection = nn.Linear(args.dim, args.vocab_size, bias=False)
        if share_input_output_embed:
            self.output_projection.weight = self.embed_tokens.weight

        self.self_decoder = SelfDecoder(args, checkpoint_activations)
        self.cross_decoder = CrossDecoder(args, checkpoint_activations)
        self.layer_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x,
        start_pos=0,
        incremental_state=None,
        is_prefilling=True,
        skip_cross_decoder=False
    ): 
        x = self.embed_scale * self.embed_tokens(x)

        x = self.self_decoder(
            x,
            incremental_state=incremental_state,
            is_prefilling=is_prefilling,
            start_pos=start_pos,
        )

        x = self.cross_decoder(
            x,
            start_pos=start_pos,
            incremental_state=incremental_state,
            skip_cross_decoder=skip_cross_decoder,
        )
            
        x = self.layer_norm(x)
        x = self.output_layer(x)

        return x, None

    def output_layer(self, features):
        if self.args.model_parallel_size > 1:
            features = copy_to_model_parallel_region(features)
        return self.output_projection(features)