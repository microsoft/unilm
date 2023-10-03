# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm as LayerNorm
from torch import nn
from diffusers.models.attention_processor import LoRALinearLayer

from .multiway_network import MultiwayWrapper
from xformers.ops import memory_efficient_attention, LowerTriangularMask, MemoryEfficientAttentionCutlassOp


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class MultiheadAttention(nn.Module):
    def __init__(
            self,
            args,
            embed_dim,
            num_heads,
            dropout=0.0,
            self_attention=False,
            encoder_decoder_attention=False,
            subln=False,
    ):
        super().__init__()
        self.args = args
        self.lora = hasattr(args, 'lora') and args.lora
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.scale_length = args.scale_length

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.q_proj = MultiwayWrapper(args, nn.Linear(embed_dim, embed_dim, bias=True))
        self.out_proj = MultiwayWrapper(
            args, nn.Linear(embed_dim, embed_dim, bias=True)
        )
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.embed_dim))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout, inplace=True)

        if self.lora:
            self.to_q_lora = LoRALinearLayer(embed_dim, embed_dim, rank=4)
            self.to_k_lora = LoRALinearLayer(embed_dim, embed_dim, rank=4)
            self.to_v_lora = LoRALinearLayer(embed_dim, embed_dim, rank=4)
            self.to_out_lora = LoRALinearLayer(embed_dim, embed_dim, rank=4)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
            self,
            query,
            key,
            value,
            incremental_state=None,
            key_padding_mask=None,
            attn_mask=None,
            rel_pos=None,
            sope_rel_pos=None,
            scale=1.0,
    ):
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        src_len, key_bsz, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert src_len, bsz == value.shape[:2]

        if self.lora:
            q = self.q_proj(query) + scale * self.to_q_lora(query)
            k = self.k_proj(key) + scale * self.to_k_lora(key)
            v = self.v_proj(value) + scale * self.to_v_lora(value)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1).contiguous()
        k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1).contiguous()
        v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1).contiguous()

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                prev_value = incremental_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            incremental_state["prev_value"] = v.view(
                bsz, self.num_heads, -1, self.head_dim
            )
            src_len = k.size(1)

        if sope_rel_pos is not None:
            assert rel_pos is None
            sin, cos, scale = sope_rel_pos
            if self.self_attention:
                k = apply_rotary_pos_emb(k, sin, cos, scale=1 / scale)
                q = apply_rotary_pos_emb(q, sin[-q.shape[1]:], cos[-q.shape[1]:], scale=scale[-q.shape[1]:])
            else:
                k = apply_rotary_pos_emb(k, sin[:k.shape[1]], cos[:k.shape[1]], scale=1 / scale[:k.shape[1]])
                q = apply_rotary_pos_emb(q, sin[k.shape[1]:], cos[k.shape[1]:], scale=scale[k.shape[1]:])

            if k.shape[1] > self.scale_length:
                scale_attention = torch.maximum(torch.ones(q.shape[1]),
                                                torch.arange(k.shape[1] - q.shape[1], k.shape[1], 1).log() / math.log(
                                                    self.scale_length)).to(q)
                q = q * scale_attention.unsqueeze(-1)

        if self.args.flash_attention and rel_pos is None and attn_mask is not None:
            attn_bias = LowerTriangularMask()
            attn = memory_efficient_attention(q, k, v, attn_bias, op=MemoryEfficientAttentionCutlassOp)
            attn_weights = None
        else:
            q *= self.scaling
            attn_weights = torch.bmm(q, k.transpose(1, 2))

            if attn_mask is not None:
                attn_weights = torch.nan_to_num(attn_weights)
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask

            if key_padding_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            if rel_pos is not None:
                rel_pos = rel_pos.view(attn_weights.size())
                attn_weights = attn_weights + rel_pos

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
                attn_weights
            )
            attn_probs = self.dropout_module(attn_weights)

            attn = torch.bmm(attn_probs, v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        if self.inner_attn_ln is not None:
            attn = self.inner_attn_ln(attn)

        if self.lora:
            attn = self.out_proj(attn) + scale * self.to_out_lora(attn)
        else:
            attn = self.out_proj(attn)

        if attn_weights is not None:
            attn_weights = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)

        return attn, attn_weights
