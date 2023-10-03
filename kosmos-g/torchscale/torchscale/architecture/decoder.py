# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import numpy as np
import torch
import torch.nn as nn
from apex.normalization import FusedLayerNorm as LayerNorm
from fairscale.nn import checkpoint_wrapper, wrap

from torchscale.architecture.utils import init_bert_params
from torchscale.component.droppath import DropPath
from torchscale.component.feedforward_network import FeedForwardNetwork, make_experts
from torchscale.component.multihead_attention import MultiheadAttention
from torchscale.component.relative_position_bias import RelativePositionBias
from torchscale.component.xmoe.moe_layer import MOELayer
from torchscale.component.xmoe.routing import Top1Gate, Top2Gate
from torchscale.component.sope_relative_position import SoPE


class DecoderLayer(nn.Module):
    def __init__(
            self,
            args,
            depth,
            is_moe_layer=False,
            is_encoder_decoder=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = torch.nn.Dropout(args.dropout, inplace=True)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.decoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.normalize_before = args.decoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        if not is_encoder_decoder:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)

        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.decoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = self.build_ffn(
                self.embed_dim,
                self.args,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    args.moe_eval_capacity_token_fraction,
                    use_xmoe=args.use_xmoe,
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim)
            self.moe_layer = MOELayer(gate, experts, args)

        self.final_layer_norm = LayerNorm(self.embed_dim)

        if args.deepnorm:
            if is_encoder_decoder:
                self.alpha = math.pow(3.0 * args.decoder_layers, 0.25)
            else:
                self.alpha = math.pow(2.0 * args.decoder_layers, 0.25)
        else:
            self.alpha = 1.0

    def build_ffn(self, embed_dim, args):
        return FeedForwardNetwork(
            embed_dim,
            self.ffn_dim,
            args.activation_fn,
            args.dropout,
            args.activation_dropout,
            args.subln,
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            args,
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False,
            encoder_decoder_attention=True,
            subln=args.subln,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(
            self,
            x,
            encoder_out=None,
            encoder_padding_mask=None,
            incremental_state=None,
            self_attn_mask=None,
            self_attn_padding_mask=None,
            self_attn_rel_pos=None,
            cross_attn_rel_pos=None,
            self_attn_sope_rel_pos=None,
            cross_attn_sope_rel_pos=None,
            scale=1.0,
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            attn_mask=self_attn_mask,
            rel_pos=self_attn_rel_pos,
            sope_rel_pos=self_attn_sope_rel_pos,
            scale=scale,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=None,
                rel_pos=cross_attn_rel_pos,
                sope_rel_pos=cross_attn_sope_rel_pos,
                scale=scale,
            )
            x = self.dropout_module(x)

            if self.drop_path is not None:
                x = self.drop_path(x)

            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        if not self.is_moe_layer:
            x = self.ffn(x)
            l_aux = None
        else:
            x = x.transpose(0, 1)
            x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x, attn, None, l_aux


class Decoder(nn.Module):
    def __init__(
            self,
            args,
            embed_tokens=None,
            embed_positions=None,
            output_projection=None,
            is_encoder_decoder=False,
            causal_mask=True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.args = args
        self.causal_mask = causal_mask

        self.dropout_module = torch.nn.Dropout(args.dropout, inplace=True)

        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        if (
                output_projection is None
                and not args.no_output_layer
                and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.decoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_decoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.output_projection = output_projection

        self.self_attn_relative_position = None
        self.cross_attn_relative_position = None
        self.self_attn_sope = None
        self.cross_attn_sope = None

        if args.sope_rel_pos:
            assert args.rel_pos_buckets == 0 and args.max_rel_pos == 0
            self.self_attn_sope = SoPE(
                args.decoder_embed_dim // args.decoder_attention_heads
            )
            if is_encoder_decoder:
                self.cross_attn_sope = SoPE(
                    args.decoder_embed_dim // args.decoder_attention_heads
                )

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.self_attn_relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.decoder_attention_heads,
            )
            if is_encoder_decoder:
                self.cross_attn_relative_position = RelativePositionBias(
                    num_buckets=args.rel_pos_buckets,
                    max_distance=args.max_rel_pos,
                    n_heads=args.decoder_attention_heads,
                )

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = math.pow(12.0 * args.decoder_layers, 0.25)
            else:
                init_scale = math.pow(8.0 * args.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                        "fc1" in name
                        or "fc2" in name
                        or "out_proj" in name
                        or "v_proj" in name
                ):
                    p.data.div_(init_scale)

        if args.subln:
            if is_encoder_decoder:
                init_scale = math.sqrt(math.log(args.decoder_layers * 3))
            else:
                init_scale = math.sqrt(math.log(args.decoder_layers * 2))
            for name, p in self.named_parameters():
                if "encoder_attn" in name:
                    continue
                if (
                        "fc1" in name
                        or "fc2" in name
                        or "out_proj" in name
                        or "v_proj" in name
                ):
                    p.data.mul_(init_scale)

    def build_output_projection(
            self,
            args,
    ):
        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim ** -0.5
            )
        return output_projection

    def build_decoder_layer(
            self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = DecoderLayer(
            args,
            depth,
            is_moe_layer=is_moe_layer,
            is_encoder_decoder=is_encoder_decoder,
        )
        if args.checkpoint_activations:
            layer = checkpoint_wrapper(layer, offload_to_cpu=False)
        if args.fsdp:
            layer = wrap(layer)
        return layer

    def forward_embedding(
            self,
            tokens,
            token_embedding=None,
            incremental_state=None,
    ):
        positions = None
        if self.embed_positions is not None:
            if tokens is not None:
                positions = self.embed_positions(
                    tokens, incremental_state=incremental_state
                )
            else:
                positions = self.embed_positions(
                    token_embedding, incremental_state=incremental_state
                )

        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        x = embed = self.embed_scale * token_embedding

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        return x, embed

    def forward(
            self,
            prev_output_tokens,
            self_attn_padding_mask=None,
            encoder_out=None,
            incremental_state=None,
            features_only=False,
            return_all_hiddens=False,
            token_embeddings=None,
            scale=1.0,
            **kwargs
    ):
        # embed tokens and positions
        x, _ = self.forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )
        x = x.transpose(0, 1)

        # relative postion
        self_attn_rel_pos_bias = None
        self_attn_sope_rel_pos = None
        slen = x.size(0)

        if self.self_attn_sope is not None:
            offset = 0 if incremental_state is None else incremental_state[0]["prev_key"].shape[2]
            self_attn_sope_rel_pos = self.self_attn_sope(slen, offset)

        if self.self_attn_relative_position is not None:
            self_attn_rel_pos_bias = self.self_attn_relative_position(
                batch_size=x.size(1), qlen=slen, klen=slen
            )
            if incremental_state is not None:
                self_attn_rel_pos_bias = self_attn_rel_pos_bias[:, -1:, :]

        cross_attn_rel_pos_bias = None
        cross_attn_sope_rel_pos = None
        if self.cross_attn_sope is not None:
            cross_attn_sope_rel_pos = self.cross_attn_sope(slen + encoder_out["encoder_out"].size(0))

        if self.cross_attn_relative_position is not None:
            cross_attn_rel_pos_bias = self.cross_attn_relative_position(
                batch_size=x.size(1),
                qlen=slen,
                klen=encoder_out["encoder_out"].size(0),
            )
            if incremental_state is not None:
                cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[:, -1:, :]

        # decoder layers
        inner_states = [x]

        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []

        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = torch.triu(
                    torch.zeros([x.size(0), x.size(0)])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(x),
                    1,
                ) if self.causal_mask else None
            else:
                self_attn_mask = None
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"]
                if encoder_out is not None and "encoder_padding_mask" in encoder_out else None,
                incremental_state[idx] if incremental_state is not None else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_rel_pos=self_attn_rel_pos_bias,
                cross_attn_rel_pos=cross_attn_rel_pos_bias,
                self_attn_sope_rel_pos=self_attn_sope_rel_pos,
                cross_attn_sope_rel_pos=cross_attn_sope_rel_pos,
                scale=scale,
            )
            l_aux.append(l_aux_i)
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)

        if not features_only and self.output_projection is not None:
            x = self.output_layer(x)

        return x, {
            "inner_states": inner_states,
            "l_aux": l_aux,
            "attn": [layer_attn.mean(dim=0)] if layer_attn is not None else None,
        }

    def output_layer(self, features):
        return self.output_projection(features)
