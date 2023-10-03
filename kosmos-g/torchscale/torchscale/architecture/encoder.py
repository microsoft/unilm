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
from torchscale.component.multiway_network import MultiwayWrapper, set_split_position
from torchscale.component.relative_position_bias import RelativePositionBias
from torchscale.component.xmoe.moe_layer import MOELayer
from torchscale.component.xmoe.routing import Top1Gate, Top2Gate


class EncoderLayer(nn.Module):
    def __init__(self, args, depth, is_moe_layer=False, is_encoder_decoder=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim))
        self.dropout_module = torch.nn.Dropout(args.dropout, inplace=True)

        if args.drop_path_rate > 0:
            drop_path_prob = np.linspace(0, args.drop_path_rate, args.encoder_layers)[
                depth
            ]
            self.drop_path = DropPath(drop_path_prob)
        else:
            self.drop_path = None

        self.normalize_before = args.encoder_normalize_before
        self.is_moe_layer = is_moe_layer
        self.ffn_dim = args.encoder_ffn_embed_dim

        if not self.is_moe_layer:
            self.ffn = MultiwayWrapper(
                args,
                self.build_ffn(
                    self.embed_dim,
                    self.args,
                ),
            )
        else:
            assert not self.args.multiway
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
        self.final_layer_norm = MultiwayWrapper(args, LayerNorm(self.embed_dim))

        if args.deepnorm:
            if is_encoder_decoder:
                self.alpha = (
                        math.pow(
                            math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                        )
                        * 0.81
                )
            else:
                self.alpha = math.pow(2.0 * args.encoder_layers, 0.25)
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
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            encoder_decoder_attention=False,
            subln=args.subln,
        )

    def residual_connection(self, x, residual):
        return residual * self.alpha + x

    def forward(self, x, encoder_padding_mask, attn_mask=None, rel_pos=None, scale=1.0):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
            rel_pos=rel_pos,
            scale=scale,
        )
        x = self.dropout_module(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

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
        return x, l_aux


class Encoder(nn.Module):
    def __init__(
            self,
            args,
            embed_tokens=None,
            embed_positions=None,
            output_projection=None,
            is_encoder_decoder=False,
            **kwargs
    ):
        self.args = args
        super().__init__(**kwargs)

        self.dropout_module = torch.nn.Dropout(args.dropout, inplace=True)

        embed_dim = args.encoder_embed_dim
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_tokens = embed_tokens
        self.embed_positions = embed_positions

        if (
                output_projection is None
                and not is_encoder_decoder
                and not args.no_output_layer
                and args.vocab_size > 0
        ):
            self.output_projection = self.build_output_projection(args)
        else:
            self.output_projection = output_projection

        if args.layernorm_embedding:
            self.layernorm_embedding = MultiwayWrapper(
                args, LayerNorm(embed_dim), dim=1
            )
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList([])

        moe_freq = args.moe_freq
        for i in range(args.encoder_layers):
            is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(
                self.build_encoder_layer(
                    args,
                    depth=i,
                    is_moe_layer=is_moe_layer,
                    is_encoder_decoder=is_encoder_decoder,
                )
            )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = MultiwayWrapper(args, LayerNorm(embed_dim))
        else:
            self.layer_norm = None

        if args.rel_pos_buckets > 0 and args.max_rel_pos > 0:
            self.relative_position = RelativePositionBias(
                num_buckets=args.rel_pos_buckets,
                max_distance=args.max_rel_pos,
                n_heads=args.encoder_attention_heads,
            )
        else:
            self.relative_position = None

        if args.bert_init:
            self.apply(init_bert_params)

        if args.deepnorm:
            if is_encoder_decoder:
                init_scale = (
                        math.pow(
                            math.pow(args.encoder_layers, 4) * args.decoder_layers, 0.0625
                        )
                        / 1.15
                )
            else:
                init_scale = math.pow(8.0 * args.encoder_layers, 0.25)
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
                init_scale = math.sqrt(
                    math.log(3 * args.decoder_layers)
                    * math.log(2 * args.encoder_layers)
                    / 3
                )
            else:
                init_scale = math.sqrt(math.log(args.encoder_layers * 2))
            for name, p in self.named_parameters():
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
        if args.share_encoder_input_output_embed:
            assert args.encoder_embedding_type == "language"
            output_projection = torch.nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = self.embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.encoder_embed_dim, args.vocab_size, bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.encoder_embed_dim ** -0.5
            )
        return output_projection

    def build_encoder_layer(
            self, args, depth, is_moe_layer=False, is_encoder_decoder=False
    ):
        layer = EncoderLayer(
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
            src_tokens,
            token_embedding=None,
    ):
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            if src_tokens is not None:
                x = embed + self.embed_positions(src_tokens)
            else:
                x = embed + self.embed_positions(x)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        return x, embed

    def forward(
            self,
            src_tokens,
            encoder_padding_mask=None,
            return_all_hiddens=False,
            token_embeddings=None,
            multiway_split_position=None,
            features_only=False,
            scale=1.0,
            **kwargs
    ):
        assert src_tokens is not None or token_embeddings is not None

        if encoder_padding_mask is None:
            if src_tokens is not None:
                encoder_padding_mask = torch.zeros_like(
                    src_tokens, device=src_tokens.device
                ).bool()
            else:
                encoder_padding_mask = torch.zeros(
                    [token_embeddings.size(0), token_embeddings.size(1)],
                    device=token_embeddings.device,
                ).bool()

        if multiway_split_position is not None:
            assert self.args.multiway
            self.apply(set_split_position(multiway_split_position))

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        rel_pos_bias = None
        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(
                batch_size=x.size(1), qlen=x.size(0), klen=x.size(0)
            )

        l_aux = []
        for layer in self.layers:
            x, l_aux_i = layer(
                x, encoder_padding_mask=encoder_padding_mask, rel_pos=rel_pos_bias, scale=scale
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            l_aux.append(l_aux_i)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if not features_only and self.output_projection is not None:
            x = self.output_projection(x)

        return {
            "encoder_out": x,
            "encoder_embedding": encoder_embedding,
            "encoder_padding_mask": encoder_padding_mask,
            "encoder_states": encoder_states,
            "l_aux": l_aux,
        }
