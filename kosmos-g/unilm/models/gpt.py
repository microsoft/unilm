from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from torch import Tensor

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import distributed_utils, utils
from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.utils import safe_getattr, safe_hasattr

from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer_lm import (
    TransformerLanguageModelConfig,
    TransformerLanguageModel,
    base_gpt3_architecture,
)
from fairseq.models.transformer.transformer_decoder import TransformerDecoder
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
from fairseq.modules import PositionalEmbedding
from omegaconf import II

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import TextEmbedding

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class GPTModelConfig(TransformerLanguageModelConfig):
    scale_final_logits: bool = field(
        default=False,
        metadata={
            "help": "scale final logits by sqrt(d)"
        },
    )

    gpt_model_path: str = field(
        default="",
        metadata={"help": "gpt checkpoint path"},
    )
    rescale_init: bool = field(
        default=False,
        metadata={
            "help": "whether to use rescale initialization"
        },
    )
    deepnet: bool = field(
        default=False,
        metadata={
            "help": "enable deepnet in decoder"
        },
    )
    last_ln_scale: bool = field(
        default=False,
        metadata={
            "help": "enable last_ln_scale in decoder"
        },
    )

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    ddp_rank: int = II("distributed_training.distributed_rank")

    deepnorm: Optional[bool] = field(
        default=False,
    )
    subln: Optional[bool] = field(
        default=False,
    )
    rel_pos_buckets: Optional[int] = field(
        default=0,
    )
    max_rel_pos: Optional[int] = field(
        default=0,
    )
    flash_attention: bool = field(
        default=False,
    )
    sope_rel_pos: Optional[bool] = field(
        default=False,
        metadata={"help": "use SoPE as the relative position embhedding"},
    )
    scale_length: Optional[int] = field(
        default=2048,
    )
    max_chunk_emb: Optional[int] = field(
        default=0,
        metadata={"help": "chunk embedding, text image text image text: 0, 1, 1, 2, 2"},
    )
    segment_emb: Optional[bool] = field(
        default=False,
    )


@register_model("gptmodel", dataclass=GPTModelConfig)
class GPTmodel(TransformerLanguageModel):

    @classmethod
    def build_model(cls, args, task):
        model = TransformerLanguageModel.build_model(args, task)

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(
                args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS
            )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.decoder_embed_dim
        )

        embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                args.decoder_embed_dim,
                task.dictionary.pad(),
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if args.share_decoder_input_output_embed:
            output_projection = torch.nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            output_projection.weight = embed_tokens.weight
        else:
            output_projection = torch.nn.Linear(
                args.decoder_embed_dim, len(task.dictionary), bias=False
            )
            torch.nn.init.normal_(
                output_projection.weight, mean=0, std=args.decoder_embed_dim ** -0.5
            )

        if getattr(args, "moe_freq", 0) > 0 and (
                getattr(args, "fp16", False)
                and not getattr(args, "memory_efficient_fp16", False)
                and getattr(args, "ddp_backend", None) != "fully_sharded"
        ):
            assert (
                args.fp16_no_flatten_grads
            ), "If training moe models, set --fp16-no-flatten-grads to calculate correct gradnorm"

        args.ddp_rank = distributed_utils.get_data_parallel_rank()

        config = DecoderConfig()
        config.override(args)

        decoder = LMDecoder(
            config,
            embed_tokens,
            embed_positions,
            output_projection,
            is_encoder_decoder=False,
            dictionary=task.dictionary,
        )
        decoder.chunk_emb = None
        if args.max_chunk_emb > 0:
            decoder.chunk_emb = TextEmbedding(args.max_chunk_emb, args.decoder_embed_dim)
        decoder.segment_emb = None
        if args.segment_emb:
            decoder.segment_emb = TextEmbedding(2, args.decoder_embed_dim)
        model.decoder = decoder
        if args.gpt_model_path != "":
            assert NotImplementedError
            # state = checkpoint_utils.load_checkpoint_to_cpu(args.gpt_model_path)
            # model.load_state_dict(state["model"], strict=True, args=args)
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        return Embedding(len(dictionary), embed_dim, dictionary.pad())


class LMDecoder(Decoder, FairseqIncrementalDecoder):
    def forward(self, src_tokens, **kwargs):
        self_attn_padding_mask = src_tokens.eq(self.dictionary.pad())
        return super().forward(src_tokens, self_attn_padding_mask, **kwargs)

    def max_positions(self):
        return self.embed_positions.max_positions

    def reorder_incremental_state_scripting(
            self,
            incremental_state,
            new_order,
    ):
        for module in incremental_state:
            for key in incremental_state[module]:
                result = incremental_state[module][key].index_select(0, new_order)
                incremental_state[module][key] = result

    def forward_embedding(
            self,
            tokens,
            token_embedding=None,
            incremental_state=None,
            mlm_features: Optional[Tensor] = None,
            gpt_input_mask: Optional[Tensor] = None,
            img_features: Optional[Tensor] = None,
            img_gpt_input_mask: Optional[Tensor] = None,
            aud_features: Optional[Tensor] = None,
            aud_gpt_input_mask: Optional[Tensor] = None,
            chunk_tokens: Optional[Tensor] = None,
            segment_tokens: Optional[Tensor] = None,
    ):
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                tokens, incremental_state=incremental_state
            )
            if self.chunk_emb is not None:
                chunk_emb = self.chunk_emb(chunk_tokens)
                positions += chunk_emb
            if self.segment_emb is not None:
                segment_emb = self.segment_emb(segment_tokens)
                positions += segment_emb

        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        if token_embedding is None:
            token_embedding = self.embed_tokens(tokens)

        gpt_embed_output = token_embedding
        if mlm_features is not None:
            gpt_embed_output[gpt_input_mask] = mlm_features
        if img_features is not None:
            gpt_embed_output[img_gpt_input_mask] = img_features
        if aud_features is not None:
            gpt_embed_output[aud_gpt_input_mask] = aud_features

        x = embed = self.embed_scale * gpt_embed_output

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
            prev_output_tokens, token_embeddings, incremental_state, **kwargs
        )
        x = x.transpose(0, 1)

        # relative position
        self_attn_rel_pos_bias = None
        slen = prev_output_tokens.size(1)
        if self.self_attn_relative_position is not None:
            self_attn_rel_pos_bias = self.self_attn_relative_position(
                batch_size=x.size(1), qlen=slen, klen=slen
            )
            if incremental_state is not None:
                self_attn_rel_pos_bias = self_attn_rel_pos_bias[:, -1:, :]
        cross_attn_rel_pos_bias = None
        if self.cross_attn_relative_position is not None:
            cross_attn_rel_pos_bias = self.cross_attn_relative_position(
                batch_size=x.size(1),
                qlen=slen,
                klen=encoder_out["encoder_out"].size(0),
            )
            if incremental_state is not None:
                cross_attn_rel_pos_bias = cross_attn_rel_pos_bias[:, -1:, :]

        # decoder layers
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
                )
            else:
                self_attn_mask = None
                if idx not in incremental_state:
                    incremental_state[idx] = {}

            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out["encoder_out"] if encoder_out is not None else None,
                encoder_out["encoder_padding_mask"] if encoder_out is not None else None,
                incremental_state[idx] if incremental_state is not None else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                self_attn_rel_pos=self_attn_rel_pos_bias,
                cross_attn_rel_pos=cross_attn_rel_pos_bias,
                scale=scale,
            )
            l_aux.append(l_aux_i)

        x = self.layer_norm(x) if self.layer_norm is not None else x

        if not features_only:
            logits = self.output_layer(x.transpose(0, 1))
        else:
            logits = None

        return logits, x, {
            "l_aux": l_aux,
            "attn": [layer_attn.mean(dim=0) if layer_attn is not None else None],
        }


@register_model_architecture("gptmodel", "gptmodel_small")
def gptmodel_small(args):
    # 125M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    base_gpt3_architecture(args)


@register_model_architecture("gptmodel", "gptmodel_medium")
def gptmodel_medium(args):
    # 350M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    base_gpt3_architecture(args)
