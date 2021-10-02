import torch.nn as nn
from fairseq.models import FairseqEncoder, register_model, FairseqEncoderDecoderModel, register_model_architecture
from fairseq.models.transformer import TransformerDecoder, Embedding, TransformerModel
from fairseq.models.transformer import base_architecture as base_transformer
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import MultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from torch.nn import Parameter
from fairseq import utils
from torch import Tensor

import torch
from torch.hub import load_state_dict_from_url

from timm.models import create_model

from functools import partial
import logging
import argparse
from typing import Dict, Optional, Tuple
import re

logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024

class UniLMMultiheadAttention(MultiheadAttention):
    def __init__(
        self,         
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8):
        super().__init__(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, self_attention=self_attention, encoder_decoder_attention=encoder_decoder_attention, q_noise=q_noise, qn_block_size=qn_block_size)
        self.qk_head_dim = 96
        self.scaling = self.qk_head_dim ** -0.5

        qk_output_dim = self.qk_head_dim * self.num_heads
        assert qk_output_dim == 1152
        self.k_proj = quant_noise(
            nn.Linear(self.kdim, qk_output_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, qk_output_dim, bias=bias), q_noise, qn_block_size
        )
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, qk_output_dim))
        else:
            self.bias_k = None
    
    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        # Disabled 
        # if (
        #     not self.onnx_trace
        #     and not is_tpu  # don't use PyTorch version on TPUs
        #     and incremental_state is None
        #     and not static_kv
        #     # A workaround for quantization to work. Otherwise JIT compilation
        #     # treats bias in linear module as method.
        #     and not torch.jit.is_scripting()
        # ):
        #     assert key is not None and value is not None
        #     return F.multi_head_attention_forward(
        #         query,
        #         key,
        #         value,
        #         self.embed_dim,
        #         self.num_heads,
        #         torch.empty([0]),
        #         torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
        #         self.bias_k,
        #         self.bias_v,
        #         self.add_zero_attn,
        #         self.dropout_module.p,
        #         self.out_proj.weight,
        #         self.out_proj.bias,
        #         self.training or self.dropout_module.apply_during_inference,
        #         key_padding_mask,
        #         need_weights,
        #         attn_mask,
        #         use_separate_proj_weight=True,
        #         q_proj_weight=self.q_proj.weight,
        #         k_proj_weight=self.k_proj.weight,
        #         v_proj_weight=self.v_proj.weight,
        #     )

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.qk_head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                .view(-1, bsz * self.num_heads, self.qk_head_dim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.qk_head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.qk_head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

from argparse import Namespace
from omegaconf import DictConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

@register_model('DeiT_TR')
class DeiTTRModel(FairseqEncoderDecoderModel):

    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):

        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)

        from fairseq.checkpoint_utils import prune_state_dict

        new_state_dict = prune_state_dict(state_dict, model_cfg)
        if not model_cfg.ape:
            model_seq_len = self.state_dict()['encoder.deit.pos_embed'].shape[1]
            ckpt_seq_len = new_state_dict['encoder.deit.pos_embed'].shape[1]
            logger.info('Load from {:d} seq len to {:d}'.format(ckpt_seq_len, model_seq_len))
            if model_seq_len <= ckpt_seq_len:
                new_state_dict['encoder.deit.pos_embed'] = new_state_dict['encoder.deit.pos_embed'][:, :model_seq_len, :]
            else:
                t = self.state_dict()['encoder.deit.pos_embed']
                t[:, :ckpt_seq_len, :] = new_state_dict['encoder.deit.pos_embed']
                new_state_dict['encoder.deit.pos_embed'] = t

        return super().load_state_dict(new_state_dict, strict=False)
    
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument(
            '--deit-arch', type=str,
            help='the arch name for the DeiT encoder'
        )        
        parser.add_argument(
            '--ape', action='store_true',
            help='if use absolute_pos_embed'
        )        
        parser.set_defaults(ape=False)
        parser.add_argument(
            '--mask-ratio', default=0.0, type=float,
            help='the mask ratio for the encoder output masking.'
        )

    @staticmethod
    def read_args_from_roberta(roberta_args: argparse.Namespace):
        # TODO: this would become easier if encoder/decoder where using a similar
        # TransformerConfig object
        args = argparse.Namespace(**vars(roberta_args))
        attr_map = [
            ("encoder_attention_heads", "decoder_attention_heads"),
            ("encoder_embed_dim", "decoder_embed_dim"),
            ("encoder_embed_dim", "decoder_output_dim"),
            ("encoder_normalize_before", "decoder_normalize_before"),
            ("encoder_layers_to_keep", "decoder_layers_to_keep"),
            ("encoder_ffn_embed_dim", "decoder_ffn_embed_dim"),
            ("encoder_layerdrop", "decoder_layerdrop"),
            ("encoder_layers", "decoder_layers"),
            ("encoder_learned_pos", "decoder_learned_pos"),
            # should this be set from here ?
            ("max_positions", "max_target_positions"),
        ]
        for k1, k2 in attr_map:
            setattr(args, k2, getattr(roberta_args, k1))

        args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
        args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
        args.share_decoder_input_output_embed = not roberta_args.untie_weights_roberta
        return args

    @classmethod
    def build_model(cls, args, task):
        encoder = DeiTTREncoder(
            args = args,
            dictionary = task.source_dictionary
        )

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        decoder_embed_tokens = cls.build_embedding(
            args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
        )

        if getattr(args, "decoder_pretrained", None) == 'unilm':
            args.decoder_attention_heads = 12

        if getattr(args, "decoder_pretrained", None).startswith('roberta2'):         
            logger.info('Using the tengchao version loading roberta.')
            pretrained_model = getattr(args, "decoder_pretrained", None)
            specified = pretrained_model.find('-')!=-1

            if specified:
                pretrained_model = pretrained_model.replace('-', '.')
                logger.info('Load pre-trained decoder parameters from {}'.format(pretrained_model))
                roberta = torch.hub.load('pytorch/fairseq', pretrained_model)
            elif args.decoder_layers == 6:
                logger.info('Load pre-trained decoder parameters from roberta.base')
                roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
            elif args.decoder_layers == 12:
                logger.info('Load pre-trained decoder parameters from roberta.large')
                roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
            else:
                raise AttributeError('Cannot determind the pre-trained model')

            roberta.model.args.encoder_layers = args.decoder_layers
            roberta.model.args.fp16 = args.fp16
            roberta_args = DeiTTRModel.read_args_from_roberta(roberta.model.args)
            roberta_args.encoder_embed_dim = args.encoder_embed_dim

            decoder = TransformerDecoder(
                roberta_args,
                task.target_dictionary,
                decoder_embed_tokens,
                no_encoder_attn=False,
            )

            roberta_layers = roberta.model.encoder.sentence_encoder.layers
            decoder_layers = decoder.layers
            offset = len(roberta_layers) - len(decoder_layers)
            assert offset >= 0


            decoder_dict = roberta.state_dict()
            new_decoder_dict = {}
            for key, val in decoder_dict.items():
                if key.startswith('model.encoder.sentence_encoder.layers.'):
                    layer_num = int(key[len('model.encoder.sentence_encoder.layers.'):].split('.')[0])
                    if layer_num - offset < 0:
                        continue
                    else:
                        new_key = 'model.encoder.sentence_encoder.layers.{}.'.format(
                            str(layer_num - offset)) + '.'.join(
                            key[len('model.encoder.sentence_encoder.layers.'):].split('.')[1:])
                        new_decoder_dict[new_key] = val
                else:
                    new_decoder_dict[key] = val
            decoder_dict = new_decoder_dict

            for k, w in list(decoder_dict.items()):
                if '.lm_head' in k:
                    k_proj = "output_projection." + k[len('model.encoder.lm_head.'):]
                    decoder_dict[k_proj] = w.detach().clone()
                    del decoder_dict[k]

            del decoder_dict['_float_tensor']
            del decoder_dict['output_projection.weight']
            del decoder_dict['output_projection.bias']
            del decoder_dict['output_projection.dense.weight']
            del decoder_dict['output_projection.dense.bias']
            del decoder_dict['output_projection.layer_norm.weight']
            del decoder_dict['output_projection.layer_norm.bias']

            new_decoder_dict = {}
            for key, val in decoder_dict.items():
                if "sentence_encoder" in key:
                    key = key[len('model.encoder.sentence_encoder.'):]
                elif "encoder" in key:
                    key = key[len('model.encoder.'):]
                new_decoder_dict[key] = val

            missing_keys, unexpected_keys = decoder.load_state_dict(
                new_decoder_dict, strict=False
            )

        elif getattr(args, "decoder_pretrained", None).startswith('roberta'):          
            decoder = TransformerDecoder(
                args = args,
                dictionary=task.target_dictionary,
                embed_tokens=decoder_embed_tokens,
                no_encoder_attn=False
            )

            pretrained_model = getattr(args, "decoder_pretrained", None)
            specified = pretrained_model.find('-')!=-1

            if specified:
                pretrained_model = pretrained_model.replace('-', '.')
                logger.info('Load pre-trained decoder parameters from {}'.format(pretrained_model))
                roberta = torch.hub.load('pytorch/fairseq', pretrained_model)
            elif args.decoder_layers == 6:
                logger.info('Load pre-trained decoder parameters from roberta.base')
                roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
            elif args.decoder_layers == 12:
                logger.info('Load pre-trained decoder parameters from roberta.large')
                roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
            else:
                raise AttributeError('Cannot determind the pre-trained model')

            decoder.embed_tokens.load_state_dict(roberta.model.encoder.sentence_encoder.embed_tokens.state_dict())
            roberta_layers = roberta.model.encoder.sentence_encoder.layers
            decoder_layers = decoder.layers
            offset = len(roberta_layers) - len(decoder_layers)
            assert offset >= 0

            for i in range(len(decoder_layers)):
                roberta_i = i + offset
                decoder_layers[i].self_attn.load_state_dict(roberta_layers[roberta_i].self_attn.state_dict())
                decoder_layers[i].self_attn_layer_norm.load_state_dict(roberta_layers[roberta_i].self_attn_layer_norm.state_dict())

        model = cls(encoder, decoder)
        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    def forward(self, imgs, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(imgs, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out


# @register_model_architecture('DeiT_TR', 'DeiT_TR_base')
# def DeiT_TR_base(args):
#     # DeiT Encoder  deit_base_distilled_patch16_224
#     args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_224")
#     # Transformer Decoder
#     args.encoder_embed_dim = 768
#     base_transformer(args)

@register_model_architecture('DeiT_TR', 'deit_base_decoder_base')
def deit_base_decoder_base(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    base_transformer(args)

# @register_model_architecture('DeiT_TR', 'DeiT_TR_large_12layers')
# def DeiT_TR_large_12layers(args):
#     # DeiT Encoder  deit_base_distilled_patch16_384
#     args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
#     # Transformer Decoder
#     args.encoder_embed_dim = 768
#     args.decoder_layers = getattr(args, "decoder_layers", 12)
#     base_transformer(args)

@register_model_architecture('DeiT_TR', 'deit_base_decoder_large')
def deit_base_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'beit_base_decoder_large')
def beit_base_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_base_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'beit_large_decoder_large')
def beit_large_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_large_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 1024
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'DeiT_TR_LargeR_BEiT_Large')
def beit_large_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_large_patch16_384")
    # Transformer Decoder
    args.encoder_embed_dim = 1024
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'deit_base_decoder_large_custom_size')
def deit_base_decoder_large_custom_size(args):
    # DeiT Encoder  deit_base_distilled_patch16_custom_size
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_custom_size")
    # Transformer Decoder
    args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)


class DeiTTREncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        
        if 'custom_size' in args.deit_arch:
            self.deit = create_model(args.deit_arch, pretrained=True, img_size=args.input_size, ape=args.ape, mask_ratio=args.mask_ratio)
        else:
            self.deit = create_model(args.deit_arch, pretrained=True, ape=args.ape, mask_ratio=args.mask_ratio)
        
        self.fp16 = args.fp16

    def forward(self, imgs):
        if self.fp16:
            imgs = imgs.half()

        x, encoder_embedding = self.deit.forward_features(imgs)  # bs, n + 2, dim
        x = x.transpose(0, 1) # n + 2, bs, dim

        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def reorder_encoder_out(self, encoder_out, new_order):
          """
          Reorder encoder output according to `new_order`.

          Args:
              encoder_out: output from the ``forward()`` method
              new_order (LongTensor): desired order

          Returns:
              `encoder_out` rearranged according to `new_order`
          """
          _encoder_out = encoder_out['encoder_out'][0]
          _encoder_padding_mask = encoder_out['encoder_padding_mask'][0]
          _encoder_embedding = encoder_out['encoder_embedding'][0]
          return {
              "encoder_out": [_encoder_out.index_select(1, new_order)],
                "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
                "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
                "encoder_states": [], 
                "src_tokens": [],
                "src_lengths": [],
        }

    

if __name__ == '__main__':
    pass
