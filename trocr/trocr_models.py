from fairseq.models import FairseqEncoder, register_model, FairseqEncoderDecoderModel, register_model_architecture
from fairseq.models.transformer import TransformerDecoder, Embedding, TransformerModel
from fairseq.models.transformer import base_architecture as base_transformer
from fairseq.models.fairseq_encoder import EncoderOut
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
from collections import OrderedDict
import os

logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024

from argparse import Namespace
from omegaconf import DictConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

try:
    from .unilm_models import UniLMDecoder
except:
    from unilm_models import UniLMDecoder

@register_model('DeiT_TR')
@register_model('TrOCR')
class TrOCRModel(FairseqEncoderDecoderModel):

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
            if model_seq_len != ckpt_seq_len and getattr(args, "adapt_encoder_pos_embed", None):
                logger.warning('Load from encoder.deit {:d} seq len to {:d}'.format(ckpt_seq_len, model_seq_len))
                if model_seq_len <= ckpt_seq_len:
                    new_state_dict['encoder.deit.pos_embed'] = new_state_dict['encoder.deit.pos_embed'][:, :model_seq_len, :]
                else:
                    t = self.state_dict()['encoder.deit.pos_embed']
                    t[:, :ckpt_seq_len, :] = new_state_dict['encoder.deit.pos_embed']
                    new_state_dict['encoder.deit.pos_embed'] = t

        # if hasattr(model_cfg, "reset_dictionary") and model_cfg.reset_dictionary:
        #     logger.info('Reset token embed weights and output projection during loading pretrained models')
        #     del new_state_dict['decoder.embed_tokens.weight'] 
        #     del new_state_dict['decoder.output_projection.weight']

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
        parser.add_argument(
            '--only-keep-pretrained-decoder-structure', action='store_true',
            help='if only keep the pretrained decoder structure'
        )
        parser.add_argument(
            '--only-keep-pretrained-encoder-structure', action='store_true',
            help='if only keep the pretrained encoder structure'
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
        encoder = TrOCREncoder(
            args = args,
            dictionary = task.source_dictionary
        )

        args.encoder_embed_dim = encoder.deit.embed_dim

        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        if getattr(args, "decoder_pretrained", None) == None or getattr(args, "decoder_pretrained", None).upper() == 'None':
            logger.info('Decoder is randomly initialized.')            
            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )            
            decoder = TransformerDecoder(
                args = args,
                dictionary=task.target_dictionary,
                embed_tokens=decoder_embed_tokens,
                no_encoder_attn=False
            )

        elif getattr(args, "decoder_pretrained", None).startswith('roberta2'):         
            logger.info('Using the learned pos embedding version loading roberta.')
            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )
            
            pretrained_model = getattr(args, "decoder_pretrained", None)
            specified = pretrained_model.find('-')!=-1

            if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] != '0':
                torch.distributed.barrier()
            if specified:
                pretrained_model = pretrained_model.replace('-', '.')
                logger.info('Load pre-trained decoder parameters from {}'.format(pretrained_model))
                roberta = torch.hub.load('pytorch/fairseq:main', pretrained_model)
            elif args.decoder_layers == 6:
                logger.info('Load pre-trained decoder parameters from roberta.base')
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.base')
            elif args.decoder_layers == 12:
                logger.info('Load pre-trained decoder parameters from roberta.large')                    
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large')
            else:
                raise AttributeError('Cannot determined the pre-trained model')

            if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] == '0':
                torch.distributed.barrier()

            roberta.model.args.encoder_layers = args.decoder_layers
            roberta.model.args.fp16 = args.fp16
            roberta_args = TrOCRModel.read_args_from_roberta(roberta.model.args)
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

            if hasattr(args, 'only_keep_pretrained_decoder_structure') and args.only_keep_pretrained_decoder_structure:
                logger.info('Only keep the pretrained decoder structure.')                
                pass
            else:
                missing_keys, unexpected_keys = decoder.load_state_dict(
                    new_decoder_dict, strict=False
                )

        elif getattr(args, "decoder_pretrained", None) == 'unilm':
            logger.info('Decoder is pretrained using the unilm.')
            
            prefix_of_parameter = 'bert'

            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )

            decoder = UniLMDecoder(
                args,
                task.target_dictionary,
                decoder_embed_tokens,
                no_encoder_attn=False,
            )            

            if hasattr(args, 'decoder_pretrained_url') and args.decoder_pretrained_url != None and args.decoder_pretrained_url != '':                
                unilm_url = args.decoder_pretrained_url
                logger.info('The unilm model url: {}.'.format(unilm_url[:unilm_url.find('?')]))
                unilm_state_dict = torch.hub.load_state_dict_from_url(unilm_url)            

                unilm_layers = OrderedDict([(k, unilm_state_dict[k]) for k in unilm_state_dict.keys() if k.startswith(prefix_of_parameter + '.encoder.layer.')])
                unilm_layers_num = []
                for k in unilm_layers.keys():
                    t = k.replace(prefix_of_parameter + '.encoder.layer.', '')
                    t = t[:t.find('.')]
                    unilm_layers_num.append(int(t))
                unilm_layers_num = max(unilm_layers_num) + 1

                offset = unilm_layers_num - len(decoder.layers)
                assert offset == 0

                decoder_dict = decoder.state_dict()
                # embedding
                new_pos_weight = torch.zeros_like(decoder_dict['embed_positions.weight'])
                # position padding will right offset padding idx + 1
                new_pos_weight[task.target_dictionary.pad() + 1:, :] = unilm_state_dict[prefix_of_parameter + '.embeddings.position_embeddings.weight']
                new_decoder_dict = {
                    'embed_tokens.weight': unilm_state_dict[prefix_of_parameter + '.embeddings.word_embeddings.weight'],
                    'embed_positions.weight': new_pos_weight,
                    'layernorm_embedding.weight': unilm_state_dict[prefix_of_parameter + '.embeddings.LayerNorm.weight'],
                    'layernorm_embedding.bias': unilm_state_dict[prefix_of_parameter + '.embeddings.LayerNorm.bias']
                }            

                # layers
                key_map = {
                    'self_attn.k_proj': 'attention.self.key',
                    'self_attn.v_proj': 'attention.self.value',                
                    'self_attn.q_proj': 'attention.self.query',
                    'self_attn.out_proj': 'attention.output.dense',
                    'self_attn_layer_norm': 'attention.output.LayerNorm',
                    'fc1': 'intermediate.dense',
                    'fc2': 'output.dense',
                    'final_layer_norm': 'output.LayerNorm'
                }
                for layer_id in range(unilm_layers_num):
                    unilm_prefix = prefix_of_parameter + '.encoder.layer.{}.'.format(layer_id)
                    decoder_prefix = 'layers.{}.'.format(layer_id)

                    for key in key_map:
                        for suffix in ['.weight', '.bias']:
                            decoder_key = decoder_prefix + key + suffix
                            unilm_key = unilm_prefix + key_map[key] + suffix
                            if decoder_key in decoder_dict and unilm_key in unilm_state_dict:
                                new_decoder_dict[decoder_key] = unilm_state_dict[unilm_key]
                            
                if hasattr(args, "reset_dictionary") and args.reset_dictionary:
                    logger.info('Reset token embedding weights during decoder initialization.')
                    del new_decoder_dict['embed_tokens.weight']
                elif hasattr(args, "adapt_dictionary") and args.adapt_dictionary:
                    unilm_embed_tokens_weight = new_decoder_dict['embed_tokens.weight']
                    logger.info('Adapt token embedding weights during decoder initialization from {} to {}'.format(unilm_embed_tokens_weight.shape[0], decoder_embed_tokens.weight.shape[0]))                
                    new_decoder_dict['embed_tokens.weight'] = torch.zeros_like(decoder_dict['embed_tokens.weight'])
                    new_decoder_dict['embed_tokens.weight'][:min(unilm_embed_tokens_weight.shape[0], decoder_dict['embed_tokens.weight'].shape[0]), :] = unilm_embed_tokens_weight[:min(unilm_embed_tokens_weight.shape[0], decoder_dict['embed_tokens.weight'].shape[0]), :]

                if hasattr(args, 'only_keep_pretrained_decoder_structure') and args.only_keep_pretrained_decoder_structure:
                    logger.info('Only keep the pretrained decoder structure.')
                    pass
                else:
                    missing_keys, unexpected_keys = decoder.load_state_dict(
                        new_decoder_dict, strict=False
                    )
            else:
                logger.warning('You must specify the unilm model url or the decoder is randomly initialized.')

            # freeze k_proj bias
            for layer in decoder.layers:
                layer.self_attn.k_proj.bias.requires_grad = False        

        elif getattr(args, "decoder_pretrained", None).startswith('roberta'):  
            logger.info('Using the old version loading roberta.')
            decoder_embed_tokens = cls.build_embedding(
                args, task.target_dictionary, args.decoder_embed_dim, args.decoder_embed_path
            )        
            decoder = TransformerDecoder(
                args = args,
                dictionary=task.target_dictionary,
                embed_tokens=decoder_embed_tokens,
                no_encoder_attn=False
            )

            pretrained_model = getattr(args, "decoder_pretrained", None)
            specified = pretrained_model.find('-')!=-1

            if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] != '0':
                torch.distributed.barrier()

            if specified:
                pretrained_model = pretrained_model.replace('-', '.')
                logger.info('Load pre-trained decoder parameters from {}'.format(pretrained_model))
                roberta = torch.hub.load('pytorch/fairseq:main', pretrained_model)
            elif args.decoder_layers == 6:
                logger.info('Load pre-trained decoder parameters from roberta.base')
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.base')
            elif args.decoder_layers == 12:
                logger.info('Load pre-trained decoder parameters from roberta.large')
                roberta = torch.hub.load('pytorch/fairseq:main', 'roberta.large')
            else:
                raise AttributeError('Cannot determined the pre-trained model')

            if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] == '0':
                torch.distributed.barrier()

            if hasattr(args, 'only_keep_pretrained_decoder_structure') and args.only_keep_pretrained_decoder_structure:
                logger.info('Only keep the pretrained decoder structure.')
                pass
            else:
                decoder.embed_tokens.load_state_dict(roberta.model.encoder.sentence_encoder.embed_tokens.state_dict())
                roberta_layers = roberta.model.encoder.sentence_encoder.layers
                decoder_layers = decoder.layers
                offset = len(roberta_layers) - len(decoder_layers)
                assert offset >= 0
                
                for i in range(len(decoder_layers)):
                    roberta_i = i + offset
                    decoder_layers[i].self_attn.load_state_dict(roberta_layers[roberta_i].self_attn.state_dict())
                    decoder_layers[i].self_attn_layer_norm.load_state_dict(roberta_layers[roberta_i].self_attn_layer_norm.state_dict())

        else:
            raise Exception('Undefined decoder pretraining method.')
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
        encoder_out = self.encoder(imgs, **kwargs) # (seq_len, batch, embed_dim)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )   # (batch, seq_len, vocab_size)
        return decoder_out


@register_model_architecture('DeiT_TR', 'deit_base_decoder_base')
def deit_base_decoder_base(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
    # Transformer Decoder
    # args.encoder_embed_dim = 768
    base_transformer(args)

@register_model_architecture('DeiT_TR', 'deit_base_decoder_large')
def deit_base_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_base_distilled_patch16_384")
    # Transformer Decoder
    # args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('TrOCR', 'trocr_base')
@register_model_architecture('DeiT_TR', 'beit_base_decoder_large')
def beit_base_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_base_patch16_384")
    # Transformer Decoder
    # args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)

@register_model_architecture('TrOCR', 'trocr_large')
@register_model_architecture('DeiT_TR', 'beit_large_decoder_large')
def beit_large_decoder_large(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "beit_large_patch16_384")
    # Transformer Decoder
    # args.encoder_embed_dim = 1024
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
    # args.encoder_embed_dim = 768
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    base_transformer(args)


def nlrv4_compressed_tiny(args):
    args.decoder_learned_pos = True
    args.layernorm_embedding = True
    args.decoder_attention_heads = 8
    args.decoder_embed_dim = 256
    args.decoder_output_dim = 256
    args.decoder_ffn_embed_dim = 1024
    args.dropout = 0.1
    args.decoder_layers = 6
    args.max_target_positions = 512

@register_model_architecture('TrOCR', 'trocr_small_224')
def trocr_small(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_small_distilled_patch16_224")

    nlrv4_compressed_tiny(args)
    # Transformer Decoder
    base_transformer(args)    

@register_model_architecture('TrOCR', 'trocr_small')
@register_model_architecture('TrOCR', 'trocr_small_384')
def trocr_small_384(args):
    # DeiT Encoder  deit_base_distilled_patch16_384
    args.deit_arch = getattr(args, "deit_arch", "deit_small_distilled_patch16_384")

    nlrv4_compressed_tiny(args)
    # Transformer Decoder
    base_transformer(args)    

class TrOCREncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        if hasattr(args, 'only_keep_pretrained_encoder_structure') and args.only_keep_pretrained_encoder_structure:
            pretrained = False
        else:
            pretrained = True
        
        if 'custom_size' in args.deit_arch:
            self.deit = create_model(args.deit_arch, pretrained=pretrained, img_size=args.input_size, ape=args.ape, mask_ratio=args.mask_ratio)
        else:
            self.deit = create_model(args.deit_arch, pretrained=pretrained, ape=args.ape, mask_ratio=args.mask_ratio)
        
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
