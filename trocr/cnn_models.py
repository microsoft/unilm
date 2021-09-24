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
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from functools import partial
import logging

from typing import Dict, Optional, Tuple


logger = logging.getLogger(__name__)

DEFAULT_MAX_TARGET_POSITIONS = 1024

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=1024):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


@register_model('ResNet_TR')
class ResNetTRModel(FairseqEncoderDecoderModel):
    
    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        # parser.add_argument(
        #     '--resnet-arch', type=str,
        #     help='the arch name for the DeiT encoder'
        # )

    @classmethod
    def build_model(cls, args, task):
        encoder = ResNetEncoder(
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

        decoder = TransformerDecoder(
            args = args,
            dictionary=task.target_dictionary,
            embed_tokens=decoder_embed_tokens,
            no_encoder_attn=False
        )

        if getattr(args, "decoder_pretrained", None).startswith('roberta'):          
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

            # decoder.
        elif getattr(args, "decoder_pretrained", None) == 'unilm':
            # unilm_base_cased
            unilm_url = 'https://msranlpintern.blob.core.windows.net/minghaoli/model/pretrained/NLRv4-Base/tnlrv4_base_cased_model.pt?sv=2020-04-08&st=2021-05-13T09%3A03%3A07Z&se=2022-05-14T09%3A03%3A00Z&sr=b&sp=r&sig=IVfrjzk7%2Ba8IXAMLrollMdrPJkXA17S2Kge%2FSj%2FsP%2B8%3D'
            unilm_state_dict = torch.hub.load_state_dict_from_url(unilm_url)
            decoder.embed_tokens.load_state_dict({'weight':unilm_state_dict['bert.embeddings.word_embeddings.weight'].float()})

            decoder_layers = decoder.layers
            unilm_layers_num = 12
            offset = unilm_layers_num - len(decoder_layers)
            assert offset == 6
            for i in range(len(decoder_layers)):
                decoder_layers[i].self_attn = UniLMMultiheadAttention(
                    args.decoder_embed_dim,
                    args.decoder_attention_heads,
                    dropout=args.attention_dropout,
                    add_bias_kv=False,
                    add_zero_attn=False,
                    self_attention=not getattr(args, "cross_self_attention", False),
                    q_noise=decoder_layers[i].quant_noise,
                    qn_block_size=decoder_layers[i].quant_noise_block_size,
                )

                unilm_i = i + offset
                decoder_layers[i].self_attn.k_proj.load_state_dict({
                    'weight': unilm_state_dict['bert.encoder.layer.{:d}.attention.self.key.weight'.format(unilm_i)].float()}, strict=False)
                decoder_layers[i].self_attn.q_proj.load_state_dict({
                    'weight':unilm_state_dict['bert.encoder.layer.{:d}.attention.self.query.weight'.format(unilm_i)].float(),
                    'bias':unilm_state_dict['bert.encoder.layer.{:d}.attention.self.query.bias'.format(unilm_i)].float()})
                decoder_layers[i].self_attn.v_proj.load_state_dict({
                    'weight':unilm_state_dict['bert.encoder.layer.{:d}.attention.self.value.weight'.format(unilm_i)].float(),
                    'bias':unilm_state_dict['bert.encoder.layer.{:d}.attention.self.value.bias'.format(unilm_i)].float()})

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


@register_model_architecture('ResNet_TR', 'ResNet50_Decoder_Base')
def ResNet50_Decoder_Base(args):
    # Transformer Decoder
    args.encoder_embed_dim = 2048
    args.decoder_embed_dim = 768
    base_transformer(args)

@register_model_architecture('ResNet_TR', 'ResNet50_Decoder_Large')
def ResNet50_Decoder_Large(args):
    # Transformer Decoder
    args.encoder_embed_dim = 2048

    args.decoder_layers = 12
    args.decoder_embed_dim = 1024
    args.decoder_ffn_embed_dim = 4096
    args.decoder_attention_heads = 16
    base_transformer(args)


class ResNetEncoder(FairseqEncoder):
    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.resnet = IntermediateLayerGetter(self.backbone, return_layers={'layer4': "0"})
        self.pos_embed = PositionEmbeddingLearned()

    def forward(self, imgs):
        x = self.resnet(imgs)['0']   # bs, dim, 7, 7
        pos = self.pos_embed(x)
        x = x + pos
        x = x.flatten(2)  #bs, dim, 49
        x = x.permute(2, 0, 1)  # 49, bs, dim  
        
        # x, encoder_embedding = self.deit.forward_features(imgs)  # bs, n + 2, dim
        # x = x.transpose(0, 1) # n + 2, bs, dim

        encoder_padding_mask = torch.zeros(*x.shape[:2]).transpose(0, 1).to(imgs.device)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            # "encoder_embedding": [encoder_embedding],  # B x T x C
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
        #   _encoder_embedding = encoder_out['encoder_embedding'][0]
          return {
              "encoder_out": [_encoder_out.index_select(1, new_order)],
                "encoder_padding_mask": [_encoder_padding_mask.index_select(0, new_order)],  # B x T
                # "encoder_embedding": [_encoder_padding_mask.index_select(0, new_order)],  # B x T x C
                "encoder_states": [], 
                "src_tokens": [],
                "src_lengths": [],
        }

    

if __name__ == '__main__':
    from data import SROIETextRecognitionDataset
    from data_aug import build_data_aug
    from fairseq.data import Dictionary
    tfm = build_data_aug((224, 224), 'valid')
    target_dict = Dictionary.load('data/SROIE_Task2_Original/gpt2.dict.txt')
    dataset = SROIETextRecognitionDataset('data/SROIE_Task2_Original/valid', tfm, None, target_dict)
    encoder = ResNetEncoder(None, None)


    for item in dataset:
        x = encoder(item['tfm_img'].unsqueeze(0))