from dataclasses import dataclass, field
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.models import (
  BaseFairseqModel,
  register_model,
  register_model_architecture,
)
from fairseq.models.roberta import (
    roberta_large_architecture,
    roberta_base_architecture,
    RobertaEncoder,
    RobertaModel,
)
from fairseq.models.transformer_lm import (
  TransformerLanguageModelConfig,
  TransformerLanguageModel,
  base_gpt3_architecture,
)
from kosmos2_5.models.connector import build_connector
from kosmos2_5.models.gpt import GPTmodel, GPTModelConfig

logger = logging.getLogger(__name__)

def slice_tokens_for_mlm(A, indx, num_elem=2):
    all_indx = indx[:,None] + torch.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]


@dataclass
class UniGPTModelConfig(GPTModelConfig):
    pass

@register_model("unigptmodel", dataclass=UniGPTModelConfig)
class UniGPTmodel(BaseFairseqModel):

    def __init__(
            self,
            args,
            gpt_model,
            img_model=None,
            img_connector=None,
            bos=0, eos=2):
        """
        text_model: bidirectional text model, such as roberta, bert, electra
        img_model: image model, such as ViT, CLIP, BEIT
        aud_model: audio model, such as HuBERT, wavLM
        """
        super().__init__()
        self.args = args
        self.gpt_model = gpt_model
        
        self.img_model = img_model
        self.img_connector = img_connector
        self.bos = bos
        self.eos = eos
        self.classification_heads = nn.ModuleDict()
        self.ft_type = args.ft_type

    @classmethod
    def build_model(cls, args, task):
        if hasattr(task, "all_dict"):
            task.dictionary = task.all_dict
        gpt_model = GPTmodel.build_model(args, task)
        logger.info("gpt args is {}".format(args))

        img_model, img_connector = cls.load_image_model(args, task)

        model = cls(args, gpt_model, 
            # text_model=text_model, text_connector=text_connector,
            img_model=img_model, img_connector=img_connector,
            # aud_model=aud_model, aud_connector=aud_connector,
            bos=task.dictionary.bos_index, 
            eos=task.dictionary.eos_index)

        return model

    def forward(self, src_tokens, 
            mlm_src_tokens=None, gpt_input_mask=None, 
            img_src_tokens=None, img_gpt_input_mask=None, 
            aud_src_tokens=None, aud_gpt_input_mask=None,
            gpt_loss_mask=None, mlm_mask=None, classification_head_name=None, **kwargs):

        if classification_head_name is None:
            if mlm_src_tokens is not None:
                # mlm
                mlm_output, _ = self.text_model(mlm_src_tokens, features_only=True)
                mlm_output = mlm_output[mlm_mask]
                if self.text_connector is not None:
                    # linear projection layer
                    mlm_output = self.text_connector(mlm_output)
            else:
                mlm_output = None
            
            if img_src_tokens is not None:
                img_output = self.get_image_representation(img_src_tokens)
            else:
                img_output = None
            
            if aud_src_tokens is not None:
                aud_output = self.get_audio_representation(aud_src_tokens, kwargs['aud_mask'])
            else:
                aud_output = None

            # gpt 
            x, extra = self.gpt_model(src_tokens, 
                mlm_features=mlm_output, gpt_input_mask=gpt_input_mask,
                img_features=img_output, img_gpt_input_mask=img_gpt_input_mask,
                aud_features=aud_output, aud_gpt_input_mask=aud_gpt_input_mask,
                **kwargs)

            # loss mask
            extra["loss_mask"] = gpt_loss_mask
            return x, extra


    def get_image_representation(self, img_src_tokens, image_attention_masks):
        # image
        img_output = self.img_model(img_src_tokens, image_attention_masks)
        img_output = F.normalize(img_output[0], dim=-1)
        src_len = img_output.size(1)
        img_output = img_output.reshape(-1, img_output.size(-1))
        if self.img_connector is not None:
            img_output = self.img_connector(img_output, src_len=src_len)
        return img_output

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    @property
    def supported_targets(self):
        return {"future"}

    @classmethod
    def load_image_model(cls, args, task):
        from transformers import Pix2StructVisionModel
        model = Pix2StructVisionModel.from_pretrained(args.image_encoder)
        connector = build_connector(args, model.config.hidden_size, args.decoder_embed_dim)

        return model, connector


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
        ft_type
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)
        self.ft_type = ft_type

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

@register_model_architecture("unigptmodel", "unigptmodel_large")
def gptmodel_large(args):
    # 1.3B params
    args.decoder_layers = getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)

    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1536 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    base_gpt3_architecture(args)
    roberta_large_architecture(args)
