from dataclasses import dataclass, field
from typing import Optional
from fairseq.dataclass import ChoiceEnum, FairseqDataclass

import logging
import numpy as np
import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.utils import safe_getattr, safe_hasattr


from fairseq.modules import LayerNorm
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
from unilm.models.connector import build_connector
from unilm.models.gpt import GPTmodel, GPTModelConfig
from unilm.models.gpt_eval import GPTEvalmodel

from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3

import pdb

logger = logging.getLogger(__name__)

def slice_tokens_for_mlm(A, indx, num_elem=2):
    all_indx = indx[:,None] + torch.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

@dataclass
class UniGPTModelConfig(GPTModelConfig):
    text_encoder: str = field(
        default="none",
        metadata={
            "help": "enable text encoder, options: none, roberta, electra"
        },
    )
    image_encoder: str = field(
        default="none",
        metadata={
            "help": "enable image encoder, options: none, clip, beit"
        },
    )
    audio_encoder: str = field(
        default="none",
        metadata={
            "help": "enable audio encoder, options: none, "
        },
    )

    # parameters for MLM
    connector: str = field(
        default='complex',
        metadata={
            "help": "connector: none, complex, simple, xconnector"
        },
    )
    latent_query_num: int = field(
        default=64,
        metadata={
            "help": "number of latent query tokens"
        },
    )
    remain_tokens: int = field(
        default=300, 
        metadata={
            "help": "at least k tokens to produce gpt loss"
        },
    )
    mlm_model_path: str = field(
        default="",
        metadata={"help": "mlm checkpoint path"},
    )
    mlm_dict: str = field(
        default="",
        metadata={"help": "mlm dict path"},
    )
    mlm_tokens_per_sample: int = field(
        default=512,
        metadata={"help": "mlm max length"},
    )

    freeze_gpt: bool = field(
        default=False,
        metadata={
            "help": "freeze gpt parameters"
        },
    )

    # parameters for visual
    visual_model_name: str = field(
        default="ViT-B-16",
        metadata={"help": "model_name for open_clip"},)
    visual_pretrained: str = field(
        default="laion400m_e32",
        metadata={"help": "model_name for visual_pretrained"},)
    visual_output_dim: int = field(
        default=768,
        metadata={"help": "output dimension for visual_pretrained"},)
    visual_output_dim_str: str = field(
        default='768',
        metadata={"help": "output dimension for visual_pretrained"},)
    no_freeze_layer: str = field(
        default='',
        metadata={
            "help": "freeze last layer of visual_pretrained"
        },)
    
    # parameters for speech
    speech_model_path: str = field(
        default="",
        metadata={"help": "speech checkpoint path"},
    )
    audio_output_dim: int = field(
        default=768,
        metadata={"help": "output dimension for audio_pretrained"},)

    # parameters for fine-tuning
    ft_type: int = field(
        default=3,
        metadata={
            "help": "fine-tuning type: \
            1: gpt only \
            2: roberta only \
            3: roberta + gpt \
            4: roberta + gpt(freeze) \
            5: roberta(freeze) + gpt "
        },
    )
    pooler_dropout: float = field(
        default=0.1,
        metadata={"help": "mlm max length"},
    )

    pretrained_ckpt_path: str = field(
        default="",
        metadata={"help": "model checkpoint path"},
    )
    

@register_model("unigptmodel", dataclass=UniGPTModelConfig)
class UniGPTmodel(BaseFairseqModel):

    def __init__(self, args, gpt_model, 
            text_model=None, img_model=None, aud_model=None, 
            text_connector=None, img_connector=None, aud_connector=None,
            bos=0, eos=2):
        """
        text_model: bidirectional text model, such as roberta, bert, electra
        img_model: image model, such as ViT, CLIP, BEIT
        aud_model: audio model, such as HuBERT, wavLM
        """
        super().__init__()
        self.args = args
        self.gpt_model = gpt_model
        
        self.text_model = text_model
        self.text_connector = text_connector
        self.img_model = img_model
        self.img_connector = img_connector
        self.aud_model = aud_model
        self.aud_connector = aud_connector

        self.bos = bos
        self.eos = eos
        self.classification_heads = nn.ModuleDict()
        self.ft_type = args.ft_type

        if args.freeze_gpt:
            for p in self.gpt_model.parameters():
                p.requires_grad = False

    @classmethod
    def build_model(cls, args, task):
        if hasattr(task, "all_dict"):
            task.dictionary = task.all_dict
        if task.__class__.__name__ == 'GenerationObjTask':
            gpt_model = GPTEvalmodel.build_model(args, task)
        else:
            gpt_model = GPTmodel.build_model(args, task)
        logger.info("gpt args is".format(args))

        text_model, text_connector = cls.load_text_model(args, task)
        img_model, img_connector = cls.load_image_model(args, task)
        aud_model, aud_connector = cls.load_audio_model(args, task)

        model = cls(args, gpt_model, 
            text_model=text_model, text_connector=text_connector,
            img_model=img_model, img_connector=img_connector,
            aud_model=aud_model, aud_connector=aud_connector,
            bos=task.dictionary.bos_index, 
            eos=task.dictionary.eos_index)

        if args.pretrained_ckpt_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.pretrained_ckpt_path)
            model.load_state_dict(state["model"], strict=True, args=args)

        # freeze text model
        if model.text_model is not None:
            for p in model.text_model.parameters():
                p.requires_grad = False
        # freeze image model
        if model.img_model is not None:
            for p_name, p in model.img_model.named_parameters():
                if args.no_freeze_layer:
                    no_freeze_layers = args.no_freeze_layer.split(',')    
                    for no_freeze_layer in no_freeze_layers:
                        if no_freeze_layer in p_name:
                            print("no_freeze_layer: {}".format(p_name))
                            p.requires_grad = True
                            break
                p.requires_grad = False
        # freeze audio model
        if model.aud_model is not None:
            for p in model.aud_model.parameters():
                p.requires_grad = False

        return model

    def forward(self, src_tokens, 
            mlm_src_tokens=None, gpt_input_mask=None, 
            img_src_tokens=None, img_gpt_input_mask=None, 
            aud_src_tokens=None, aud_gpt_input_mask=None,
            gpt_loss_mask=None, mlm_mask=None, classification_head_name=None, **kwargs):

        if classification_head_name is None:
            # pre-training

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

            # pdb.set_trace()
            # gpt 
            x, extra = self.gpt_model(src_tokens, 
                mlm_features=mlm_output, gpt_input_mask=gpt_input_mask,
                img_features=img_output, img_gpt_input_mask=img_gpt_input_mask,
                aud_features=aud_output, aud_gpt_input_mask=aud_gpt_input_mask,
                **kwargs)

            # loss mask
            extra["loss_mask"] = gpt_loss_mask
            return x, extra

        # fine-tuning
        raise NotImplementedError

    def get_image_representation(self, img_src_tokens):
        # image
        img_output = self.img_model(img_src_tokens)
        src_len = img_output.size(0)
        img_output = img_output.transpose(0, 1) # T x B x C -> B x T x C
        img_output = img_output.reshape(-1, img_output.size(-1))
        
        if self.img_connector is not None:
            img_output = self.img_connector(img_output, src_len=src_len)
        return img_output

    def get_audio_representation(self, aud_src_tokens, aud_mask):
        raise NotImplementedError

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        self.classification_heads[name] = ClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
            self.args.ft_type
        )

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
        
    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            [] if not hasattr(self, 'classification_heads')
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

    @classmethod
    def load_text_model(cls, args, task):
        """Load a roberta model from the fairseq library."""
        if args.text_encoder == "none":
            return None, None
        mlm_args = copy.deepcopy(args)
        mlm_task = task
        logger.info("Roberta dictionary: {} types".format(len(mlm_task.dictionary)))

        mlm_args.layernorm_embedding = True
        mlm_args.no_scale_embedding = True
        mlm_args.dropout = 0.1
        mlm_args.attention_dropout = 0.1
        mlm_args.tokens_per_sample = mlm_args.mlm_tokens_per_sample
        mlm_model = RobertaModel.build_model(mlm_args, mlm_task)
        logger.info("mlm args is {}".format(mlm_args))
        if args.mlm_model_path != "":
            state = checkpoint_utils.load_checkpoint_to_cpu(args.mlm_model_path)
            mlm_model.load_state_dict(state["model"], strict=True, args=mlm_args)
        connector = build_connector(args, args.encoder_embed_dim, args.decoder_embed_dim)
        return mlm_model, connector

    @classmethod
    def load_image_model(cls, args, task):

        def build_backbone_clip(args, visual_model_name, visual_pretrained):
            from unilm.models.vl.clip import create_model
            force_quick_gelu = False
            if 'ViT-L' in visual_model_name:
                force_quick_gelu = True
            model = create_model(visual_model_name, 
                                 pretrained=visual_pretrained, 
                                 force_quick_gelu=force_quick_gelu)
            return model
        
        if args.image_encoder == "none":
            return None, None
        
        if args.image_encoder == "clip":
            model = build_backbone_clip(args, args.visual_model_name, args.visual_pretrained)
            connector = build_connector(args, args.visual_output_dim, args.decoder_embed_dim)
            return model, connector
        
        raise NotImplementedError("Unknown model name {}".format(args.image_encoder))

    @classmethod
    def load_audio_model(cls, args, task):
        return None, None

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


@register_model_architecture("unigptmodel", "unigptmodel_small")
def gptmodel_small(args):
    # 125M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 12)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 768)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 12)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    base_gpt3_architecture(args)
    roberta_base_architecture(args)

@register_model_architecture("unigptmodel", "unigptmodel_medium")
def gptmodel_medium(args):
    # 355M params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1024)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.1)
    base_gpt3_architecture(args)
    roberta_base_architecture(args)

@register_model_architecture("unigptmodel", "unigptmodel_large")
def gptmodel_large(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 1536)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 16)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    base_gpt3_architecture(args)
    roberta_large_architecture(args)

@register_model_architecture("unigptmodel", "unigptmodel_xl")
def gptmodel_xl(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 24)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    base_gpt3_architecture(args)
    roberta_large_architecture(args)

@register_model_architecture("unigptmodel", "unigptmodel_2b")
def gptmodel_2B(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 36)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 2048)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 32)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    base_gpt3_architecture(args)
    roberta_large_architecture(args)

@register_model_architecture("unigptmodel", "unigptmodel_6b")
def gptmodel_6B(args):
    # 1.3B params
    args.decoder_layers = safe_getattr(args, "decoder_layers", 40)
    args.decoder_embed_dim = safe_getattr(args, "decoder_embed_dim", 3584)
    args.decoder_attention_heads = safe_getattr(args, "decoder_attention_heads", 28)
    args.decoder_learned_pos = safe_getattr(args, "decoder_learned_pos", False)

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.1)
    base_gpt3_architecture(args)
    roberta_large_architecture(args)
