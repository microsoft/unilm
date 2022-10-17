# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import contextlib
import torch
import torch.nn as nn
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any
from fairseq import checkpoint_utils, tasks, utils
from fairseq.models import FairseqEncoderDecoderModel, register_model
from fairseq.models.fairseq_decoder import FairseqDecoder
from fairseq.models.fairseq_encoder import FairseqEncoder
from fairseq.tasks import FairseqTask
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data.data_utils import lengths_to_padding_mask

from fairseq.models.hubert import HubertAsrConfig
from speechlm.modules.transformer_decoder import TransformerDecoderScriptable

@dataclass
class SpeechLMS2TConfig(HubertAsrConfig):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )
    use_rel_pos_enc: bool = field(
        default=True,
        metadata={"help": "whether to use relative positional encoding for decoder"},
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension, used for enc-dec att"}
    )
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_output_dim: int = field(
        default=768, metadata={"help": "decoder output dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=12, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each decoder block"}
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside the decoder"
        },
    )
    share_decoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    ### the following config is only for the compatibility to fairseq speech_to_text task
    input_feat_per_channel: Any = None
    input_channels: Any = None
    speaker_to_id: Any = None

@register_model("speechlm_st_legacy", dataclass=SpeechLMS2TConfig)
class SpeechLMS2T(FairseqEncoderDecoderModel):
    def __init__(self, cfg: SpeechLMS2TConfig, encoder: FairseqEncoder, decoder: FairseqDecoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict
    
    @classmethod
    def build_model(cls, cfg: SpeechLMS2TConfig, task: FairseqTask):
        """Build a new model instance."""
        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder = SpeechLMEncoder(cfg, task)
        assert cfg.encoder_embed_dim == encoder.w2v_model.encoder.embedding_dim
        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
        decoder = TransformerDecoderScriptable(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)


class SpeechLMEncoder(FairseqEncoder):
    """
    Modified from fairseq.models.hubert.hubert_asr.HubertEncoder
    1. make it compatible with fairseq speech_to_text task
    2. make it compatible with encoder-decoder model
    """
    def __init__(self, cfg: HubertAsrConfig, task):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert task.data_cfg.standardize_audio() == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        pretrain_task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            pretrain_task.load_state_dict(state["task_state"])
        else:
            pretrain_task.load_state_dict(task.state_dict())

        model = pretrain_task.build_model(w2v_args.model, from_checkpoint=True)
        if state is not None and not cfg.no_pretrained_weights:
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(pretrain_task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, src_tokens=None, src_lengths=None, **kwargs):

        w2v_args = {
            "source": src_tokens,
            "padding_mask": lengths_to_padding_mask(src_lengths),
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)
            # B x T x C -> T x B x C
            x = x.transpose(0, 1)

        x = self.final_dropout(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [padding_mask],  # B x T
            "padding_mask": [padding_mask],
        }
    
    def forward_torchscript(self, net_input):
        """A TorchScript-compatible version of forward.

        Encoders which use additional arguments may want to override
        this method for TorchScript compatibility.
        """
        _net_input = {
            "source": net_input["src_tokens"],
            "padding_mask": lengths_to_padding_mask(net_input["src_lengths"]),
            "mask": False,
        }

        x, padding_mask = self.w2v_model.extract_features(**_net_input)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_out = {
            "encoder_out" : [x],
            "encoder_padding_mask" : [padding_mask],
        }
        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = [
                x.index_select(1, new_order) for x in encoder_out["encoder_out"]
            ]
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [
                x.index_select(0, new_order) for x in encoder_out["encoder_padding_mask"]
            ]
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
