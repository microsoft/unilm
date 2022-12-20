# --------------------------------------------------------
# BEATs: Audio Pre-Training with Acoustic Tokenizers (https://arxiv.org/abs/2212.09058)
# Github source: https://github.com/microsoft/unilm/tree/master/beats
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------


import torch
import torch.nn as nn
from torch.nn import LayerNorm
import torchaudio.compliance.kaldi as ta_kaldi

from backbone import (
    TransformerEncoder,
)
from quantizer import (
    NormEMAVectorQuantizer,
)

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class TokenizersConfig:
    def __init__(self, cfg=None):
        self.input_patch_size: int = -1  # path size of patch embedding
        self.embed_dim: int = 512  # patch embedding dimension
        self.conv_bias: bool = False  # include bias in conv encoder

        self.encoder_layers: int = 12  # num encoder layers in the transformer
        self.encoder_embed_dim: int = 768  # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072  # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12  # num encoder attention heads
        self.activation_fn: str = "gelu"  # activation function to use

        self.layer_norm_first: bool = False  # apply layernorm first in the transformer
        self.deep_norm: bool = False  # apply deep_norm first in the transformer

        # dropouts
        self.dropout: float = 0.1  # dropout probability for the transformer
        self.attention_dropout: float = 0.1  # dropout probability for attention weights
        self.activation_dropout: float = 0.0  # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0  # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0  # dropout to apply to the input (after feat extr)

        # positional embeddings
        self.conv_pos: int = 128  # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16  # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False  # apply relative position embedding
        self.num_buckets: int = 320  # number of buckets for relative position embedding
        self.max_distance: int = 1280  # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False  # apply gated relative position embedding

        # quantizer
        self.quant_n: int = 1024 # codebook number in quantizer
        self.quant_dim: int = 256    # codebook dimension in quantizer

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class Tokenizers(nn.Module):
    def __init__(
            self,
            cfg: TokenizersConfig,
    ) -> None:
        super().__init__()
        logger.info(f"Tokenizers Config: {cfg.__dict__}")

        self.cfg = cfg

        self.embed = cfg.embed_dim
        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.input_patch_size = cfg.input_patch_size
        self.patch_embedding = nn.Conv2d(1, self.embed, kernel_size=self.input_patch_size, stride=self.input_patch_size,
                                         bias=cfg.conv_bias)

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        assert not cfg.deep_norm or not cfg.layer_norm_first
        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=cfg.quant_n, embedding_dim=cfg.quant_dim, beta=1.0, kmeans_init=True, decay=0.99,
        )
        self.quant_n = cfg.quant_n
        self.quantize_layer = nn.Sequential(
            nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim),
            nn.Tanh(),
            nn.Linear(cfg.encoder_embed_dim, cfg.quant_dim)  # for quantize
        )

    def forward_padding_mask(
            self,
            features: torch.Tensor,
            padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def preprocess(
            self,
            source: torch.Tensor,
            fbank_mean: float = 15.41663,
            fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        fbanks = []
        for waveform in source:
            waveform = waveform.unsqueeze(0) * 2 ** 15
            fbank = ta_kaldi.fbank(waveform, num_mel_bins=128, sample_frequency=16000, frame_length=25, frame_shift=10)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        return fbank

    def extract_labels(
            self,
            source: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            fbank_mean: float = 15.41663,
            fbank_std: float = 6.55582,
    ):
        fbank = self.preprocess(source, fbank_mean=fbank_mean, fbank_std=fbank_std)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(fbank, padding_mask)

        fbank = fbank.unsqueeze(1)
        features = self.patch_embedding(fbank)
        features = features.reshape(features.shape[0], features.shape[1], -1)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        x = self.dropout_input(features)

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
        )

        quantize_input = self.quantize_layer(x)
        quantize_feature, embed_loss, embed_ind = self.quantize(quantize_input)

        return embed_ind

