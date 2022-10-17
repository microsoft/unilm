# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import torch.nn as nn
import torch
import contextlib

from fairseq import utils
from fairseq.modules import (
    AdaptiveSoftmax,
)

class TextDecoderPostnet(nn.Module):
    """

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(self, embed_tokens, dictionary, args, output_projection=None,):
        super(TextDecoderPostnet, self).__init__()
        self.output_embed_dim = args.decoder_output_dim
        self.output_projection = output_projection
        self.adaptive_softmax = None
        self.share_input_output_embed = args.share_input_output_embed
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)
        self.freeze_decoder_updates = args.freeze_decoder_updates
        self.num_updates = 0

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        # num_base_layers = getattr(args, "base_layers", 0)
        # for i in range(num_base_layers):
        #     self.layers.insert(
        #         ((i + 1) * args.decoder_layers) // (num_base_layers + 1),
        #         BaseLayer(args),
        #     )

    def forward(self, x):
        ft = self.freeze_decoder_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            return self._forward(x)

    def _forward(self, x):
        # embed positions
        x = self.output_layer(x)

        return x

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.num_updates = num_updates
