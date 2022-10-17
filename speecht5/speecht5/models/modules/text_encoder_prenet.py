# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import torch.nn as nn

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding


class TextEncoderPrenet(nn.Module):
    """

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        embed_tokens,
        args,
    ):
        super(TextEncoderPrenet, self).__init__()
        self.padding_idx = embed_tokens.padding_idx
        # define encoder prenet
        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if args.enc_use_scaled_pos_enc else PositionalEncoding
        )

        self.encoder_prenet = nn.Sequential(
            embed_tokens,
            pos_enc_class(args.encoder_embed_dim, args.transformer_enc_positional_dropout_rate, max_len=args.max_text_positions),
        )

    def forward(self, src_tokens):
        return self.encoder_prenet(src_tokens), src_tokens.eq(self.padding_idx)
