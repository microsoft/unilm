# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn

from torchscale.architecture.decoder import Decoder
from torchscale.architecture.encoder import Encoder


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        args,
        encoder_embed_tokens=None,
        encoder_embed_positions=None,
        decoder_embed_tokens=None,
        decoder_embed_positions=None,
        output_projection=None,
        **kwargs
    ):
        super().__init__()
        self.args = args
        if args.share_all_embeddings:
            args.share_decoder_input_output_embed = True

        self.encoder = Encoder(
            args,
            encoder_embed_tokens,
            encoder_embed_positions,
            is_encoder_decoder=True,
            **kwargs
        )

        if args.share_all_embeddings and decoder_embed_tokens is None:
            decoder_embed_tokens = self.encoder.embed_tokens

        self.decoder = Decoder(
            args,
            decoder_embed_tokens,
            decoder_embed_positions,
            output_projection,
            is_encoder_decoder=True,
            **kwargs
        )

    def forward(
        self,
        src_tokens,
        prev_output_tokens,
        return_all_hiddens=False,
        features_only=False,
        **kwargs
    ):
        encoder_out = self.encoder(src_tokens, return_all_hiddens=return_all_hiddens)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out
