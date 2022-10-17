# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import contextlib
import torch
import torch.nn as nn

import torch.nn.functional as F
from espnet.nets.pytorch_backend.tacotron2.decoder import Prenet as TacotronDecoderPrenet
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class SpeechDecoderPrenet(nn.Module):
    """

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        odim,
        args,
    ):
        super(SpeechDecoderPrenet, self).__init__()
        # define decoder prenet
        if args.dprenet_layers != 0:
            # decoder prenet
            decoder_input_layer = torch.nn.Sequential(
                TacotronDecoderPrenet(
                    idim=odim,
                    n_layers=args.dprenet_layers,
                    n_units=args.dprenet_units,
                    dropout_rate=args.dprenet_dropout_rate,
                ),
                torch.nn.Linear(args.dprenet_units, args.decoder_embed_dim),
            )
        else:
            decoder_input_layer = "linear"

        pos_enc_class = (
            ScaledPositionalEncoding if args.dec_use_scaled_pos_enc else PositionalEncoding
        )

        if decoder_input_layer == "linear":
            self.decoder_prenet = torch.nn.Sequential(
                torch.nn.Linear(odim, args.decoder_embed_dim),
                torch.nn.LayerNorm(args.decoder_embed_dim),
                torch.nn.Dropout(args.transformer_dec_dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(args.decoder_embed_dim, args.transformer_dec_positional_dropout_rate),
            )
        elif isinstance(decoder_input_layer, torch.nn.Module):
            self.decoder_prenet = torch.nn.Sequential(
                decoder_input_layer, pos_enc_class(args.decoder_embed_dim, args.transformer_dec_positional_dropout_rate, max_len=args.max_speech_positions)
            )

        if args.spk_embed_integration_type == 'pre':
            self.spkembs_layer = torch.nn.Sequential(
                torch.nn.Linear(args.spk_embed_dim + args.decoder_embed_dim, args.decoder_embed_dim), torch.nn.ReLU()
            )
        self.num_updates = 0
        self.freeze_decoder_updates = args.freeze_decoder_updates

    def forward(self, prev_output_tokens, tgt_lengths_in=None, spkembs=None):
        ft = self.freeze_decoder_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            prev_output_tokens = self.decoder_prenet(prev_output_tokens)

            if spkembs is not None:
                spkembs = F.normalize(spkembs).unsqueeze(1).expand(-1, prev_output_tokens.size(1), -1)
                prev_output_tokens = self.spkembs_layer(torch.cat([prev_output_tokens, spkembs], dim=-1))

            if tgt_lengths_in is not None:
                tgt_frames_mask = ~(self._source_mask(tgt_lengths_in).squeeze(1))
            else:
                tgt_frames_mask = None
            return prev_output_tokens, tgt_frames_mask

    def _source_mask(self, ilens):
        """Make masks for self-attention.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor for self-attention.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                    [[1, 1, 1, 0, 0]]], dtype=torch.uint8)
        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.num_updates = num_updates
