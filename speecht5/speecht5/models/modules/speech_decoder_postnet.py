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

from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet


class SpeechDecoderPostnet(nn.Module):
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
        super(SpeechDecoderPostnet, self).__init__()
        # define decoder postnet
        # define final projection
        self.feat_out = torch.nn.Linear(args.decoder_embed_dim, odim * args.reduction_factor)
        self.prob_out = torch.nn.Linear(args.decoder_embed_dim, args.reduction_factor)

        # define postnet
        self.postnet = (
            None
            if args.postnet_layers == 0
            else Postnet(
                idim=0,
                odim=odim,
                n_layers=args.postnet_layers,
                n_chans=args.postnet_chans,
                n_filts=args.postnet_filts,
                use_batch_norm=args.use_batch_norm,
                dropout_rate=args.postnet_dropout_rate,
            )
        )

        self.odim = odim
        self.num_updates = 0
        self.freeze_decoder_updates = args.freeze_decoder_updates

    def forward(self, zs):
        ft = self.freeze_decoder_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            # (B, Lmax//r, odim * r) -> (B, Lmax//r * r, odim)
            before_outs = self.feat_out(zs).view(zs.size(0), -1, self.odim)
            # (B, Lmax//r, r) -> (B, Lmax//r * r)
            logits = self.prob_out(zs).view(zs.size(0), -1)
            # postnet -> (B, Lmax//r * r, odim)
            if self.postnet is None:
                after_outs = before_outs
            else:
                after_outs = before_outs + self.postnet(
                    before_outs.transpose(1, 2)
                ).transpose(1, 2)

        return before_outs, after_outs, logits
    
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.num_updates = num_updates
