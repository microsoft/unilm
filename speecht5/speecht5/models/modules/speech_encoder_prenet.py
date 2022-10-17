# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import math
import torch
import contextlib
from typing import List, Tuple
import torch.nn as nn

from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.data.data_utils import compute_mask_indices
from fairseq.modules import (
    PositionalEmbedding,
    Fp32GroupNorm,
    FairseqDropout,
    SamePad,
    GradMultiply,
    LayerNorm,
    Fp32LayerNorm,
    TransposeLast,
)
import numpy as np

logger = logging.getLogger(__name__)


class LinearLayer(nn.Module):
    def __init__(self, idim, odom, dropout=0):
        super(LinearLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(idim, odom),
            nn.LayerNorm(odom),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        return out
    
    def forward(self, src_tokens, src_lengths):
        """
        src_tokens: [B, T, C]
        src_lengths: [B]
        """
        x = self.linear(src_tokens)
        x = x.transpose(0, 1).contiguous() # -> T x B x C
        return x, src_lengths


class SpeechEncoderPrenet(nn.Module):
    """

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(self, args):
        super(SpeechEncoderPrenet, self).__init__()
        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1
        self.freeze_encoder_updates = args.freeze_encoder_updates
        self.num_updates = 0
        assert args.encoder_speech_prenet in ["conv", "linear"], args.encoder_speech_prenet
        feature_enc_layers = eval(args.conv_feature_layers)  # noqa
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )
        feature_ds_rate = np.prod([s for _, _, s in feature_enc_layers])
        self.feat2tar_ratio = (
            args.label_rates * feature_ds_rate / args.sample_rate
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim
            else None
        )

        self.use_conv_pos = args.use_conv_pos
        self.use_sinc_pos = args.use_sinc_pos
        self.use_abs_pos = getattr(args, "use_abs_pos", False)

        self.feature_grad_mult = args.feature_grad_mult
        if self.use_conv_pos:
            self.layer_norm = LayerNorm(self.embed)
            self.pos_conv = nn.Conv1d(
                args.encoder_embed_dim,
                args.encoder_embed_dim,
                kernel_size=args.conv_pos,
                padding=args.conv_pos // 2,
                groups=args.conv_pos_groups,
            )
            dropout = 0
            std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * args.encoder_embed_dim))
            nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
            nn.init.constant_(self.pos_conv.bias, 0)
            self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
            self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        assert not (self.use_sinc_pos and self.use_abs_pos), f"sinc pos: {self.use_sinc_pos} abs pos: {self.use_abs_pos}"
        if self.use_sinc_pos:
            self.embed_positions = PositionalEmbedding(
                args.max_speech_positions, args.encoder_embed_dim, self.padding_idx
            )
        if self.use_abs_pos:
            self.embed_positions = PositionalEmbedding(
                args.max_speech_positions, args.encoder_embed_dim, self.padding_idx, learned=True
            )
        
        # Hubert
        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.hubert_mask_length = args.hubert_mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.encoder_embed_dim).uniform_()
        )

    def forward(self, src_tokens, require_feat_pen=False, target_list=None, padding_mask=None, mask=True):
        ft = self.freeze_encoder_updates <= self.num_updates
        with torch.no_grad() if not ft else contextlib.ExitStack():
            return self._forward(src_tokens, require_feat_pen, target_list, padding_mask, mask)

    def _forward(self, src_tokens, require_feat_pen=False, target_list=None, padding_mask=None, mask=True):
        if self.feature_grad_mult > 0:
            x = self.feature_extractor(src_tokens)
            x = x.transpose(1, 2).transpose(0, 1)  # [length, batch, hidden_size]
            if self.feature_grad_mult != 1.0:
                x = GradMultiply.apply(x, self.feature_grad_mult)
        else:
            with torch.no_grad():
                x = self.feature_extractor(src_tokens)
                x = x.transpose(1, 2).transpose(0, 1)  # [length, batch, hidden_size]
        x = x.transpose(0, 1) # [batch, length, hidden_size]

        encoder_padding_mask = padding_mask

        x = x.transpose(1, 2) # [batch, hidden_size, length]
        if target_list is not None:
            x, target_list = self.forward_targets(x, target_list)
        features_pen = x.float().pow(2).mean()
        x = x.transpose(1, 2) # [batch, length, hidden_size]
        x = self.layer_norm(x)
        encoder_padding_mask = self.forward_padding_mask(x, encoder_padding_mask)
        if self.post_extract_proj is not None:
            x = self.post_extract_proj(x)
        x = self.dropout_module(x)
        if mask:
            x, mask_indices = self.apply_hubert_mask(
                x, encoder_padding_mask
            )
        else:
            x = x
            mask_indices = None

        if self.use_conv_pos:
            positions = self.pos_conv(x.transpose(1, 2))
            positions = positions.transpose(1, 2)
        #else:
        #    positions = self.embed_positions(encoder_padding_mask)
            x = x + positions

        if self.use_sinc_pos:
            positions = self.embed_positions(encoder_padding_mask)
            x = x + positions

        # x = self.dropout_module(x)

        if require_feat_pen:
            return (x, features_pen, mask_indices, target_list), encoder_padding_mask
        else:
            # For consistence with encoder
            return x, encoder_padding_mask

    def forward_targets(
        self, features: torch.Tensor, target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_padding_mask(
        self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def get_src_lengths(self, src_lengths):
        return self.feature_extractor.get_out_seq_lens_tensor(src_lengths)

    def apply_hubert_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.hubert_mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self.num_updates = num_updates

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        self.conv_layers_infos = conv_layers
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x

    def get_out_seq_lens_nonmask_after_a_layer(self, in_seq_lens_tensor, i):
        """Returns the out_seq_lens_nonmask 0/1 tensor after a layer.

        Args:
            in_seq_lens_tensor (LongTensor): length

        Returns:
            LongTensor: length
        """
        out_lengths = in_seq_lens_tensor.clone()
        out_lengths = ((out_lengths.float() - (self.conv_layers_infos[i][1] - 1) - 1) / self.conv_layers_infos[i][-1] + 1).floor().long()
        out_nonmask = (~lengths_to_padding_mask(out_lengths)).float()
        return out_nonmask, out_lengths

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for i in range(len(self.conv_layers)):
            out = ((out.float() - (self.conv_layers_infos[i][1] - 1) - 1) / self.conv_layers_infos[i][-1] + 1).floor().long()
        return out
