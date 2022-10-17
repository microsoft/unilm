# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import torch.nn as nn
import math
import torch
import torch.nn.functional as F


class AngularMargin(nn.Module):
    """
    An implementation of Angular Margin (AM) proposed in the following
    paper: '''Margin Matters: Towards More Discriminative Deep Neural Network
    Embeddings for Speaker Recognition''' (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity
    scale : float
        The scale for cosine similiarity

    Return
    ---------
    predictions : torch.Tensor

    Example
    -------
    >>> pred = AngularMargin()
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0):
        super(AngularMargin, self).__init__()
        self.margin = margin
        self.scale = scale

    def forward(self, outputs, targets):
        """Compute AM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        outputs = outputs - self.margin * targets
        return self.scale * outputs


class AdditiveAngularMargin(AngularMargin):
    """
    An implementation of Additive Angular Margin (AAM) proposed
    in the following paper: '''Margin Matters: Towards More Discriminative Deep
    Neural Network Embeddings for Speaker Recognition'''
    (https://arxiv.org/abs/1906.07317)

    Arguments
    ---------
    margin : float
        The margin for cosine similiarity, usually 0.2.
    scale: float
        The scale for cosine similiarity, usually 30.

    Returns
    -------
    predictions : torch.Tensor
        Tensor.
    Example
    -------
    >>> outputs = torch.tensor([ [1., -1.], [-1., 1.], [0.9, 0.1], [0.1, 0.9] ])
    >>> targets = torch.tensor([ [1., 0.], [0., 1.], [ 1., 0.], [0.,  1.] ])
    >>> pred = AdditiveAngularMargin()
    >>> predictions = pred(outputs, targets)
    >>> predictions[:,0] > predictions[:,1]
    tensor([ True, False,  True, False])
    """

    def __init__(self, margin=0.0, scale=1.0, easy_margin=False):
        super(AdditiveAngularMargin, self).__init__(margin, scale)
        self.easy_margin = easy_margin

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, outputs, targets):
        """
        Compute AAM between two tensors

        Arguments
        ---------
        outputs : torch.Tensor
            The outputs of shape [N, C], cosine similarity is required.
        targets : torch.Tensor
            The targets of shape [N, C], where the margin is applied for.

        Return
        ---------
        predictions : torch.Tensor
        """
        cosine = outputs.float()
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        outputs = (targets * phi) + ((1.0 - targets) * cosine)
        return self.scale * outputs


class SpeakerDecoderPostnet(nn.Module):
    """Speaker Identification Postnet.

    Arguments
    ---------
    embed_dim : int
        The size of embedding.
    class_num: int
        The number of classes.
    args : Namespace

    Return
    ---------
    embed : torch.Tensor
    output : torch.Tensor
    """

    def __init__(self, embed_dim, class_num, args):
        super(SpeakerDecoderPostnet, self).__init__()
        self.embed_dim = embed_dim
        self.class_num = class_num
        self.no_pooling_bn = getattr(args, "sid_no_pooling_bn", False)
        self.no_embed_postnet = getattr(args, "sid_no_embed_postnet", False)
        self.normalize_postnet = getattr(args, "sid_normalize_postnet", False)
        self.softmax_head = getattr(args, "sid_softmax_type", "softmax")
        if not self.no_pooling_bn:
            self.bn_pooling = nn.BatchNorm1d(args.decoder_output_dim) 
        else:
            self.bn_pooling = None
        if not self.no_embed_postnet:
            self.output_embedding = nn.Linear(args.decoder_output_dim, embed_dim, bias=False)
            self.bn_embedding = nn.BatchNorm1d(embed_dim)
        else:
            self.output_embedding = None
            self.bn_embedding = None
            self.embed_dim = args.decoder_output_dim
        self.output_projection = nn.Linear(self.embed_dim, class_num, bias=False)
        if self.softmax_head == "amsoftmax":
            self.output_layer = AngularMargin(args.softmax_margin, args.softmax_scale)
        elif self.softmax_head == "aamsoftmax":
            self.output_layer = AdditiveAngularMargin(args.softmax_margin, args.softmax_scale, args.softmax_easy_margin)
        else:
            self.output_layer = None
        if self.output_embedding is not None:
            nn.init.normal_(self.output_embedding.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.normal_(self.output_projection.weight, mean=0, std=class_num ** -0.5)

    def forward(self, x, target=None):
        """
        Parameters
        ----------
        x : torch.Tensor of shape [batch, channel] or [batch, time, channel]
        target : torch.Tensor of shape [batch, channel]
        """
        if self.bn_pooling is not None:
            x = self.bn_pooling(x)
        if self.output_embedding is not None and self.bn_embedding is not None:
            embed = self.bn_embedding(self.output_embedding(x))
        else:
            embed = x
        if self.output_layer is not None or self.normalize_postnet:
            x_norm = F.normalize(embed, p=2, dim=1)
            w_norm = F.normalize(self.output_projection.weight, p=2, dim=1) # [out_dim, in_dim]
            output = F.linear(x_norm, w_norm)
            if self.training and target is not None and self.output_layer is not None:
                output = self.output_layer(output, target)
        else:
            output = self.output_projection(embed)
        return output, embed
