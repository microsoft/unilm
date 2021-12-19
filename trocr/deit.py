# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_



logger = logging.getLogger(__name__)

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

from itertools import repeat
import collections.abc

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


class PatchEmbedForApe(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ape=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        x = self.proj(x)
        return x


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, ape=False, mask_ratio=0.0):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, representation_size=representation_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer)
        self.ape = ape
        self.mask_ratio = mask_ratio

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))        
        self.patch_embed = PatchEmbedForApe(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)        
        num_patches = self.patch_embed.num_patches if not self.ape else 576        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)  # B C Wh Ww

        if self.ape:
            Wh, Ww = x.size(2), x.size(3)
            adapt_pos_embed = self.pos_embed[:, 2:, :].view(self.pos_embed.shape[0], 24, 24, self.pos_embed.shape[-1])  # B 24 24 768
            adapt_pos_embed = adapt_pos_embed.permute(0, 3, 1, 2)
            absolute_pos_embed = F.interpolate(adapt_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = x.flatten(2).transpose(1, 2)  # B Wh*Ww C
            if self.mask_ratio != 0:
                probability_matrix = torch.full(x.shape[:2], self.mask_ratio)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                x[masked_indices] = 0

            x = x + absolute_pos_embed.flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)  # B Wh*Ww C
            if self.mask_ratio != 0:
                probability_matrix = torch.full(x.shape[:2], self.mask_ratio)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                x[masked_indices] = 0

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        if not self.ape:
            x = x + self.pos_embed

        input_embedding = x
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x, input_embedding

    def forward(self, x):
        x, input_embedding = self.forward_features(x)
        return x, input_embedding

class AdaptedVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, ape=False, mask_ratio=0.0):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, representation_size=representation_size, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, hybrid_backbone=hybrid_backbone, norm_layer=norm_layer)
        self.ape = ape
        self.mask_ratio = mask_ratio

        self.patch_embed = PatchEmbedForApe(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)        
        num_patches = self.patch_embed.num_patches if not self.ape else 576        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if self.ape:
            Wh, Ww = x.size(2), x.size(3)
            adapt_pos_embed = self.pos_embed[:, 1:, :].view(self.pos_embed.shape[0], 24, 24, self.pos_embed.shape[-1])  # B 24 24 768
            adapt_pos_embed = adapt_pos_embed.permute(0, 3, 1, 2)
            absolute_pos_embed = F.interpolate(adapt_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = x.flatten(2).transpose(1, 2)  # B Wh*Ww C
            if self.mask_ratio != 0:
                probability_matrix = torch.full(x.shape[:2], self.mask_ratio)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                x[masked_indices] = 0

            x = x + absolute_pos_embed.flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)  # B Wh*Ww C
            if self.mask_ratio != 0:
                probability_matrix = torch.full(x.shape[:2], self.mask_ratio)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                x[masked_indices] = 0

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        
        if not self.ape:
            x = x + self.pos_embed

        input_embedding = x
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)        
        return x, input_embedding
    
    def forward(self, x):
        x, input_embedding = self.forward_features(x)
        return x, input_embedding


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_small_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        # adapt 224 model to 384
        model_seq_len = model.state_dict()['pos_embed'].shape[1]
        ckpt_seq_len = checkpoint['model']['pos_embed'].shape[1]
        logger.warning('Deit load {:d} seq len to {:d} APE {}'.format(ckpt_seq_len, model_seq_len, str(model.ape)))
        if not model.ape:
            if model_seq_len <= ckpt_seq_len:
                checkpoint['model']['pos_embed'] = checkpoint['model']['pos_embed'][:, :model_seq_len, :]
            else:
                t = model.state_dict()['pos_embed']
                t[:, :ckpt_seq_len, :] = checkpoint['model']['pos_embed']
                checkpoint['model']['pos_embed'] = t

        model.load_state_dict(checkpoint["model"])
    return model



@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_custom_size(pretrained=False, img_size=384, **kwargs):
    model = DistilledVisionTransformer(
        img_size=img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        # checkpoint['model']['pos_embed'] = checkpoint['model']['pos_embed'][:, :502, :]        
        # ape torch.Size([1, 578, 768]) from checkpoint, the shape in current model is torch.Size([1, 1026, 768]).
        model_seq_len = model.state_dict()['pos_embed'].shape[1]
        ckpt_seq_len = checkpoint['model']['pos_embed'].shape[1]
        logger.warning('Deit load {:d} seq len to {:d} APE {}'.format(ckpt_seq_len, model_seq_len, str(model.ape)))
        if not model.ape:
            if model_seq_len <= ckpt_seq_len:
                checkpoint['model']['pos_embed'] = checkpoint['model']['pos_embed'][:, :model_seq_len, :]
            else:
                t = model.state_dict()['pos_embed']
                t[:, :ckpt_seq_len, :] = checkpoint['model']['pos_embed']
                checkpoint['model']['pos_embed'] = t

        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model = AdaptedVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model = AdaptedVisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model