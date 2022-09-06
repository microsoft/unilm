# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.vision_transformer import Attention, Block
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

logger = logging.getLogger(__name__)

class Fp16FixedAttention(Attention):

    def cogview_attn(self, attention_scores, alpha=32):
        '''
        https://arxiv.org/pdf/2105.13290.pdf
        Section 2.4 Stabilization of training: Precision Bottleneck Relaxation (PB-Relax).
        A replacement of the original nn.Softmax(dim=-1)(attention_scores)
        Seems the new attention_probs will result in a slower speed and a little bias
        Can use torch.allclose(standard_attention_probs, cogview_attention_probs, atol=1e-08) for comparison
        The smaller atol (e.g., 1e-08), the better.
        '''
        scaled_attention_scores = attention_scores / alpha
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        # max_value = scaled_attention_scores.amax(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1)
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        return nn.Softmax(dim=-1)(new_attention_scores)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q.float() @ k.float().transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1).type_as(x)
        attn = self.cogview_attn(attn).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x     

class Fp16FixedBlock(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                         attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer,
                         norm_layer=norm_layer)
        self.attn = Fp16FixedAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)


class AdaptedVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        self.ape = kwargs.pop('ape', 0)
        self.mask_ratio = kwargs.pop('mask_ratio', 0.0)        
        self.patch_size = kwargs.get('patch_size')    
        self.fp16fixed = kwargs.pop('fp16fixed', False)   
        weight_init = kwargs.get('weight_init', '')     
        super().__init__(*args, **kwargs)    
        
        if self.ape:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.ape + self.num_tokens, self.embed_dim))
        if self.fp16fixed:
            # img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
            #      num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
            #      drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
            #      act_layer=None, weight_init=''
            embed_dim = kwargs.get('embed_dim', 768)
            num_heads = kwargs.get('num_heads', 12)
            mlp_ratio = kwargs.get('mlp_ratio', 4.)
            qkv_bias = kwargs.get('qkv_bias', True)
            drop_rate = kwargs.get('drop_rate', 0.)
            attn_drop_rate = kwargs.get('attn_drop_rate', 0.)
            drop_path_rate = kwargs.get('drop_path_rate', 0.)
            depth = kwargs.get('depth', 12)
            norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
            act_layer = kwargs.get('act_layer', nn.GELU)

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.blocks = nn.Sequential(*[
            Fp16FixedBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
    
        self.init_weights(weight_init)

    def forward_features(self, x):
        _, _, H, W = x.shape
        Wh = H // self.patch_size
        Ww = W // self.patch_size

        x = self.patch_embed(x)

        if self.mask_ratio != 0:
            probability_matrix = torch.full(x.shape[:2], self.mask_ratio)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            x[masked_indices] = 0

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.ape:            
            pos_embed_patch_num = int(self.pos_embed.size(1) ** 0.5)
            offset = self.num_tokens
            adapt_pos_embed = self.pos_embed[:, offset:, :].view(self.pos_embed.shape[0], pos_embed_patch_num, pos_embed_patch_num, self.pos_embed.shape[-1])  # B 24 24 768
            adapt_pos_embed = adapt_pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(adapt_pos_embed, size=(Wh, Ww), mode='bicubic')
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # B Wh*Ww C
            pos_embed = torch.cat((pos_embed, self.pos_embed[:, :offset, :]), dim=1)
        else:
            pos_embed = self.pos_embed

        input_embedding = x + pos_embed

        x = self.pos_drop(input_embedding)
        x = self.blocks(x)
        x = self.norm(x)
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
    model = AdaptedVisionTransformer(distilled=True,
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
    model = AdaptedVisionTransformer(distilled=True,
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
    model = AdaptedVisionTransformer(distilled=True,
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
    model = AdaptedVisionTransformer(distilled=True,
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
    model = AdaptedVisionTransformer(distilled=True,
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
    model = AdaptedVisionTransformer(distilled=True,
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