# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import math
import torch
import torch.nn as nn
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed, RelativePositionBias
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_heads = num_heads

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v


    def forward_intermediate(self, x, bool_masked_pos=None, layer_id=12):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def get_last_selfattention(self, x):
        B, nc, w, h = x.shape

        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            if x.shape[1] != self.pos_embed.shape[1]:
                x = x + self.interpolate_pos_encoding(x, w, h)
            else:
                x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)
                
class VisionTransformerForMaskedImageModelingCLS(VisionTransformerForMaskedImageModeling):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02,
                 early_layers=6, head_layers=2, shared_lm_head=True):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias, init_std=init_std)

        self.early_layers = early_layers
        print(f'early layer {early_layers}, late layer {depth - early_layers}, condenser head layers {head_layers}, shared_lm_head {shared_lm_head}')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(depth, early_layers + head_layers))]  # stochastic depth decay rule
        self.cls_pt_layers = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(early_layers, early_layers + head_layers)])
        self.fix_init_cls_pt_weight()

        self.shared_lm_head = shared_lm_head
        if not shared_lm_head:
            self.cls_pt_norm = norm_layer(embed_dim)
            self.cls_pt_lm_head = nn.Linear(embed_dim, vocab_size)

            self.cls_pt_norm.apply(self._init_weights)
            self.cls_pt_lm_head.apply(self._init_weights)

    def fix_init_cls_pt_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.cls_pt_layers):
            rescale(layer.attn.proj.weight.data, self.early_layers + layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, self.early_layers + layer_id + 1)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i + 1 == self.early_layers:
                early_states = x[:, 1:]

        x_cls_pt = torch.cat([x[:, [0]], early_states], dim=1)
        for blk in self.cls_pt_layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=rel_pos_bias)

        return self.norm(x), self.norm(x_cls_pt) if self.shared_lm_head else self.cls_pt_norm(x_cls_pt)

    def forward(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        x, x_cls_pt = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        if return_patch_tokens:
            return [x, x_cls_pt]
        if return_all_tokens:
            return [self.lm_head(x), self.lm_head(x_cls_pt) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt)]
        else:
            # return the masked tokens
            return [self.lm_head(x[bool_masked_pos]), self.lm_head(x_cls_pt[bool_masked_pos]) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt[bool_masked_pos])]
    

@register_model
def beit_base_patch16_224_8k_vocab_cls_pt(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_192_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=192, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_256_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_24x544_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=224, patch_size=16, embed_dim=544, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_24x544_patch16_224_8k_vocab_cls_pt(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        img_size=224, patch_size=16, embed_dim=544, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_large_patch16_224_8k_vocab_cls_pt(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_huge_patch14_224_8k_vocab(pretrained=False, **kwargs):
    # patch_size=14, embed_dim=1280, depth=32, num_heads=16
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=8192, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model