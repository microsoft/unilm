# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
try:
    from flash_attn import flash_attn_func
    has_flash_attn2 = torch.cuda.get_device_properties(0).major >= 8
except ImportError:
    has_flash_attn2 = False
    print("flash_attn2 not found")

from .DiT import LabelEmbedder, TimestepEmbedder, FinalLayer, SwiGLU, modulate
from .kernel.rotary import apply_rotary_pos_emb as apply_rotary_emb
from .RMSNorm import RMSNorm

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, num_kv_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.n_rep = num_heads // num_kv_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim + 2 * self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, start_pos, rel_pos, incremental_state=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q, k, v = torch.split(qkv, [self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=2)
        q = apply_rotary_emb(q, *rel_pos, interleaved=True)
        k = apply_rotary_emb(k, *rel_pos, interleaved=True)
        if incremental_state is not None:
            incremental_state["key"][:B, start_pos : start_pos + N] = k
            incremental_state["value"][:B, start_pos : start_pos + N] = v
            k = incremental_state["key"][:B, :start_pos + N]
            v = incremental_state["value"][:B, :start_pos + N]
        if has_flash_attn2 and (x.dtype == torch.float16 or x.dtype == torch.bfloat16):
            x = flash_attn_func(q, k, v, causal=True, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            q = q.transpose(1, 2)
            k = repeat_kv(k.transpose(1, 2), self.n_rep)
            v = repeat_kv(v.transpose(1, 2), self.n_rep)
            x = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=incremental_state is None,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
            x = x.transpose(1, 2)

        x = self.proj(x.reshape(B, N, C))
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, mlp_ratio=4.0, proj_drop=0., attn_drop=0., **block_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, num_kv_heads=num_kv_heads, qkv_bias=False, proj_drop=proj_drop, attn_drop=attn_drop, **block_kwargs)
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3 / 64) * 64
        self.mlp = SwiGLU(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x, start_pos, rel_pos, incremental_state=None):
        x = x + self.attn(self.norm1(x), start_pos, rel_pos, incremental_state)
        x = x + self.mlp(self.norm2(x))
        return x

class MLPBlock(nn.Module):
    def __init__(self, hidden_size,  mlp_ratio=4.0, drop=0.0, **block_kwargs):
        super().__init__()
        self.norm = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio * 2 / 3 / 64) * 64
        self.mlp = SwiGLU(hidden_size, mlp_hidden_dim, drop=drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        )

    def forward(self, x, c):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)
        x = x + gate_mlp * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x

class ConditionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=1,
        flatten_input=False,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        diffusion_depth=3,
        num_heads=16,
        num_kv_heads=None,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        posi_scale=1,
        drop=0.0,
        norm_layer=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.flatten_input_size = input_size * input_size
        self.flatten_input = flatten_input
        self.posi_scale = posi_scale

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, strict_img_size=False, norm_layer=norm_layer) if not flatten_input else nn.Linear(in_channels, hidden_size, bias=False)
        self.noisy_x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, strict_img_size=False, norm_layer=norm_layer) if not flatten_input else nn.Linear(in_channels, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self._precomputed_freqs_cis = None

        self.blocks = nn.ModuleList([
            Block(hidden_size, self.num_heads, self.num_kv_heads, mlp_ratio=mlp_ratio, proj_drop=drop, attn_drop=drop) for _ in range(depth)
        ])
        self.diffusion_blocks = nn.ModuleList([
            MLPBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(diffusion_depth)
        ])
        self.condition_layer = ConditionLayer(hidden_size)
        self.final_layer = FinalLayer(hidden_size, patch_size * patch_size * self.out_channels if not flatten_input else self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if not self.flatten_input:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table, timestep embedding MLP, and CLS:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.diffusion_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def build_rel_pos(self, x, start_pos = 0):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / ((10000 * self.posi_scale) ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(self.flatten_input_size).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))

        return rel_pos

    def forward(self, x_noise, t, x_start, y, batch_mul=1):
        """
        Forward pass of ransformer.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        condition = self.forward_parallel(x_start, y)
        condition = condition.repeat_interleave(batch_mul, dim=0)
        x = self.forward_diffusion(x_noise, t, condition)
        return x
    
    def forward_parallel(self, x, y):
        x = self.x_embedder(x)
        y = self.y_embedder(y, self.training)
        x = torch.cat((y.unsqueeze(1), x[:, :-1]), dim=1)
        rel_pos = self.build_rel_pos(x)
        for block in self.blocks:
            x = block(x, 0, rel_pos)
 
        x = self.condition_layer(x)
        return x
    
    def forward_recurrent(self, x, start_pos = 0, incremental_state = None):
        start_pos = start_pos if start_pos != 0 else 0
        x = self.y_embedder(x, self.training).unsqueeze(1) if start_pos == 0 else self.x_embedder(x)
        rel_pos = self.build_rel_pos(x, start_pos)
        for idx, block in enumerate(self.blocks):
            if incremental_state is not None and idx not in incremental_state:
                incremental_state[idx] = {
                    "key": torch.empty(x.shape[0], self.flatten_input_size, self.num_kv_heads, self.head_dim, device=x.device, dtype=x.dtype),
                    "value": torch.empty(x.shape[0], self.flatten_input_size, self.num_kv_heads, self.head_dim, device=x.device, dtype=x.dtype),
                }
            x = block(x, start_pos, rel_pos, incremental_state[idx])
            
        x = self.condition_layer(x[:, -1:])
        return x
    
    def forward_diffusion(self, x, t, condition):
        bsz, seq_len = t.shape if t.dim() > 1 else (t.shape[0], 1)
        t = self.t_embedder(t.view(-1)).view(bsz, seq_len, -1)
        c = condition + t
        x = self.noisy_x_embedder(x)
        
        for block in self.diffusion_blocks:
            x = block(x, c)
            
        x = self.final_layer(x, c)
        if not self.flatten_input:
            x = self.unpatchify(x)         # (N, out_channels, H, W)
        return x
    
    def sample_with_cfg(self, prev_token, cfg_scale, sample_func):
        bsz, half_bsz = prev_token.shape[0], prev_token.shape[0] // 2
        incremental_state = {}
        samples = []
        for i in range(self.flatten_input_size):
            if self.flatten_input:
                z = torch.randn(bsz, 1, self.in_channels, device=self.device, dtype=self.dtype)
            else:
                p = self.noisy_x_embedder.patch_size[0]
                h = w = self.input_size // p
                z = torch.randn(bsz, self.in_channels, p, p, device=self.device, dtype=self.dtype)
            recurrent_input = torch.cat([prev_token, prev_token], dim=0) if i != 0 else prev_token
            condition = self.forward_recurrent(recurrent_input, start_pos = i, incremental_state=incremental_state)
            prev_token = sample_func(functools.partial(self.forward_with_cfg, condition=condition, cfg_scale=cfg_scale), z)
            prev_token, _ = prev_token.chunk(2, dim=0)  # Remove null class samples
            samples.append(prev_token)
        if self.flatten_input:
            samples = torch.cat(samples, 1)
        else:
            samples = torch.stack(samples, 2).view(half_bsz, self.in_channels, h, w, p, p).permute(0, 1, 2, 4, 3, 5).reshape(half_bsz, self.in_channels, h * p, w * p)
        return samples

    def forward_with_cfg(self, x, t, condition, cfg_scale):
        """
        Forward pass of ClassTransformer, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward_diffusion(combined, t, condition)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps

#################################################################################
#                                   Transformer Configs                                  #
#################################################################################

def Transformer_13B(**kwargs):
    return Transformer(depth=40, hidden_size=5120, num_heads=40, mlp_ratio=6, **kwargs)

def Transformer_7B(**kwargs):
    return Transformer(depth=32, hidden_size=4096, num_heads=32, mlp_ratio=6, **kwargs)

def Transformer_3B(**kwargs):
    return Transformer(depth=32, hidden_size=2560, num_heads=20, mlp_ratio=6, **kwargs)

def Transformer_XL(**kwargs):
    return Transformer(depth=24, hidden_size=2048, num_heads=16, mlp_ratio=6, **kwargs)

def Transformer_Large(**kwargs):
    return Transformer(depth=24, hidden_size=1536, num_heads=12, mlp_ratio=6, **kwargs)

def Transformer_Medium(**kwargs):
    return Transformer(depth=24, hidden_size=1024, num_heads=16, mlp_ratio=6, **kwargs)

def Transformer_Base(**kwargs):
    return Transformer(depth=12, hidden_size=768, num_heads=12, mlp_ratio=6, **kwargs)

def Transformer_H(**kwargs):
    return Transformer(depth=40, hidden_size=1280, num_heads=20, mlp_ratio=4, diffusion_depth=12, **kwargs)

def Transformer_L(**kwargs):
    return Transformer(depth=32, hidden_size=1024, num_heads=16, mlp_ratio=4, diffusion_depth=8, **kwargs)

def Transformer_B(**kwargs):
    return Transformer(depth=24, hidden_size=768, num_heads=12, mlp_ratio=4, diffusion_depth=6, **kwargs)

Transformer_models = {
    'Transformer-13B': Transformer_13B, 'Transformer-7B': Transformer_7B, 'Transformer-3B': Transformer_3B, 'Transformer-XL': Transformer_XL, 'Transformer-Large': Transformer_Large, 'Transformer-Medium':  Transformer_Medium, 'Transformer-Base': Transformer_Base,
    'Transformer-H': Transformer_H, 'Transformer-L': Transformer_L, 'Transformer-B': Transformer_B
}