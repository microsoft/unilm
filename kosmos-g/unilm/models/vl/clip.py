import logging
import os
import torch

from copy import deepcopy
from typing import Tuple, Union, Callable, Optional
from torch import nn
from torch.nn import functional as F
from open_clip.model import CLIP, CLIPVisionCfg, QuickGELU, TimmModel, ModifiedResNet, VisualTransformer, to_2tuple, LayerNorm, Transformer
from open_clip.factory import _MODEL_CONFIGS, list_models, load_checkpoint, get_pretrained_url, download_pretrained, load_state_dict


logger = logging.getLogger(__name__)


class VisualTransformer4Seq2Seq(nn.Module):
    def __init__(
        self, image_size: int, patch_size: int, width: int, layers: int, heads: int, mlp_ratio: float, output_dim: int, act_layer: Callable = nn.GELU):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)    # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)    # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)    # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)    # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)    # NLD -> LND
        x = self.transformer(x)
        # NOTE encoder output is T, B, C for seq2seq
        # x = x.permute(1, 0, 2)    # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x) # [*, grid ** 2 + 1, width]

        # if self.proj is not None:
        #     x = x @ self.proj
        return x


class ClipVisualOnly(nn.Module):

    # text_cfg for compatibility with original CLIP
    def __init__(self, embed_dim, vision_cfg, text_cfg, quick_gelu=False):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        # NOTE: timm models always use native GELU regardless of quick_gelu flag.
        act_layer = QuickGELU if quick_gelu else nn.GELU

        if vision_cfg.timm_model_name:
            raise NotImplementedError
            self.visual = TimmModel(
                vision_cfg.timm_model_name,
                pretrained=vision_cfg.timm_model_pretrained,
                pool=vision_cfg.timm_pool,
                proj=vision_cfg.timm_proj,
                embed_dim=embed_dim,
                image_size=vision_cfg.image_size)
            act_layer = nn.GELU    # so that text transformer doesn't use QuickGELU w/ timm models
        elif isinstance(vision_cfg.layers, (tuple, list)):
            raise NotImplementedError
            vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
            self.visual = ModifiedResNet(
                layers=vision_cfg.layers,
                output_dim=embed_dim,
                heads=vision_heads,
                image_size=vision_cfg.image_size,
                width=vision_cfg.width)
        else:
            vision_heads = vision_cfg.width // vision_cfg.head_width
            self.visual = VisualTransformer4Seq2Seq(
                image_size=vision_cfg.image_size,
                patch_size=vision_cfg.patch_size,
                width=vision_cfg.width,
                layers=vision_cfg.layers,
                heads=vision_heads,
                mlp_ratio=vision_cfg.mlp_ratio,
                output_dim=embed_dim,
                act_layer=act_layer,)
        self.init_parameters()
    
    def init_parameters(self):
        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
    
    def encode_image(self, image):
        return self.visual(image)
    
    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        return image_features


def load_checkpoint4vision_only(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
    model_name: str,
    pretrained: str = '',
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,):
    model_name = model_name.replace('/', '-')    # for callers using old naming with / in ViT names

    if pretrained.lower() == 'openai':
        raise NotImplementedError
    else:
        if model_name in _MODEL_CONFIGS:
            logger.info(f'Loading {model_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                    # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        model = ClipVisualOnly(**model_cfg)
        
        if pretrained:
            checkpoint_path = ''
            url = get_pretrained_url(model_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                # NOTE TODO remove, strict=True is only for debug
                load_checkpoint4vision_only(model, checkpoint_path, strict=False)
                # reload attn weights into ts attn
                dim = model.visual.transformer.resblocks[0].attn.in_proj_weight.shape[0] // 3
                for resblock in model.visual.transformer.resblocks:
                    resblock.ts_attn.q_proj.weight = nn.Parameter(resblock.attn.in_proj_weight[:dim].clone())
                    resblock.ts_attn.q_proj.bias = nn.Parameter(resblock.attn.in_proj_bias[:dim].clone())
                    resblock.ts_attn.k_proj.weight = nn.Parameter(resblock.attn.in_proj_weight[dim:2*dim].clone())
                    resblock.ts_attn.k_proj.bias = nn.Parameter(resblock.attn.in_proj_bias[dim:2*dim].clone())
                    resblock.ts_attn.v_proj.weight = nn.Parameter(resblock.attn.in_proj_weight[2*dim:].clone())
                    resblock.ts_attn.v_proj.bias = nn.Parameter(resblock.attn.in_proj_bias[2*dim:].clone())
                    resblock.ts_attn.out_proj.weight = nn.Parameter(resblock.attn.out_proj.weight.clone())
                    resblock.ts_attn.out_proj.bias = nn.Parameter(resblock.attn.out_proj.bias.clone())
                    resblock.attn = None
            else:
                logging.warning(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
                raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')

        if jit: model = torch.jit.script(model)

    return model