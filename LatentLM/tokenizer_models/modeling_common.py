import torch
from torch import nn

from timm.models.layers import trunc_normal_ as __call_trunc_normal_

from .modeling_utils import VisionTransformer
from .modeling_beit3_vision import beit3_base_vision

from functools import partial


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class EncoderDecoderArchForImageReconstrction(nn.Module):
    # This is the main class for the encoder-decoder architecture
    # It is used for image reconstruction
    # contains encoer backbone, decoder backbone
    def __init__(
            self, 
            encoder_config: dict, 
            encoder_post_processor: nn.Module,
            decoder_pre_processor: nn.Module,
            decoder_config: dict, 
            decoder_post_processor: nn.Module,
    ):
        super().__init__()
        self.img_size = encoder_config['img_size']

        self.encoder = self.build_encoder(encoder_config)
        self.encoder_post_processor = encoder_post_processor
        
        self.decoder_pre_processor = decoder_pre_processor
        self.decoder = self.build_decoder(decoder_config)
        self.decoder_post_processor = decoder_post_processor
        
    def init_weights(self):
        if self.encoder_post_processor is not None:
            self.encoder_post_processor.apply(self._init_weights)
        if self.decoder_pre_processor is not None:
            self.decoder_pre_processor.apply(self._init_weights)
        if self.decoder_post_processor is not None:
            self.decoder_post_processor.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def build_encoder(config):
        backbone = config.pop('arch')
        if backbone.startswith('vit'):
            module = VisionTransformer(**config)
        elif backbone == 'beit3-base':
            module = beit3_base_vision(image_size=config["img_size"])
        
        return module

    @staticmethod
    def build_decoder(config):
        backbone = config.pop('arch')
        return VisionTransformer(**config)

    def encode(self, img):
        encoder_features = self.encoder(img, return_patch_tokens=True)
        return self.encoder_post_processor(encoder_features)

    def decode(self, quantize, **decoder_kwargs):
        quantize = self.decoder_pre_processor(quantize)
        decoder_features = self.decoder(quantize, return_patch_tokens=True, **decoder_kwargs)
        return self.decoder_post_processor(decoder_features)

def get_model_default_params(
        embed_dim=768, depth=12, img_size=256,
        patch_size=16, in_chans=3, num_heads=12,
):
    return dict(
        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
        embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.,
        qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def get_basic_config(
        img_size=256, patch_size=16, 
        encoder_arch='beit3-base', decoder_arch='vit-base',
        **kwargs,
):
    if encoder_arch in ('vit-base', 'beit3-base'):
        encoder_config = get_model_default_params(
            embed_dim=768, depth=12, num_heads=12, 
        )
    else:
        raise ValueError(f"Unknown encoder arch: {encoder_arch}")

    encoder_config['patch_size'] = patch_size
    encoder_config['img_size'] = img_size
    encoder_config['arch'] = encoder_arch

    if decoder_arch == 'vit-base':
        decoder_config = get_model_default_params(
            embed_dim=768, depth=12, num_heads=12, 
        )
    else:
        raise ValueError(f"Unknown decoder arch: {decoder_arch}")
    
    decoder_config['arch'] = decoder_arch
    return {
        'encoder_config': encoder_config,
        'decoder_config': decoder_config,
        'patch_size': patch_size,
    }, kwargs
