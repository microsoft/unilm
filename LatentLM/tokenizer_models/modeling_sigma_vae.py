import torch
from torch import nn
from torch.nn import functional as F
from timm.models.registry import register_model

from .modeling_common import EncoderDecoderArchForImageReconstrction, get_basic_config


class DecodeHeadBLC(nn.Module):
    def __init__(self, decoder_output_dim, patch_size, output_channels, patches_shape):
        super().__init__()
        num_pixels_per_patch = patch_size * patch_size * output_channels
        self.patch_size = patch_size
        self.output_channels = output_channels

        self.fc1 = nn.Linear(decoder_output_dim, decoder_output_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(decoder_output_dim, num_pixels_per_patch)
        self.patches_shape = patches_shape

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        bsz = x.size(0)
        x = x.view(
            bsz, self.patches_shape[0], self.patches_shape[1], 
            self.output_channels, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 4, 2, 5)
        x =  x.reshape(
            bsz, self.output_channels, 
            self.patches_shape[0] * self.patch_size, 
            self.patches_shape[1] * self.patch_size, 
        )
        return x


class GaussianDistribution(object):
    def __init__(self, parameters, std):
        self.parameters = parameters
        self.mean = parameters
        self.std = std

    def sample(self, sampling_std=None):
        if sampling_std is not None:
            x = self.mean + sampling_std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        else:
            batch_size = self.mean.size(0)
            value = self.std / 0.8
            std = torch.randn(batch_size).to(device=self.parameters.device) * value

            while std.dim() < self.mean.dim():
                std = std.unsqueeze(-1)

            x = self.mean + std * torch.randn(self.mean.shape).to(device=self.parameters.device)

        return x

    def kl(self):
        target = torch.zeros_like(self.mean)
        return F.mse_loss(self.mean, target, reduction='mean')

    def mode(self):
        return self.mean


class EncodeHeadBLC(nn.Module):
    def __init__(self, output_dim, latent_size, patches_shape, std):
        super().__init__()
        self.dense = nn.Linear(output_dim, latent_size)
        self.patches_shape = patches_shape
        self.latent_size = latent_size
        self.std = std

    def forward(self, x):
        bsz = x.size(0)
        x = self.dense(x)
        x = x.reshape(bsz, self.patches_shape[0], self.patches_shape[1], self.latent_size)
        x = x.permute(0, 3, 1, 2)

        x = GaussianDistribution(x, self.std)
        return x


class SigmaVAE(EncoderDecoderArchForImageReconstrction):
    # SigmaVAE
    def __init__(
            self, 
            encoder_config: dict, 
            decoder_config: dict, 
            patch_size: int, 
            latent_size: int = 16,
            kl_weight: float = 1e-2,
            std: float = 0.75,
    ):
        img_size = encoder_config['img_size']
        patches_shape = (img_size // patch_size, img_size // patch_size, latent_size)
        num_patches = (encoder_config['img_size'] // patch_size) ** 2
        self.num_patches = num_patches

        encoder_post_processor = EncodeHeadBLC(
            encoder_config['embed_dim'], latent_size, 
            patches_shape, std=std
        )
        
        decoder_pre_processor = nn.Identity()

        decoder_post_processor = DecodeHeadBLC(
            decoder_config['embed_dim'], patch_size, encoder_config['in_chans'], patches_shape)

        super().__init__(
            encoder_config=encoder_config,
            encoder_post_processor=encoder_post_processor,
            decoder_pre_processor=decoder_pre_processor,
            decoder_config=decoder_config,
            decoder_post_processor=decoder_post_processor,
        )
        self.kl_weight = kl_weight

        self.init_weights()

    
@register_model
def sigma_vae(latent_size, std, **kwargs):
    basic_config, unused_kwargs = get_basic_config(**kwargs)
    decoder_config = basic_config.pop('decoder_config')

    decoder_config['patch_size'] = 1
    # if decoder is vit arch, adjust the image size to be the size of the latent space 
    # without modification for the vit implementation
    decoder_config['img_size'] = kwargs['img_size'] // kwargs['patch_size']
    decoder_config['in_chans'] = latent_size

    print("Unused args = %s" % str(unused_kwargs))
    model = SigmaVAE(
        latent_size=latent_size, std=std, 
        decoder_config=decoder_config, **basic_config)
    return model
