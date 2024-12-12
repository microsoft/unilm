# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch.nn as nn

from torchscale.architecture.encoder import Encoder
from torchscale.component.embedding import (
    PositionalEmbedding,
    VisionEmbedding,
)
from torchscale.architecture.config import EncoderConfig


class BEiT3Vision(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        assert not args.multiway
        assert not args.share_encoder_input_output_embed
        self.vision_embed = VisionEmbedding(
            args.img_size,
            args.patch_size,
            args.in_chans,
            args.encoder_embed_dim,
            contain_mask_token=True,
            prepend_cls_token=True,
        )
        # being consistent with Fairseq, which starts from 2 for position embedding
        embed_positions = PositionalEmbedding(self.vision_embed.num_position_embeddings() + 2, args.encoder_embed_dim)
        self.encoder = Encoder(
            args,
            embed_tokens=None,
            embed_positions=embed_positions,
            output_projection=None,
            is_encoder_decoder=False,
        )

    def forward(
        self,
        visual_tokens=None,
        vision_masked_position=None,
        return_patch_tokens=False,
    ):
        x = self.vision_embed(visual_tokens, vision_masked_position)

        x = self.encoder(
            src_tokens=None,
            encoder_padding_mask=None,
            token_embeddings=x,
        )

        encoder_out = x["encoder_out"]

        if return_patch_tokens:
            return encoder_out[:, 1:]
        else:
            return encoder_out[:, 0]


def beit3_base_vision(image_size):
    config = EncoderConfig(
        img_size=image_size, patch_size=16, vocab_size=64010, multiway=False, 
        layernorm_embedding=False, normalize_output=True, no_output_layer=True, 
        drop_path_rate=0, encoder_embed_dim=768, encoder_attention_heads=12, 
        encoder_ffn_embed_dim=int(768 * 4), encoder_layers=12, 
    )
    return BEiT3Vision(config)
