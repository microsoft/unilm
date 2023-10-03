import torch
import torch.nn.functional as F
from torch import nn
from torchscale.architecture.config import EncoderDecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.architecture.encoder import Encoder
from torchscale.component.embedding import PositionalEmbedding
from transformers import CLIPTextModel


class Aligner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.clip_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder",
                                                          torch_dtype=torch.float16, revision="fp16")
        cfg = EncoderDecoderConfig(
            checkpoint_activations=args.checkpoint_activations,
            flash_attention=args.flash_attention,
        )
        self.encoder_proj = Encoder(
            cfg,
            embed_tokens=nn.Linear(args.decoder_embed_dim, 768),
            embed_positions=PositionalEmbedding(32768, 768),
            is_encoder_decoder=True,
        )
        self.encoder_query = nn.Parameter(torch.randn(77, 768))
        self.encoder = Decoder(
            cfg,
            is_encoder_decoder=True,
            causal_mask=False
        )
        self.decoder_query = nn.Parameter(torch.randn(32768, 768))
        self.decoder = Decoder(
            cfg,
            is_encoder_decoder=True,
            causal_mask=False
        )
        self.decoder_proj = Encoder(
            cfg,
            embed_positions=PositionalEmbedding(32768, 768),
            output_projection=nn.Linear(768, args.decoder_embed_dim),
        )

    def forward(self, condition, padding_mask, clip_tokens):
        gpt_embed = self.encoder_proj(
            src_tokens=condition,
            encoder_padding_mask=padding_mask
        )
        gpt_embed = self.encoder(
            prev_output_tokens=None,
            token_embeddings=self.encoder_query.unsqueeze(0).expand(gpt_embed["encoder_out"].shape[1], -1, -1),
            encoder_out=gpt_embed
        )[0]
        with torch.no_grad():
            clip_embed = self.clip_encoder(clip_tokens)[0]
        mse_loss = F.mse_loss(gpt_embed.float(), clip_embed.float(), reduction='mean')
        gpt_embed = self.decoder(
            prev_output_tokens=None,
            token_embeddings=self.decoder_query[:condition.shape[1]].unsqueeze(0).expand(gpt_embed.shape[0], -1, -1),
            encoder_out={"encoder_out": gpt_embed.transpose(0, 1)}
        )[0]
        gpt_embed = self.decoder_proj(
            src_tokens=None,
            token_embeddings=gpt_embed
        )["encoder_out"].transpose(0, 1)
        rec_loss = F.mse_loss(gpt_embed.float(), condition.float(), reduction='mean') * (77 / condition.shape[1])
        return {'clip_loss': {'mse_loss': mse_loss, 'rec_loss': rec_loss}}


class Aligner_encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        cfg = EncoderDecoderConfig(
            checkpoint_activations=args.checkpoint_activations,
            flash_attention=args.flash_attention,
        )
        self.encoder_proj = Encoder(
            cfg,
            embed_tokens=nn.Linear(args.decoder_embed_dim, 768),
            embed_positions=PositionalEmbedding(32768, 768),
            is_encoder_decoder=True,
        )
        self.encoder_query = nn.Parameter(torch.randn(77, 768))
        self.encoder = Decoder(
            cfg,
            is_encoder_decoder=True,
            causal_mask=False
        )

    def forward(self, condition, padding_mask):
        condition = self.encoder_proj(
            src_tokens=condition,
            encoder_padding_mask=padding_mask,
        )
        condition = self.encoder(
            prev_output_tokens=None,
            token_embeddings=self.encoder_query.unsqueeze(0).expand(condition["encoder_out"].shape[1], -1, -1),
            encoder_out=condition,
        )[0]
        return condition
