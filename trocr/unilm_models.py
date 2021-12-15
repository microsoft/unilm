import torch.nn as nn
import torch

from fairseq.modules.quant_noise import quant_noise
from fairseq.modules import MultiheadAttention
from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.models.transformer import TransformerDecoderBase, TransformerDecoder
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from fairseq.models.transformer import TransformerConfig




class UniLMMultiheadAttention(MultiheadAttention):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, self_attention=False, encoder_decoder_attention=False, q_noise=0, qn_block_size=8):
        super().__init__(embed_dim, num_heads, kdim=kdim, vdim=vdim, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, self_attention=self_attention, encoder_decoder_attention=encoder_decoder_attention, q_noise=q_noise, qn_block_size=qn_block_size)  

        self.k_proj = quant_noise(nn.Linear(self.kdim, embed_dim, bias=True), q_noise, qn_block_size)
        self.k_proj.bias = nn.Parameter(torch.zeros_like(self.k_proj.bias, requires_grad=False))

class UniLMDecoderLayer(TransformerDecoderLayerBase):
    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return UniLMMultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

class UniLMDecoderBase(TransformerDecoderBase):
    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = UniLMDecoderLayer(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

class UniLMDecoder(UniLMDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )
