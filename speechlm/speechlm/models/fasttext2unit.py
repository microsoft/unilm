# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import logging
import torch

from fairseq import utils
from fairseq.models import (
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.text_to_speech import fastspeech2

logger = logging.getLogger(__name__)

class VarianceAdaptor(fastspeech2.VarianceAdaptor):
    def __init__(self, args):
        super().__init__(args)
        self.use_pitch = args.use_pitch
        self.use_energe = args.use_energe
    
    def forward(
        self,
        x,
        padding_mask,
        durations=None,
        pitches=None,
        energies=None,
        d_factor=1.0,
        p_factor=1.0,
        e_factor=1.0,
    ):
        # x: B x T x C
        log_dur_out = self.duration_predictor(x)
        dur_out = torch.clamp(
            torch.round((torch.exp(log_dur_out) - 1) * d_factor).long(), min=0
        )
        dur_out.masked_fill_(padding_mask, 0)

        if self.use_pitch:
            pitch_out, pitch_emb = self.get_pitch_emb(x, pitches, p_factor)
            x = x + pitch_emb
        else:
            pitch_out = None

        if self.use_energe:
            energy_out, energy_emb = self.get_energy_emb(x, energies, e_factor)
            x = x + energy_emb
        else:
            energy_out = None

        x, out_lens = self.length_regulator(
            x, dur_out if durations is None else durations
        )

        return x, out_lens, log_dur_out, pitch_out, energy_out


class FastSpeech2Encoder(fastspeech2.FastSpeech2Encoder):
    def __init__(self, args, src_dict, embed_speaker):
        super().__init__(args, src_dict, embed_speaker)
        self.var_adaptor = VarianceAdaptor(args)
        self.apply(fastspeech2.model_init)

@register_model("fasttext2unit")
class FastText2UnitModel(FairseqEncoderModel):
    """
    Implementation for https://arxiv.org/abs/2006.04558
    """

    NON_AUTOREGRESSIVE = True


    @staticmethod
    def add_args(parser):
        parser.add_argument("--dropout", type=float)
        parser.add_argument("--output-frame-dim", type=int)
        parser.add_argument("--speaker-embed-dim", type=int)
        # FFT blocks
        parser.add_argument("--fft-hidden-dim", type=int)
        parser.add_argument("--fft-kernel-size", type=int)
        parser.add_argument("--attention-dropout", type=float)
        parser.add_argument("--encoder-layers", type=int)
        parser.add_argument("--encoder-embed-dim", type=int)
        parser.add_argument("--encoder-attention-heads", type=int)
        parser.add_argument("--decoder-layers", type=int)
        parser.add_argument("--decoder-embed-dim", type=int)
        parser.add_argument("--decoder-attention-heads", type=int)
        # variance predictor
        parser.add_argument("--var-pred-n-bins", type=int)
        parser.add_argument("--var-pred-hidden-dim", type=int)
        parser.add_argument("--var-pred-kernel-size", type=int)
        parser.add_argument("--var-pred-dropout", type=float)
        # postnet
        parser.add_argument("--add-postnet", action="store_true")
        parser.add_argument("--postnet-dropout", type=float)
        parser.add_argument("--postnet-layers", type=int)
        parser.add_argument("--postnet-conv-dim", type=int)
        parser.add_argument("--postnet-conv-kernel-size", type=int)
        # pitch & energe
        parser.add_argument("--use-pitch", action="store_true")
        parser.add_argument("--use-energe", action="store_true")


    def __init__(self, encoder, args, src_dict):
        super().__init__(encoder)
        self._num_updates = 0

    @classmethod
    def build_model(cls, args, task):
        embed_speaker = task.get_speaker_embeddings(args)
        if args.output_frame_dim == -1:
            args.output_frame_dim = len(task.tgt_dict)
        encoder = FastSpeech2Encoder(args, task.src_dict, embed_speaker)
        return cls(encoder, args, task.src_dict)

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self._num_updates = num_updates

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)


@register_model_architecture("fasttext2unit", "fasttext2unit_s")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.output_frame_dim = getattr(args, "output_frame_dim", -1)
    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 256)
    # FFT blocks
    args.fft_hidden_dim = getattr(args, "fft_hidden_dim", 1024)
    args.fft_kernel_size = getattr(args, "fft_kernel_size", 9)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    # variance predictor
    args.var_pred_n_bins = getattr(args, "var_pred_n_bins", 256)
    args.var_pred_hidden_dim = getattr(args, "var_pred_hidden_dim", 256)
    args.var_pred_kernel_size = getattr(args, "var_pred_kernel_size", 3)
    args.var_pred_dropout = getattr(args, "var_pred_dropout", 0.5)
    # postnet
    args.add_postnet = getattr(args, "add_postnet", False)
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
    # pitch & energe
    args.use_pitch = getattr(args, "use_pitch", False)
    args.use_energe = getattr(args, "use_energe", False)


@register_model_architecture("fasttext2unit", "fasttext2unit_m")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.output_frame_dim = getattr(args, "output_frame_dim", -1)
    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 256)
    # FFT blocks
    args.fft_hidden_dim = getattr(args, "fft_hidden_dim", 1024)
    args.fft_kernel_size = getattr(args, "fft_kernel_size", 9)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
    # variance predictor
    args.var_pred_n_bins = getattr(args, "var_pred_n_bins", 256)
    args.var_pred_hidden_dim = getattr(args, "var_pred_hidden_dim", 256)
    args.var_pred_kernel_size = getattr(args, "var_pred_kernel_size", 3)
    args.var_pred_dropout = getattr(args, "var_pred_dropout", 0.5)
    # postnet
    args.add_postnet = getattr(args, "add_postnet", False)
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
    # pitch & energe
    args.use_pitch = getattr(args, "use_pitch", False)
    args.use_energe = getattr(args, "use_energe", False)


@register_model_architecture("fasttext2unit", "fasttext2unit_l")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.2)
    args.output_frame_dim = getattr(args, "output_frame_dim", -1)
    args.speaker_embed_dim = getattr(args, "speaker_embed_dim", 256)
    # FFT blocks
    args.fft_hidden_dim = getattr(args, "fft_hidden_dim", 1536)
    args.fft_kernel_size = getattr(args, "fft_kernel_size", 9)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 384)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 384)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 6)
    # variance predictor
    args.var_pred_n_bins = getattr(args, "var_pred_n_bins", 256)
    args.var_pred_hidden_dim = getattr(args, "var_pred_hidden_dim", 256)
    args.var_pred_kernel_size = getattr(args, "var_pred_kernel_size", 3)
    args.var_pred_dropout = getattr(args, "var_pred_dropout", 0.5)
    # postnet
    args.add_postnet = getattr(args, "add_postnet", False)
    args.postnet_dropout = getattr(args, "postnet_dropout", 0.5)
    args.postnet_layers = getattr(args, "postnet_layers", 5)
    args.postnet_conv_dim = getattr(args, "postnet_conv_dim", 512)
    args.postnet_conv_kernel_size = getattr(args, "postnet_conv_kernel_size", 5)
    # pitch & energe
    args.use_pitch = getattr(args, "use_pitch", False)
    args.use_energe = getattr(args, "use_energe", False)
