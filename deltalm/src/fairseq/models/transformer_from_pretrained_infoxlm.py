# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

import torch
from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
from fairseq.file_io import PathManager


@register_model("transformer_from_pretrained_infoxlm")
class TransformerFromPretrainedInfoXLMModel(TransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-infoxlm-checkpoint",
            type=str,
            metavar="STR",
            help="InfoXLM model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the InfoXLM weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the InfoXLM weights and embeddings into encoder",
        )

    @classmethod
    def build_model(self, args, task, cls_dictionary=MaskedLMDictionary):
        # assert hasattr(args, "pretrained_infoxlm_checkpoint"), (
        #     "You must specify a path for --pretrained-infoxlm-checkpoint to use "
        #     "--arch transformer_from_pretrained_infoxlm"
        # )
        # assert isinstance(task.source_dictionary, cls_dictionary) and isinstance(
        #     task.target_dictionary, cls_dictionary
        # ), (
        #     "You should use a MaskedLMDictionary when using --arch "
        #     "transformer_from_pretrained_xlm because the pretrained XLM model "
        #     "was trained using data binarized with MaskedLMDictionary. "
        #     "For translation, you may want to use --task "
        #     "translation_from_pretrained_xlm"
        # )
        # assert not (
        #     getattr(args, "init_encoder_only", False)
        #     and getattr(args, "init_decoder_only", False)
        # ), "Only one of --init-encoder-only and --init-decoder-only can be set."
        return super().build_model(args, task)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderFromPretrainedInfoXLM(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderFromPretrainedInfoXLM(args, tgt_dict, embed_tokens, no_encoder_attn=getattr(args, "no_cross_attention", False))


def upgrade_state_dict_with_infoxlm_weights(
    state_dict: Dict[str, Any], pretrained_infoxlm_checkpoint: str, num_layers: int, shared_cross_attn: bool=False
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_infoxlm_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    if not os.path.exists(pretrained_infoxlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_infoxlm_checkpoint))

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_infoxlm_checkpoint)
    with open(PathManager.get_local_path(pretrained_infoxlm_checkpoint), "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    infoxlm_state_dict = state["model"]
    # print(state_dict.keys())
    
    for key in infoxlm_state_dict.keys():
        if 'layers' in key and int(key.split('.')[3]) > num_layers-1:
            continue
        if not key.startswith('decoder.'):
            continue
        if 'lm_head' not in key:
            if 'in_proj_weight' in key:
                q, k ,v = infoxlm_state_dict[key].chunk(3, dim=0)
                state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight')] = q
                state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight')] = k
                state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight')] = v
                if shared_cross_attn:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'q_proj.weight').replace('self_attn', 'encoder_attn')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'k_proj.weight').replace('self_attn', 'encoder_attn')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_weight', 'v_proj.weight').replace('self_attn', 'encoder_attn')] = v
            elif 'in_proj_bias' in key:
                q, k ,v = infoxlm_state_dict[key].chunk(3, dim=0)
                state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias')] = q
                state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias')] = k
                state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias')] = v
                if shared_cross_attn:
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'q_proj.bias').replace('self_attn', 'encoder_attn')] = q
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'k_proj.bias').replace('self_attn', 'encoder_attn')] = k
                    state_dict[key.replace('decoder.sentence_encoder.', '').replace('in_proj_bias', 'v_proj.bias').replace('self_attn', 'encoder_attn')] = v
            elif 'emb_layer_norm' in key:
                state_dict[key.replace('decoder.sentence_encoder.emb_layer_norm', 'layernorm_embedding')] = infoxlm_state_dict[key]
            elif 'embed_positions' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key][:state_dict[key.replace('decoder.sentence_encoder.', '')].size(0)]
            elif 'embed_tokens' in key:
                state_dict[key.replace('decoder.sentence_encoder.', '')][:infoxlm_state_dict[key].size(0)] = infoxlm_state_dict[key]
            else:
                state_dict[key.replace('decoder.sentence_encoder.', '')] = infoxlm_state_dict[key]

    return state_dict


class TransformerEncoderFromPretrainedInfoXLM(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if not getattr(args, "init_encoder_only", False):
            return

        assert hasattr(args, "pretrained_infoxlm_checkpoint"), (
            "--pretrained-infoxlm-checkpoint must be specified to load Transformer "
            "encoder from pretrained XLM"
        )
        if hasattr(args, "pretrained_infoxlm_checkpoint") and os.path.exists(args.pretrained_infoxlm_checkpoint):
            infoxlm_loaded_state_dict = upgrade_state_dict_with_infoxlm_weights(
                state_dict=self.state_dict(),
                pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                num_layers=args.encoder_layers,
            )
            self.load_state_dict(infoxlm_loaded_state_dict, strict=False)
            print("Loading encoder from {0}".format(args.pretrained_infoxlm_checkpoint))


class TransformerDecoderFromPretrainedInfoXLM(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if not getattr(args, "init_decoder_only", False):
            return

        assert hasattr(args, "pretrained_infoxlm_checkpoint"), (
            "--pretrained-infoxlm-checkpoint must be specified to load Transformer "
            "decoder from pretrained XLM"
        )
        if hasattr(args, "pretrained_infoxlm_checkpoint") and os.path.exists(args.pretrained_infoxlm_checkpoint):
            infoxlm_loaded_state_dict = upgrade_state_dict_with_infoxlm_weights(
                state_dict=self.state_dict(),
                pretrained_infoxlm_checkpoint=args.pretrained_infoxlm_checkpoint,
                num_layers=args.decoder_layers,
            )
            self.load_state_dict(infoxlm_loaded_state_dict, strict=False)
            print("Loading decoder from {0}".format(args.pretrained_infoxlm_checkpoint))


@register_model_architecture(
    "transformer_from_pretrained_infoxlm", "transformer_from_pretrained_infoxlm"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 12)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)
    args.layer_wise_attention = getattr(args, "layer_wise_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.init_encoder_only = getattr(args, "init_encoder_only", False)
    args.init_decoder_only = getattr(args, "init_decoder_only", False)
    args.max_positions = getattr(args, "max_positions", 512)

