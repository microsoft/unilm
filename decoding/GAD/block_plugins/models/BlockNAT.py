# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq import utils
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATModel
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import torch
from fairseq.models.nat.nonautoregressive_transformer import NATransformerEncoder, NATransformerDecoder, NATransformerModel
import logging
import random
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def torch_seed(seed):
    state = torch.random.get_rng_state()
    state_cuda = torch.cuda.random.get_rng_state()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)
        torch.cuda.random.set_rng_state(state_cuda)


@register_model("block")
class BlockNAT(FairseqNATModel):
    forward_decoder = NATransformerModel.forward_decoder
    initialize_output_tokens = NATransformerModel.initialize_output_tokens

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )

    @classmethod
    def build_encoder(cls, args, tgt_dict, embed_tokens):
        encoder = NATransformerEncoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
            self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat=None, **kwargs
    ):
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        nonpad_positions = tgt_tokens.ne(self.pad)
        mask_positions = prev_output_tokens.eq(self.unk) & nonpad_positions
        mask_lens = (mask_positions).sum(1)
        l2r_positions = prev_output_tokens.ne(self.unk) & prev_output_tokens.ne(self.pad)
        l2r_lens = (l2r_positions).sum(1)
        rand_seed = random.randint(0, 19260817)
        glat_info = None
        if glat and tgt_tokens is not None:
            with torch.no_grad():
                with torch_seed(rand_seed):
                    word_ins_out = self.decoder(
                        normalize=False,
                        prev_output_tokens=prev_output_tokens,
                        encoder_out=encoder_out,
                    )
                pred_tokens = word_ins_out.argmax(-1)
                same_num = ((pred_tokens == tgt_tokens) & mask_positions).sum(1)
                input_mask = torch.ones_like(nonpad_positions)
                bsz, seq_len = tgt_tokens.size()
                for li in range(bsz):
                    target_num = (((mask_lens[li] - same_num[li].sum()).float()) * glat['context_p']).long()
                    if target_num > 0:
                        input_mask[li].scatter_(dim=0, index=(torch.randperm(mask_lens[li])[:target_num].cuda() +
                                                              l2r_lens[li]).cuda(), value=0)
                input_mask = input_mask.eq(1)
                tgt_mask = input_mask.masked_fill(~mask_positions, False)
                glat_prev_output_tokens = prev_output_tokens.masked_fill(~input_mask, 0) + tgt_tokens.masked_fill(
                    input_mask, 0)
                glat_tgt_tokens = tgt_tokens.masked_fill(~tgt_mask, self.pad)

                prev_output_tokens, tgt_tokens = glat_prev_output_tokens, glat_tgt_tokens
                glat_info = {
                    "glat_accu": (same_num.sum() / mask_lens.sum()).item(),
                    "glat_context_p": glat['context_p'],
                }
        with torch_seed(rand_seed):
            word_ins_out = self.decoder(
                normalize=False,
                prev_output_tokens=prev_output_tokens,
                encoder_out=encoder_out,
            )

        ret = {
            "word_ins": {
                "out": word_ins_out,
                "tgt": tgt_tokens,
                "mask": tgt_tokens.ne(self.pad),
                "ls": self.args.label_smoothing,
                "nll_loss": True,
            }
        }
        if glat_info is not None:
            ret.update(glat_info)
        return ret


@register_model_architecture(
    "block", "block_6e6d512"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
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
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)


@register_model_architecture(
    "block", "block"
)
def block_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", args.encoder_embed_dim*4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", args.encoder_embed_dim//64)

    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.decoder_embed_dim*4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", args.decoder_embed_dim//64)
    base_architecture(args)

@register_model_architecture(
    "block", "block_base"
)
def base_architecture2(args):
    base_architecture(args)
