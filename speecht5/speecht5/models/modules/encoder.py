# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import contextlib
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    TransformerEncoderLayer,
)
from torch import Tensor
from .transformer_layer import TransformerSentenceEncoderLayer



DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, maxlen=1000, embed_v=False):
        super(RelativePositionalEncoding, self).__init__()

        self.d_model = d_model
        self.maxlen = maxlen
        self.pe_k = torch.nn.Embedding(2*maxlen, d_model) 
        if embed_v:
            self.pe_v = torch.nn.Embedding(2*maxlen, d_model)
        self.embed_v = embed_v


    def forward(self, pos_seq):
        pos_seq[pos_seq < -self.maxlen] = -self.maxlen
        pos_seq[pos_seq >= self.maxlen] = self.maxlen - 1
        pos_seq = pos_seq + self.maxlen
        if self.embed_v:
            return self.pe_k(pos_seq), self.pe_v(pos_seq)
        else:
            return self.pe_k(pos_seq), None

class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, tgt_dict=None, embed_tokens=None):
        self.args = args
        super().__init__(None)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop
        self.freeze_encoder_updates = args.freeze_encoder_updates
        if args.no_freeze_encoder_layer is not None:
            self.no_freeze_encoder_layer = eval(args.no_freeze_encoder_layer)
        else:
            self.no_freeze_encoder_layer = None
        self.num_updates = 0
        export = getattr(args, "export", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self.use_sent_enc_layer = args.use_sent_enc_layer
        self.unb_enc_layer = getattr(args, "unb_enc_layer", -1)

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(args.encoder_embed_dim, eps=args.layer_norm_eps, export=export)
        
        if args.share_ctc_embed and embed_tokens is not None:
            self.proj = nn.Linear(
                embed_tokens.weight.shape[1],
                embed_tokens.weight.shape[0],
                bias=False,
            )
            self.proj.weight = embed_tokens.weight
        elif tgt_dict is not None:
            self.proj = Linear(args.encoder_embed_dim, len(tgt_dict))
        else:
            self.proj = None
        
        if args.relative_position_embedding:
            self.pos_emb = RelativePositionalEncoding(args.encoder_embed_dim//args.encoder_attention_heads, args.encoder_max_relative_position)


    def build_encoder_layer(self, args):
        if args.use_sent_enc_layer:
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
                has_relative_attention_bias=args.relative_position_embedding,
            )
        else:
            layer = TransformerEncoderLayer(args)
        return layer

    def forward(
        self,
        encoder_in,
        encoder_padding_mask,
        return_all_hiddens: bool = False,
        tgt_layer=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.no_freeze_encoder_layer is None:
            ft = self.freeze_encoder_updates <= self.num_updates
        else:
            ft = True
        with torch.no_grad() if not ft else contextlib.ExitStack():
            encoder_out = self.forward_scriptable(
                encoder_in, encoder_padding_mask, return_all_hiddens, tgt_layer=tgt_layer,
            )

        # CTC and bert
        if self.proj:
            x_for_ctc = self.proj(self.dropout_module(encoder_out["encoder_out"][0]))
        else:
            x_for_ctc = None

        encoder_out["encoder_out_for_ctc"] = [x_for_ctc] # T x B x C

        return encoder_out

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        encoder_in,
        encoder_padding_mask,
        return_all_hiddens: bool = False,
        tgt_layer=None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.no_freeze_encoder_layer is not None:
            ft = self.freeze_encoder_updates <= self.num_updates
        else:
            ft = True
        with torch.no_grad() if not ft else contextlib.ExitStack():
            # compute padding mask
            if not self.use_sent_enc_layer:
                has_pads = encoder_in.device.type == "xla" or encoder_padding_mask.any()

            if not self.layer_norm_first:
                encoder_in = self.layer_norm(encoder_in)

            encoder_in = self.dropout_module(encoder_in)

            # B x T x C -> T x B x C
            x = encoder_in.transpose(0, 1)

            encoder_states = []

            if return_all_hiddens:
                encoder_states.append(x)

            ## relative position embedding
            if self.args.relative_position_embedding:
                x_len = x.shape[0]
                pos_seq = torch.arange(0, x_len).long().to(x.device)
                pos_seq = pos_seq[:, None] - pos_seq[None, :]
                pos_k, pos_v = self.pos_emb(pos_seq)
            else:
                pos_k = None

        # encoder layers
        r = None
        d = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()

            with torch.no_grad() if (not ft) and i not in self.no_freeze_encoder_layer else contextlib.ExitStack():
                if not self.training or (dropout_probability > self.encoder_layerdrop) or i == self.unb_enc_layer:
                    if self.use_sent_enc_layer:
                        x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, self_attn_mask=None, need_weights=False, pos_bias=pos_k)
                        # x, _ = layer(x, self_attn_padding_mask=encoder_padding_mask, need_weights=False, pos_bias=pos_k)
                    else:
                        x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask=None)
                        # x = layer(x, encoder_padding_mask=encoder_padding_mask if has_pads else None)
                if i == self.unb_enc_layer:
                    d = x

                if i == tgt_layer:
                    r = x
                    break

                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        with torch.no_grad() if not ft else contextlib.ExitStack():
            # Finally T x B x C
            if self.layer_norm_first:
                x = self.layer_norm(x.transpose(0, 1)).transpose(0, 1)

            if r is not None:
                x = r

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "decoder_input": [d],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]

        if len(encoder_out["encoder_out_for_ctc"]) == 0:
            new_x_for_ctc = []
        else:
            new_x_for_ctc = [encoder_out["encoder_out_for_ctc"][0].index_select(1, new_order)]

        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["decoder_input"]) == 0 or encoder_out["decoder_input"][0] is None:
            new_decoder_input = []
        else:
            new_decoder_input = [
                encoder_out["decoder_input"][0].index_select(0, new_order)
            ]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "encoder_out_for_ctc": new_x_for_ctc, # T x B x C
            "decoder_input": new_decoder_input,
        }

    # def max_positions(self):
    #     """Maximum input length supported by the encoder."""
    #     return self.max_source_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        # if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
        #     weights_key = "{}.embed_positions.weights".format(name)
        #     if weights_key in state_dict:
        #         print("deleting {0}".format(weights_key))
        #         del state_dict[weights_key]
        #     state_dict[
        #         "{}.embed_positions._float_tensor".format(name)
        #     ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            if not isinstance(self.layers[i], TransformerSentenceEncoderLayer):
                self.layers[i].upgrade_state_dict_named(
                    state_dict, "{}.layers.{}".format(name, i)
                )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates
    