"""Whole-page LayoutLMv3: text inputs longer than the 512 positions of the released checkpoints.

Two changes to Hugging Face ``transformers``' LayoutLMv3, both applied to an already loaded model:

* ``extend_position_embeddings``: tile the learned 1D position embeddings to more positions (as Longformer did for
  RoBERTa). No pre-training; the model is then fine-tuned on long pages as usual.
* ``use_lut_attention``: memory-lean exact attention. The 1D/2D relative position biases depend only on the bucketed
  distance between two tokens, so they are gathered once per forward pass from small per-distance tables, in the
  compute dtype, shared by all layers, and passed to ``scaled_dot_product_attention`` as its additive mask. The
  memory-efficient SDPA kernel runs on V100 tensor cores and does not store attention probabilities.
"""

from __future__ import annotations

import math
import types

import torch
import torch.nn.functional as F
from torch import nn


def extend_position_embeddings(model: nn.Module, max_length: int) -> nn.Module:
    """Give a LayoutLMv3 model 1D position embeddings for ``max_length`` text tokens by repeating the learned ones.

    ``model`` is a ``LayoutLMv3Model`` or any model with a ``.layoutlmv3`` backbone. LayoutLMv3 numbers positions
    RoBERTa-style, from ``padding_idx + 1``, so the reserved rows are kept and the learned rows after them are tiled:
    position ``512 + i`` reuses the vector of position ``i``. The relative position biases are bucketed by distance
    and need no change.
    """
    backbone = getattr(model, "layoutlmv3", model)
    emb = backbone.embeddings
    old = emb.position_embeddings.weight.data
    offset = emb.padding_idx + 1
    n = max_length + offset
    if n <= old.shape[0]:
        return model
    learned = old[offset:]
    index = torch.arange(n - offset, device=old.device) % learned.shape[0]
    new = nn.Embedding(n, old.shape[1], padding_idx=emb.padding_idx).to(old.device, old.dtype)
    new.weight.data = torch.cat([old[:offset], learned[index]]).clone()
    emb.position_embeddings = new
    emb.register_buffer("position_ids", torch.arange(n, device=old.device).expand((1, -1)), persistent=False)
    backbone.config.max_position_embeddings = n
    if backbone is not model:
        model.config.max_position_embeddings = n
    return model


def _bucket(rel: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
    """``LayoutLMv3Encoder.relative_position_bucket``."""
    num_buckets //= 2
    ret = (rel > 0).to(torch.int64) * num_buckets
    n = torch.abs(rel)
    max_exact = num_buckets // 2
    large = max_exact + (torch.log(torch.clamp(n, min=1).float() / max_exact)
                         / math.log(max_distance / max_exact) * (num_buckets - max_exact)).to(torch.int64)
    large = torch.clamp(large, max=num_buckets - 1)
    return ret + torch.where(n < max_exact, n, large)


def use_lut_attention(model: nn.Module) -> nn.Module:
    """Replace LayoutLMv3's self-attention by SDPA with a relative position bias built once per forward pass.

    The results are the same as ``transformers``' eager attention. ``output_attentions`` and ``head_mask`` are not
    supported. As in ``transformers`` since 4.42, the relative position bias tables are not trained
    (huggingface/transformers#50636).
    """
    from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Encoder, LayoutLMv3SelfAttention

    encoder = next(m for m in model.modules() if isinstance(m, LayoutLMv3Encoder))
    state: dict = {}

    # The encoder normally builds (batch, heads, n, n) biases here; record the positions and boxes instead.
    def record_1d(self, position_ids):
        state["pos"] = position_ids
        return position_ids.new_zeros(())

    def record_2d(self, bbox):
        state["x"], state["y"] = bbox[:, :, 0], bbox[:, :, 3]
        return bbox.new_zeros(())

    encoder._cal_1d_pos_emb = types.MethodType(record_1d, encoder)
    encoder._cal_2d_pos_emb = types.MethodType(record_2d, encoder)
    n_xy = 1024   # bbox coordinates are 0-1000

    def tables(dtype, device, scale):
        """(heads, distances) bias tables, already bucketed and scaled, indexed by distance + offset."""
        n_pos = encoder.config.max_position_embeddings
        key = (dtype, device, scale, n_pos)
        if state.get("tables_key") != key:
            def table(weights, distances, bins, max_rel):
                return (weights.weight.detach().t()[_bucket(distances, bins, max_rel)] * scale).to(dtype).t().contiguous()

            d1 = torch.arange(-n_pos, n_pos + 1, device=device)
            d2 = torch.arange(-n_xy, n_xy + 1, device=device)
            t1 = (table(encoder.rel_pos_bias, d1, encoder.rel_pos_bins, encoder.max_rel_pos)
                  if encoder.has_relative_attention_bias else None)
            tx = ty = None
            if encoder.has_spatial_attention_bias:
                tx = table(encoder.rel_pos_x_bias, d2, encoder.rel_2d_pos_bins, encoder.max_rel_2d_pos)
                ty = table(encoder.rel_pos_y_bias, d2, encoder.rel_2d_pos_bins, encoder.max_rel_2d_pos)
            state["tables"], state["tables_key"] = (t1, tx, ty, n_pos), key
        return state["tables"]

    def shared_bias(self, q, attention_mask, scale):
        key = (id(state.get("pos")), id(attention_mask), q.dtype)
        if state.get("bias_key") == key:
            return state["bias"]
        t1, tx, ty, n_pos = tables(q.dtype, q.device, scale)
        bias = None   # (heads, batch, n, n); int32 indices keep the index temporaries small
        if self.has_relative_attention_bias:
            p = state["pos"].int()
            bias = t1[:, p[:, None, :] - p[:, :, None] + n_pos]
        if self.has_spatial_attention_bias:
            for coord, t in ((state["x"], tx), (state["y"], ty)):
                c = coord.int()
                g = t[:, c[:, None, :] - c[:, :, None] + n_xy]
                bias = g if bias is None else bias.add_(g)
                del g
        valid = None
        if attention_mask is not None:   # transformers' extended additive mask -> (batch, keys) bool
            valid = attention_mask[:, 0, 0, :] > torch.finfo(attention_mask.dtype).min / 2
        if bias is None and valid is not None:
            bias = q.new_zeros(1, valid.shape[0], q.shape[2], q.shape[2])
        if bias is not None and valid is not None:
            bias.masked_fill_(~valid[None, :, None, :], torch.finfo(bias.dtype).min)
        bias = None if bias is None else bias.transpose(0, 1)   # (batch, heads, n, n) view
        # keep the inputs alive so their ids cannot be reused while the cached bias is in use
        state["bias"], state["bias_key"], state["bias_refs"] = bias, key, (state.get("pos"), attention_mask)
        return bias

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, rel_pos=None,
                rel_2d_pos=None):
        if output_attentions or head_mask is not None:
            raise NotImplementedError("use_lut_attention does not support output_attentions or head_mask")
        b, n, _ = hidden_states.shape
        shape = (b, n, self.num_attention_heads, self.attention_head_size)
        q = self.query(hidden_states).view(shape).transpose(1, 2)
        k = self.key(hidden_states).view(shape).transpose(1, 2)
        v = self.value(hidden_states).view(shape).transpose(1, 2)
        scale = 1 / math.sqrt(self.attention_head_size)
        bias = shared_bias(self, q, attention_mask, scale)
        if bias is not None and bias.shape[1] == 1:
            bias = bias.expand(b, self.num_attention_heads, n, n)
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                                 dropout_p=self.dropout.p if self.training else 0.0, scale=scale)
        return (context.transpose(1, 2).reshape(b, n, self.all_head_size),)

    for module in model.modules():
        if isinstance(module, LayoutLMv3SelfAttention):
            module.forward = types.MethodType(forward, module)
    return model
