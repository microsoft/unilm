from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from transformers.modeling_bert import \
    BertPreTrainedModel, BertSelfOutput, BertIntermediate, \
    BertOutput, BertPredictionHeadTransform, BertPooler
from transformers.modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_distilbert import DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_xlm_roberta import XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_electra import ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.file_utils import WEIGHTS_NAME

from s2s_ft.config import BertForSeq2SeqConfig
from s2s_ft.convert_state_dict import get_checkpoint_from_transformer_cache, state_dict_convert

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm

UNILM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'unilm-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin",
    'unilm-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin",
    'unilm1-base-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin",
    'unilm1-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin",
    'unilm1.2-base-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased.bin", 
    'unilm2-base-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm2-base-uncased.bin", 
    'unilm2-large-uncased': "https://unilm.blob.core.windows.net/ckpt/unilm2-large-uncased.bin", 
    'unilm2-large-cased': "https://unilm.blob.core.windows.net/ckpt/unilm2-large-cased.bin", 
}

MINILM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'minilm-l12-h384-uncased': "https://unilm.blob.core.windows.net/ckpt/minilm-l12-h384-uncased.bin",
}

class BertPreTrainedForSeq2SeqModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertForSeq2SeqConfig
    supported_convert_pretrained_model_archive_map = {
        "bert": BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        "roberta": ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        "xlm-roberta": XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, 
        "unilm": UNILM_PRETRAINED_MODEL_ARCHIVE_MAP, 
        "minilm": MINILM_PRETRAINED_MODEL_ARCHIVE_MAP, 
    }
    base_model_prefix = "unilm_for_seq2seq"
    pretrained_model_archive_map = {
        **ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
        **XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, 
        **BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
        **UNILM_PRETRAINED_MODEL_ARCHIVE_MAP,
        **MINILM_PRETRAINED_MODEL_ARCHIVE_MAP, 
        **ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP, 
    }

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, reuse_position_embedding=None, *model_args, **kwargs):
        model_type = kwargs.pop('model_type', 'unilm')
        if model_type is not None and "state_dict" not in kwargs:
            if model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    state_dict = state_dict_convert[model_type](state_dict)
                    kwargs["state_dict"] = state_dict
                    logger.info("Load HF ckpts")
                elif os.path.isfile(pretrained_model_name_or_path):
                    state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                elif os.path.isdir(pretrained_model_name_or_path):
                    state_dict = torch.load(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME), map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                else:
                    raise RuntimeError("Not fined the pre-trained checkpoint !")

        if kwargs["state_dict"] is None:
            logger.info("s2s-ft does't support the model !")
            raise NotImplementedError()

        config = kwargs["config"]
        state_dict = kwargs["state_dict"]
        # initialize new position embeddings (From Microsoft/UniLM)
        _k = 'bert.embeddings.position_embeddings.weight'
        # if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
        #     logger.info("config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
        #         config.max_position_embeddings, state_dict[_k].shape[0]))
        #     if config.max_position_embeddings > state_dict[_k].shape[0]:
        #         old_size = state_dict[_k].shape[0]
        #         # state_dict[_k].data = state_dict[_k].data.resize_(config.max_position_embeddings, state_dict[_k].shape[1])
        #         state_dict[_k].resize_(
        #             config.max_position_embeddings, state_dict[_k].shape[1])
        #         start = old_size
        #         while start < config.max_position_embeddings:
        #             chunk_size = min(
        #                 old_size, config.max_position_embeddings - start)
        #             state_dict[_k].data[start:start+chunk_size,
        #                                 :].copy_(state_dict[_k].data[:chunk_size, :])
        #             start += chunk_size
        #     elif config.max_position_embeddings < state_dict[_k].shape[0]:
        #         state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " % (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        fix_word_embedding = getattr(config, "fix_word_embedding", None)
        if fix_word_embedding:
            self.word_embeddings.weight.requires_grad = False
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        if self.token_type_embeddings:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_ids


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def multi_head_attention(self, query, key, value, attention_mask, rel_pos):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        if rel_pos is not None:
            attention_scores = attention_scores + rel_pos

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)

    def forward(self, hidden_states, attention_mask=None, 
                encoder_hidden_states=None, 
                split_lengths=None, rel_pos=None):
        mixed_query_layer = self.query(hidden_states)
        if split_lengths:
            assert not self.output_attentions

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        if split_lengths:
            query_parts = torch.split(mixed_query_layer, split_lengths, dim=1)
            key_parts = torch.split(mixed_key_layer, split_lengths, dim=1)
            value_parts = torch.split(mixed_value_layer, split_lengths, dim=1)

            key = None
            value = None
            outputs = []
            sum_length = 0
            for (query, _key, _value, part_length) in zip(query_parts, key_parts, value_parts, split_lengths):
                key = _key if key is None else torch.cat((key, _key), dim=1)
                value = _value if value is None else torch.cat((value, _value), dim=1)
                sum_length += part_length
                outputs.append(self.multi_head_attention(
                    query, key, value, attention_mask[:, :, sum_length - part_length: sum_length, :sum_length], 
                    rel_pos=None if rel_pos is None else rel_pos[:, :, sum_length - part_length: sum_length, :sum_length], 
                )[0])
            outputs = (torch.cat(outputs, dim=1), )
        else:
            outputs = self.multi_head_attention(
                mixed_query_layer, mixed_key_layer, mixed_value_layer, 
                attention_mask, rel_pos=rel_pos)
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, 
                split_lengths=None, rel_pos=None):
        self_outputs = self.self(
            hidden_states, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states, 
            split_lengths=split_lengths, rel_pos=rel_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, split_lengths=None, rel_pos=None):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, 
            split_lengths=split_lengths, rel_pos=rel_pos)
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, split_lengths=None, rel_pos=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, 
                split_lengths=split_lengths, rel_pos=rel_pos)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance /
                                                    max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret


class BertModel(BertPreTrainedForSeq2SeqModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if not isinstance(config, BertForSeq2SeqConfig):
            self.pooler = BertPooler(config)
        else:
            self.pooler = None

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins, config.num_attention_heads, bias=False)
        else:
            self.rel_pos_bias = None

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None, split_lengths=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output, position_ids = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        if self.config.rel_pos_bins > 0:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(
                rel_pos_mat, num_buckets=self.config.rel_pos_bins, max_distance=self.config.max_rel_pos)
            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins).type_as(embedding_output)
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = None
        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask, 
            split_lengths=split_lengths, rel_pos=rel_pos)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output, ) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        if self.pooler is None:
            return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
        else:
            pooled_output = self.pooler(sequence_output)
            return sequence_output, pooled_output


class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.float().repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, decoder_weight):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_weight = decoder_weight

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = F.linear(hidden_states, weight=self.decoder_weight, bias=self.bias)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, decoder_weight):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, decoder_weight)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


def create_mask_and_position_ids(num_tokens, max_len, offset=None):
    base_position_matrix = torch.arange(
        0, max_len, dtype=num_tokens.dtype, device=num_tokens.device).view(1, -1)
    mask = (base_position_matrix < num_tokens.view(-1, 1)).type_as(num_tokens)
    if offset is not None:
        base_position_matrix = base_position_matrix + offset.view(-1, 1)
    position_ids = base_position_matrix * mask
    return mask, position_ids


class BertForSequenceToSequence(BertPreTrainedForSeq2SeqModel):
    MODEL_NAME = 'basic class'

    def __init__(self, config):
        super(BertForSequenceToSequence, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.init_weights()

        self.log_softmax = nn.LogSoftmax()

        self.source_type_id = config.source_type_id
        self.target_type_id = config.target_type_id

        if config.label_smoothing > 0:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
            self.crit_mask_lm = None
        else:
            self.crit_mask_lm_smoothed = None
            self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')


class BertForSequenceToSequenceWithPseudoMask(BertForSequenceToSequence):
    MODEL_NAME = "BertForSequenceToSequenceWithPseudoMask"

    @staticmethod
    def create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids):
        weight = torch.cat((torch.zeros_like(source_position_ids), target_span_ids, -target_span_ids), dim=1)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)

        true_tokens = (0 <= to_weight) & (torch.cat((source_mask, target_mask, target_mask), dim=1) == 1).unsqueeze(1)
        true_tokens_mask = (from_weight >= 0) & true_tokens & (to_weight <= from_weight)
        pseudo_tokens_mask = (from_weight < 0) & true_tokens & (-to_weight > from_weight)
        pseudo_tokens_mask = pseudo_tokens_mask | ((from_weight < 0) & (to_weight == from_weight))

        return (true_tokens_mask | pseudo_tokens_mask).type_as(source_mask)

    def forward(
            self, source_ids, target_ids, label_ids, pseudo_ids, 
            num_source_tokens, num_target_tokens, target_span_ids=None, target_no_offset=None):
        source_len = source_ids.size(1)
        target_len = target_ids.size(1)
        pseudo_len = pseudo_ids.size(1)
        assert target_len == pseudo_len
        assert source_len > 0 and target_len > 0
        split_lengths = (source_len, target_len, pseudo_len)

        input_ids = torch.cat((source_ids, target_ids, pseudo_ids), dim=1)

        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(target_ids) * self.target_type_id,
             torch.ones_like(pseudo_ids) * self.target_type_id), dim=1)

        source_mask, source_position_ids = \
            create_mask_and_position_ids(num_source_tokens, source_len)
        target_mask, target_position_ids = \
            create_mask_and_position_ids(
                num_target_tokens, target_len, offset=None if target_no_offset else num_source_tokens)

        position_ids = torch.cat((source_position_ids, target_position_ids, target_position_ids), dim=1)
        if target_span_ids is None:
            target_span_ids = target_position_ids
        attention_mask = self.create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids)

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, split_lengths=split_lengths)

        sequence_output = outputs[0]
        pseudo_sequence_output = sequence_output[:, source_len + target_len:, ]

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        prediction_scores_masked = self.cls(pseudo_sequence_output)

        if self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), label_ids)
        else:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), label_ids)
        pseudo_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), target_mask)

        return pseudo_lm_loss


class BertForSequenceToSequenceUniLMV1(BertForSequenceToSequence):
    MODEL_NAME = "BertForSequenceToSequenceUniLMV1"

    @staticmethod
    def create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids):
        weight = torch.cat((torch.zeros_like(source_position_ids), target_span_ids), dim=1)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)

        true_tokens = torch.cat((source_mask, target_mask), dim=1).unsqueeze(1)
        return ((true_tokens == 1) & (to_weight <= from_weight)).type_as(source_mask)

    def forward(self, source_ids, target_ids, masked_ids, masked_pos, masked_weight, num_source_tokens, num_target_tokens):
        source_len = source_ids.size(1)
        target_len = target_ids.size(1)
        split_lengths = (source_len, target_len)

        input_ids = torch.cat((source_ids, target_ids), dim=1)

        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(target_ids) * self.target_type_id), dim=1)

        source_mask, source_position_ids = \
            create_mask_and_position_ids(num_source_tokens, source_len)
        target_mask, target_position_ids = \
            create_mask_and_position_ids(
                num_target_tokens, target_len, offset=num_source_tokens)

        position_ids = torch.cat((source_position_ids, target_position_ids), dim=1)
        attention_mask = self.create_attention_mask(
            source_mask, target_mask, source_position_ids, target_position_ids)

        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
            position_ids=position_ids, split_lengths=split_lengths)

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))
        
        sequence_output = outputs[0]
        target_sequence_output = sequence_output[:, source_len:, ]
        masked_sequence_output = gather_seq_out_by_pos(target_sequence_output, masked_pos)

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        prediction_scores_masked = self.cls(masked_sequence_output)

        if self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_ids)
        else:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_ids)
        pseudo_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weight)

        return pseudo_lm_loss


class UniLMForSequenceClassification(BertPreTrainedForSeq2SeqModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
