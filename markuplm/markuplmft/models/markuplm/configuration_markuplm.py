# coding=utf-8
# Copyright 2010, The Microsoft Research Asia MarkupLM Team authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" MarkupLM model configuration """

from transformers.utils import logging
from transformers.models.roberta.configuration_roberta import RobertaConfig

logger = logging.get_logger(__name__)

MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/config.json",
    "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/config.json",
}


class MarkupLMConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.MarkupLMModel`. It is used to
    instantiate a MarkupLM model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MarkupLM `markuplm-base-uncased
    <https://huggingface.co/microsoft/markuplm-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.BertConfig` and can be used to control the model outputs.
    Read the documentation from :class:`~transformers.BertConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the MarkupLM model. Defines the different tokens that can be represented by the
            `inputs_ids` passed to the forward method of :class:`~transformers.MarkupLMModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed into :class:`~transformers.MarkupLMModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.

        max_tree_id_unit_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum value that the tree id unit embedding might ever used.
            Typically set this to something large just in case (e.g., 1024).

        max_xpath_tag_unit_embeddings (:obj:`int`, `optional`, defaults to 256):
            The maximum value that the xpath tag unit embedding might ever used.
            Typically set this to something large just in case (e.g., 256).

        max_xpath_subs_unit_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum value that the xpath subscript unit embedding might ever used.
            Typically set this to something large just in case (e.g., 1024).

        tree_id_unit_hidden_size (:obj:`int`, `optional`, defaults to 32):
            The hidden size of each tree id unit, since we set max_depth as 50,
            then we weill have (50*tree_id_unit_hidden_size)-dim for one
            complete tree index.

        xpath_tag_unit_hidden_size (:obj:`int`, `optional`, defaults to 32):
            The hidden size of each tree id unit, since we set max_depth as 50,
            then we weill have (50*xpath_tag_unit_hidden_size)-dim for one
            complete tree index.

        xpath_subs_unit_hidden_size (:obj:`int`, `optional`, defaults to 32):
            The hidden size of each tree id unit, since we set max_depth as 50,
            then we weill have (50*xpath_subs_unit_hidden_size)-dim for one
            complete tree index.

        max_depth : maximum depth in xpath

    Examples::

        no examples~

    """
    model_type = "markuplm"

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            gradient_checkpointing=False,

            max_xpath_tag_unit_embeddings=256,
            max_xpath_subs_unit_embeddings=1024,
            xpath_unit_hidden_size=32,
            max_depth=50,
            **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            gradient_checkpointing=gradient_checkpointing,
            **kwargs,
        )
        self.max_depth = max_depth
        self.max_xpath_tag_unit_embeddings = max_xpath_tag_unit_embeddings
        self.max_xpath_subs_unit_embeddings = max_xpath_subs_unit_embeddings
        self.xpath_unit_hidden_size = xpath_unit_hidden_size