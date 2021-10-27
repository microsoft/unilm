from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from transformers import BertConfig, RobertaConfig
from s2s_ft.configuration_unilm import UnilmConfig

logger = logging.getLogger(__name__)


class BertForSeq2SeqConfig(BertConfig):
    def __init__(self, label_smoothing=0.1, source_type_id=0, target_type_id=1, 
                 rel_pos_bins=0, max_rel_pos=0, fix_word_embedding=False, **kwargs):
        super(BertForSeq2SeqConfig, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fix_word_embedding = fix_word_embedding

    @classmethod
    def from_exist_config(cls, config, label_smoothing=0.1, max_position_embeddings=None, fix_word_embedding=False):
        required_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "hidden_act", "intermediate_size", "hidden_dropout_prob", "attention_probs_dropout_prob",
            "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps", 
            ]

        kwargs = {}
        for key in required_keys:
            assert hasattr(config, key)
            kwargs[key] = getattr(config, key)

        kwargs["vocab_size_or_config_json_file"] = kwargs["vocab_size"]
        if isinstance(config, RobertaConfig):
            kwargs["type_vocab_size"] = 0
            kwargs["max_position_embeddings"] = kwargs["max_position_embeddings"] - 2
        
        additional_keys = [
            "source_type_id", "target_type_id", "rel_pos_bins", "max_rel_pos", 
        ]
        for key in additional_keys:
            if hasattr(config, key):
                kwargs[key] = getattr(config, key)

        if max_position_embeddings is not None and max_position_embeddings > config.max_position_embeddings:
            kwargs["max_position_embeddings"] = max_position_embeddings
            logger.info("  **  Change max position embeddings to %d  ** " % max_position_embeddings)

        return cls(label_smoothing=label_smoothing, fix_word_embedding=fix_word_embedding, **kwargs)
