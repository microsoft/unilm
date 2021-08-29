from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from transformers import BertConfig, RobertaConfig
from s2s_ft.configuration_unilm import UnilmConfig
# from s2s_ft.modeling import LayoutlmConfig

logger = logging.getLogger(__name__)


class BertForSeq2SeqConfig(BertConfig):
    def __init__(self, label_smoothing=0.1, source_type_id=0, target_type_id=1, **kwargs):
        super(BertForSeq2SeqConfig, self).__init__(**kwargs)
        self.label_smoothing = label_smoothing
        self.source_type_id = source_type_id
        self.target_type_id = target_type_id

    @classmethod
    def from_exist_config(cls, config, label_smoothing=0.1, max_position_embeddings=None, max_source_length=None,
                          base_model_type='bert', layoutlm_only_layout_flag=False):
        required_keys = [
            "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "hidden_act", "intermediate_size", "hidden_dropout_prob", "attention_probs_dropout_prob",
            "max_position_embeddings", "type_vocab_size", "initializer_range", "layer_norm_eps"]

        kwargs = {}
        for key in required_keys:
            assert hasattr(config, key)
            kwargs[key] = getattr(config, key)

        kwargs["vocab_size_or_config_json_file"] = kwargs["vocab_size"]
        if isinstance(config, RobertaConfig):
            kwargs["type_vocab_size"] = 0
            kwargs["max_position_embeddings"] = kwargs["max_position_embeddings"] - 2
        
        additional_keys = [
            "source_type_id", "target_type_id"
        ]
        for key in additional_keys:
            if hasattr(config, key):
                kwargs[key] = getattr(config, key)

        # if isinstance(config, LayoutlmConfig):
        if hasattr(config, 'max_2d_position_embeddings'):
            layoutlm_special_keys = ['max_2d_position_embeddings',]
            for key in layoutlm_special_keys:
                kwargs[key] = getattr(config, key)

        kwargs['base_model_type'] = base_model_type
        kwargs['layoutlm_only_layout'] = layoutlm_only_layout_flag

        if max_position_embeddings is not None and max_position_embeddings > config.max_position_embeddings:
            kwargs["max_position_embeddings"] = max_position_embeddings
            logger.info("  **  Change max position embeddings to %d  ** " % max_position_embeddings)

        if max_source_length is not None:
            kwargs['max_source_length'] = max_source_length

        return cls(label_smoothing=label_smoothing, **kwargs)
