from transformers import CONFIG_MAPPING, MODEL_FOR_QUESTION_ANSWERING_MAPPING, \
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter
from transformers.file_utils import PRESET_MIRROR_DICT

from .models.markuplm import (
    MarkupLMConfig,
    MarkupLMTokenizer,
    MarkupLMForQuestionAnswering,
    MarkupLMForTokenClassification,
    MarkupLMTokenizerFast,
)

CONFIG_MAPPING.update(
    [
        ("markuplm", MarkupLMConfig),
    ]
)
MODEL_NAMES_MAPPING.update([("markuplm", "MarkupLM")])

TOKENIZER_MAPPING.update(
    [
        (MarkupLMConfig, (MarkupLMTokenizer, MarkupLMTokenizerFast)),
    ]
)

SLOW_TO_FAST_CONVERTERS.update(
    {"MarkupLMTokenizer": RobertaConverter}
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING.update(
    [(MarkupLMConfig, MarkupLMForQuestionAnswering)]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(MarkupLMConfig, MarkupLMForTokenClassification)]
)
