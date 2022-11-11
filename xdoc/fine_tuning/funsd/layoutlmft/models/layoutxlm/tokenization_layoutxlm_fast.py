# coding=utf-8
from transformers import XLMRobertaTokenizerFast
from transformers.file_utils import is_sentencepiece_available
from transformers.utils import logging


if is_sentencepiece_available():
    from .tokenization_layoutxlm import LayoutXLMTokenizer
else:
    LayoutXLMTokenizer = None


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "layoutxlm-base": "https://huggingface.co/layoutxlm-base/resolve/main/sentencepiece.bpe.model",
        "layoutxlm-large": "https://huggingface.co/layoutxlm-large/resolve/main/sentencepiece.bpe.model",
    },
    "tokenizer_file": {
        "layoutxlm-base": "https://huggingface.co/layoutxlm-base/resolve/main/tokenizer.json",
        "layoutxlm-large": "https://huggingface.co/layoutxlm-large/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "layoutxlm-base": 512,
    "layoutxlm-large": 512,
}


class LayoutXLMTokenizerFast(XLMRobertaTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = LayoutXLMTokenizer

    def __init__(self, model_max_length=512, **kwargs):
        super().__init__(model_max_length=model_max_length, **kwargs)
