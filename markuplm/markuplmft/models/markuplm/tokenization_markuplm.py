# coding=utf-8
# Copyright 2018 The Microsoft Research Asia MarkupLM Team Authors.
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
""" Tokenization class for model MarkupLM."""

from transformers.utils import logging
from transformers import RobertaTokenizer

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/vocab.json",
        "markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/vocab.json",
    },
    "merges_file": {
        "markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/merges.txt",
        "markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/merges.txt",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "markuplm-base": 512,
    "markuplm-large": 512,
}


class MarkupLMTokenizer(RobertaTokenizer):
    r"""
    Constructs a MarkupLM tokenizer.

    :class:`~transformers.LayoutLMTokenizer is identical to :class:`~transformers.RobertaTokenizer` and runs end-to-end
    tokenization.

    Refer to superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]
