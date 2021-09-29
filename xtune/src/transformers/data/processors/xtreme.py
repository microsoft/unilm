# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" GLUE processors and helpers """

import logging
import os
import random

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def xtreme_convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        word_dropout_rate=0.0,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = xtreme_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = xtreme_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, word_dropout_rate=word_dropout_rate,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("text a: %s" % (example.text_a))
            logger.info("text b: %s" % (example.text_b))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label,
                guid=example.guid
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class PawsxProcessor(DataProcessor):
    """Processor for the PAWS-X data set (XTREME version)."""

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train-en.tsv")), "train")

    def get_translate_train_examples(self, data_dir):
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, "translate-train/en-{}-translated.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("translate", i)
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_valid_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev-{}.tsv".format(self.language))), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test-{}.tsv".format(self.language))),
                                     "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_translate_train_dict(self, data_dir, tgt2src_dict, tgt2src_cnt):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language

        lines = self._read_tsv(os.path.join(data_dir, "translate-train/en-{}-translated.tsv".format(lg)))
        dict = {}
        cnt = {}
        for (i, line) in enumerate(lines):
            text_a = line[0].strip()
            text_b = line[1].strip()
            translated_text_a = line[2].strip()
            translated_text_b = line[3].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and \
                   isinstance(translated_text_a, str) and isinstance(translated_text_b, str)

            if text_a not in cnt:
                cnt[text_a] = 0
            cnt[text_a] += 1

            if text_b not in cnt:
                cnt[text_b] = 0
            cnt[text_b] += 1

            if text_a not in dict or random.random() <= 1.0 / cnt[text_a]:
                dict[text_a] = translated_text_a
            if text_b not in dict or random.random() <= 1.0 / cnt[text_b]:
                dict[text_b] = translated_text_b

            if translated_text_a not in tgt2src_cnt:
                tgt2src_cnt[translated_text_a] = 0
            tgt2src_cnt[translated_text_a] += 1

            if translated_text_b not in tgt2src_cnt:
                tgt2src_cnt[translated_text_b] = 0
            tgt2src_cnt[translated_text_b] += 1

            if translated_text_a not in tgt2src_dict or random.random() <= 1.0 / tgt2src_cnt[translated_text_a]:
                tgt2src_dict[translated_text_a] = text_a
            if translated_text_b not in tgt2src_dict or random.random() <= 1.0 / tgt2src_cnt[translated_text_b]:
                tgt2src_dict[translated_text_b] = text_b

        return dict


class XnliProcessor(DataProcessor):
    """Processor for the XNLI data set (XTREME version)."""

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language

        lines = self._read_tsv(os.path.join(data_dir, "train-{}.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[2] == "contradictory" else line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_translate_train_examples(self, data_dir):
        lg = self.language if self.train_language is None else self.train_language
        lines = self._read_tsv(os.path.join(data_dir, "translate-train/en-{}-translated.tsv".format(lg)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = "contradiction" if line[4] == "contradictory" else line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_translate_train_dict(self, data_dir, tgt2src_dict, tgt2src_cnt):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language

        lines = self._read_tsv(os.path.join(data_dir, "translate-train/en-{}-translated.tsv".format(lg)))
        dict = {}
        cnt = {}
        for (i, line) in enumerate(lines):
            text_a = line[0].strip()
            text_b = line[1].strip()
            translated_text_a = line[2].strip()
            translated_text_b = line[3].strip()
            assert isinstance(text_a, str) and isinstance(text_b, str) and \
                   isinstance(translated_text_a, str) and isinstance(translated_text_b, str)

            if text_a not in cnt:
                cnt[text_a] = 0
            cnt[text_a] += 1

            if text_b not in cnt:
                cnt[text_b] = 0
            cnt[text_b] += 1

            if text_a not in dict or random.random() <= 1.0 / cnt[text_a]:
                dict[text_a] = translated_text_a
            if text_b not in dict or random.random() <= 1.0 / cnt[text_b]:
                dict[text_b] = translated_text_b

            if translated_text_a not in tgt2src_cnt:
                tgt2src_cnt[translated_text_a] = 0
            tgt2src_cnt[translated_text_a] += 1

            if translated_text_b not in tgt2src_cnt:
                tgt2src_cnt[translated_text_b] = 0
            tgt2src_cnt[translated_text_b] += 1

            if translated_text_a not in tgt2src_dict or random.random() <= 1.0 / tgt2src_cnt[translated_text_a]:
                tgt2src_dict[translated_text_a] = text_a
            if translated_text_b not in tgt2src_dict or random.random() <= 1.0 / tgt2src_cnt[translated_text_b]:
                tgt2src_dict[translated_text_b] = text_b

        return dict

    def get_valid_examples(self, data_dir):
        """See base class."""
        return self.get_test_valid_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        return self.get_test_valid_examples(data_dir, "test")

    def get_test_valid_examples(self, data_dir, split):
        assert split in ["test", "dev"]

        lines = self._read_tsv(os.path.join(data_dir, "{}-{}.tsv".format(split, self.language)))
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (split, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


xtreme_tasks_num_labels = {
    "xnli": 3,
    "pawsx": 2,
}

xtreme_processors = {
    "xnli": XnliProcessor,
    "pawsx": PawsxProcessor,
}

xtreme_output_modes = {
    "xnli": "classification",
    "pawsx": "classification",
}
