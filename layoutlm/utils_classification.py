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

import random
import numpy as np
import torch
import re
import os
from tqdm import tqdm
from lxml import html
import copy
import json
from transformers import DataProcessor, is_tf_available
import xml.etree.ElementTree as ET
import logging
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

logger = logging.getLogger(__name__)


def get_text(node):
    textnodes = node.xpath(".//text()")
    s = "".join([text for text in textnodes])
    return re.sub(r"\s+", " ", s).strip()


def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


class DocExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, bbox=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.bbox = bbox
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class CdipProcessor(DataProcessor):
    """Processor for the CDIP data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return DocExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "labels", "train.txt")) as f:
            lines = []
            for line in tqdm(f.readlines(), desc="Gettting train examples"):
                file, label = line.split()
                text, bbox = self.read_hocr_file(data_dir, file)
                lines.append([text, bbox, label])
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "labels", "val.txt")) as f:
            lines = []
            for line in tqdm(f.readlines(), desc="Gettting dev examples"):
                file, label = line.split()
                text, bbox = self.read_hocr_file(data_dir, file)
                lines.append([text, bbox, label])
        return self._create_examples(lines, "val")

    def read_hocr_file(
        self, data_dir, file,
    ):
        hocr_file = os.path.join(data_dir, "images", file[:-4] + ".html")
        text_buffer = []
        bbox_buffer = []
        doc = html.parse(hocr_file)
        for page in doc.xpath("//*[@class='ocr_page']"):
            page_bbox = [int(x) for x in get_prop(page, "bbox").split()]
            width, height = page_bbox[2], page_bbox[3]
            for word in doc.xpath("//*[@class='ocrx_word']"):
                textnodes = word.xpath(".//text()")
                s = "".join([text for text in textnodes])
                text = re.sub(r"\s+", " ", s).strip()
                if text:
                    text_buffer.append(text)
                    bbox = [int(x) for x in get_prop(word, "bbox").split()]
                    bbox = [
                        bbox[0] / width,
                        bbox[1] / height,
                        bbox[2] / width,
                        bbox[3] / height,
                    ]
                    bbox = [int(x * 1000) for x in bbox]
                    bbox_buffer.append(bbox)
        return text_buffer, bbox_buffer

    def get_labels(self):
        """See base class."""
        return [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
        ]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line[0]
            bbox = line[1]
            label = line[2]
            examples.append(
                DocExample(guid=guid, text_a=text, text_b=None, bbox=bbox, label=label)
            )
        return examples


processors["cdip"] = CdipProcessor
output_modes["cdip"] = "classification"


class DocFeature(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, bboxes, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.bboxes = bboxes
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
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

    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens = []
        bboxes = []

        bboxes.append([0, 0, 0, 0])

        if len(example.text_a) == 0:
            bboxes.append([0, 0, 0, 0])
            tokens.append(pad_token)

        for token, bbox in zip(example.text_a, example.bbox):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                bboxes.append(bbox)
                tokens.append(sub_token)

        bboxes = bboxes[: max_length - 1]  # added cls but did not add sep
        bboxes.append([1000, 1000, 1000, 1000])

        inputs = tokenizer.encode_plus(
            tokens, None, add_special_tokens=True, max_length=max_length
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            bboxes = ([[0, 0, 0, 0]] * padding_length) + bboxes
            attention_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            bboxes = bboxes + ([[0, 0, 0, 0]] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert len(bboxes) == max_length, "Error with input length {} vs {}".format(
            len(bboxes), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
            len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in bboxes]))
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
            )
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            DocFeature(
                input_ids=input_ids,
                bboxes=bboxes,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )
    return features


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cdip":
        print(preds)
        print(labels)
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
