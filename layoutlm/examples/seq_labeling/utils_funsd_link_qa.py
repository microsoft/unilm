from __future__ import absolute_import, division, print_function
import json
import logging
import math
import collections
from io import open
from tqdm import tqdm
import pdb
from transformers.tokenization_bert import BasicTokenizer, whitespace_tokenize
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
""" Load Funsd dataset and frame link as QA. """
"""
squad data structure
entry
    paragraphs - list
        context - string of doc
        qas - list of dict
            answers - list
                answer_start # char position
                text # answer text
            question - str
            id  - unique str
    title - str
"""
# from __future__ import absolute_import, division, print_function

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
# from utils_squad_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores

logger = logging.getLogger(__name__)


class FunsdLinkExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 boxes,
                 actual_bboxes,
                 file_name,
                 page_size,
                 question_start_position,
                 question_end_position,
                 orig_answer_text=None,
                 answer_start_position=None,
                 answer_end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.answer_start_position = answer_start_position
        self.answer_end_position = answer_end_position
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.is_impossible = is_impossible
        self.file_name = file_name
        self.page_size = page_size
        self.question_start_position = question_start_position
        self.question_end_position = question_end_position
        self.answers = [{"text": orig_answer_text}]

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.answer_start_position:
            s += ", start_position: %d" % (self.answer_start_position)
        if self.answer_end_position:
            s += ", end_position: %d" % (self.answer_end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 boxes,
                 actual_bboxes,
                 file_name,
                 page_size,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


def read_funsd_link_examples(input_file,  is_training, version_2_with_negative):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)
    examples = []
    for fn, entry in input_data.items():
        doc_tokens, bboxes_str, actual_bboxes_str, qas = entry['doc_tokens'], entry[
            'bboxes'], entry['actual_bboxes'], entry['qas']
        boxes, actual_bboxes = [], []
        for bbox in bboxes_str:
            box = [int(b) for b in bbox.split()]
            boxes.append(box)
        for abbox_str in actual_bboxes_str:
            abox, ps = abbox_str.split('\t')
            box = [int(b) for b in abox.split()]
            actual_bboxes.append(box)
            page_size = [int(i) for i in ps.split()]

        if len(qas) == 0:
            print('no valid qa pair')
            continue
        for qa in qas:
            example = FunsdLinkExample(
                qas_id=qa['uid'],
                question_text=qa['question_text'],
                doc_tokens=doc_tokens,
                boxes=boxes,
                actual_bboxes=actual_bboxes,
                file_name=fn,
                page_size=page_size,
                question_start_position=qa['question_start_position'],
                question_end_position=qa['question_end_position'],
                orig_answer_text=qa['orig_answer_text'],
                answer_start_position=qa['answer_start_position'],
                answer_end_position=qa['answer_end_position'],
                is_impossible=False)
            examples.append(example)
    return examples


def read_funsd_link_examples2(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of FunsdLinkExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in tqdm(input_data):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset
                                                           + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(
                            doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                # start_position and end_position are 0-based token position
                example = FunsdLinkExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                    question_start_position=question_start_position,
                    question_end_position=question_end_position,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)
    return examples


def convert_examples_to_features_simple(examples, tokenizer, max_seq_length,
                                        doc_stride, max_query_length, is_training,
                                        cls_token_at_end=False,
                                        cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                        cls_token_box=[0, 0, 0, 0], sep_token_box=[1000, 1000, 1000, 1000],
                                        pad_token_box=[0, 0, 0, 0],
                                        sequence_a_segment_id=0, sequence_b_segment_id=1,
                                        cls_token_segment_id=0, pad_token_segment_id=0,
                                        mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s.
    [CLS] Question [SEP] Context [SEP]
    """

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    for (example_index, example) in enumerate(examples):
        doc_tokens = example.doc_tokens
        question_start_position = example.question_start_position
        question_end_position = example.question_end_position
        answer_start_position = example.answer_start_position
        answer_end_position = example.answer_end_position
        boxes = example.boxes
        actual_bboxes = example.actual_bboxes
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size
        tokens = []
        segment_ids = []
        # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
        # Original TF implem also keep the classification token (set to 0) (not sure why...)
        p_mask = []
        token_boxes = []
        actual_token_bboxes = []
        # query sentence cap
        len_query = question_end_position - question_start_position + 1
        if len_query > max_query_length:
            question_end_position = question_end_position - \
                (len_query - max_query_length)
        query_tokens = doc_tokens[question_start_position: question_end_position + 1]
        query_boxes = boxes[question_start_position: question_end_position + 1]
        query_actual_boxes = actual_bboxes[question_start_position: question_end_position + 1]
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # cap max tokens
        if (len(doc_tokens) > max_tokens_for_doc):
            doc_tokens = doc_tokens[:max_tokens_for_doc]
            boxes = boxes[:max_tokens_for_doc]
            actual_bboxes = actual_bboxes[:max_tokens_for_doc]
        
        # CLS token at the beginning
        if not cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            token_boxes.append(cls_token_box)
            actual_token_bboxes.append([0, 0, width, height])
            p_mask.append(0)
            cls_index = 0

        # Query
        for i in range(len(query_tokens)):
            tokens.append(query_tokens[i])
            token_boxes.append(query_boxes[i])
            actual_token_bboxes.append(query_actual_boxes[i])
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_a_segment_id)
        token_boxes.append(sep_token_box)
        actual_token_bboxes.append([0, 0, width, height])
        p_mask.append(1)
        # Paragraph
        for i in range(len(doc_tokens)):
            tokens.append(doc_tokens[i])
            segment_ids.append(sequence_b_segment_id)
            token_boxes.append(boxes[i])
            actual_token_bboxes.append(actual_bboxes[i])
            p_mask.append(0)
        paragraph_len = max_tokens_for_doc

        # SEP token
        tokens.append(sep_token)
        segment_ids.append(sequence_b_segment_id)
        token_boxes.append(sep_token_box)
        actual_token_bboxes.append([0, 0, width, height])
        p_mask.append(1)

        # CLS token at the end
        if cls_token_at_end:
            tokens.append(cls_token)
            segment_ids.append(cls_token_segment_id)
            token_boxes.append(cls_token_box)
            actual_token_bboxes.append([0, 0, width, height])
            p_mask.append(0)
            cls_index = len(tokens) - 1  # Index of classification token
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(pad_token)
            input_mask.append(0 if mask_padding_with_zero else 1)
            token_boxes.append(cls_token_box)
            actual_token_bboxes.append(pad_token_box)
            segment_ids.append(pad_token_segment_id)
            p_mask.append(1)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length
        assert len(actual_token_bboxes) == max_seq_length

        span_is_impossible = example.is_impossible
        start_position = None
        end_position = None
        if answer_end_position > max_tokens_for_doc:
            start_position = 0
            end_position = 0
            span_is_impossible = True
        else:
            doc_offset = len(query_tokens) + 2
            start_position = answer_start_position + doc_offset
            end_position = answer_end_position + doc_offset
            span_is_impossible = False

        if is_training and span_is_impossible:
            start_position = cls_index
            end_position = cls_index

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.qas_id)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join(
                [str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join(
                [str(x) for x in segment_ids]))
            logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))
            logger.info("actual_bboxes: %s", " ".join(
                [str(x) for x in actual_token_bboxes]))
        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                doc_span_index=None,
                tokens=tokens,
                token_to_orig_map={},
                token_is_max_context={},
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                cls_index=cls_index,
                p_mask=p_mask,
                paragraph_len=paragraph_len,
                boxes=token_boxes,
                actual_bboxes=actual_token_bboxes,
                file_name=file_name,
                page_size=page_size,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible))
        unique_id += 1
    return features


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 cls_token_box=[0, 0, 0, 0], sep_token_box=[1000, 1000, 1000, 1000],
                                 pad_token_box=[0, 0, 0, 0],
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s.
    [CLS] Question [SEP] Context [SEP]
    """

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    features = []
    for (example_index, example) in enumerate(examples):
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size
        # if example_index % 100 == 0:
        #     logger.info('Converting %s/%s pos %s neg %s', example_index, len(examples), cnt_pos, cnt_neg)

        query_tokens = tokenizer.tokenize(example.question_text)
        pdb.set_trace()
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)
        pdb.set_trace()
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)
            pdb.set_trace()
            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(
                    tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" %
                            " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(
                        tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            pdb.set_trace()
            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
                                           ["unique_id", "start_top_log_probs", "start_top_index",
                                            "end_top_log_probs", "end_top_index", "cls_logits"])
