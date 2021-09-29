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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

import argparse
import glob
import logging
import os
import random
import timeit
import itertools
import json
import copy
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    BertConfig,
    BertForQuestionAnswering,
    BertTokenizer,
    XLMRobertaConfig,
    XLMRobertaForQuestionAnsweringStable,
    XLMRobertaTokenizer,
    CamembertConfig,
    CamembertForQuestionAnswering,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForQuestionAnswering,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    XLMConfig,
    XLMForQuestionAnswering,
    XLMTokenizer,
    XLNetConfig,
    XLNetForQuestionAnswering,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
)

from transformers.data.metrics.evaluate_mlqa import evaluate_with_path as mlqa_evaluate_with_path
from transformers.data.metrics.evaluate_squad import evaluate_with_path as squad_evaluate_with_path
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, MLQAProcessor, \
    TyDiQAProcessor, XQuADProcessor
from transformers.tokenization_bert import whitespace_tokenize
from transformers.data.processors.squad import _improve_answer_span, _new_check_is_max_context, SquadFeatures

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, CamembertConfig, RobertaConfig, XLNetConfig, XLMConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
    "camembert": (CamembertConfig, CamembertForQuestionAnswering, CamembertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForQuestionAnsweringStable, XLMRobertaTokenizer),
}


class NoisedDataGenerator(object):
    def __init__(self,
                 task_name="mlqa",
                 r1_lambda=5.0,
                 enable_r1_loss=False,
                 original_loss=True,
                 noised_loss=False,
                 keep_boundary_unchanged=False,
                 r1_on_boundary_only=False,
                 noised_max_seq_length=512,
                 max_seq_length=512,
                 doc_stride=128,
                 max_query_length=64,
                 overall_ratio=1.0,
                 enable_bpe_switch=False,
                 bpe_switch_ratio=0.5,
                 tokenizer_dir=None,
                 do_lower_case=False,
                 tokenizer_languages=None,
                 enable_bpe_sampling=False,
                 bpe_sampling_ratio=0.5,
                 tokenizer=None,
                 sampling_alpha=0.3,
                 sampling_nbest_size=-1,
                 enable_random_noise=False,
                 noise_detach_embeds=False,
                 noise_eps=1e-5,
                 noise_type='uniform',
                 enable_code_switch=False,
                 code_switch_ratio=0.5,
                 dict_dir=None,
                 dict_languages=None,
                 translation_path=None,
                 disable_translate_labels=False,
                 translate_languages=None,
                 enable_data_augmentation=False,
                 augment_ratio=0.0,
                 augment_method=None,
                 r2_lambda=1.0,
                 use_hard_labels=False):
        if enable_code_switch:
            assert dict_dir is not None
            assert dict_languages is not None
        assert tokenizer is not None
        if enable_random_noise:
            assert noise_type in ['uniform', 'normal']

        self.task_name = task_name.lower()
        self.n_tokens = 0
        self.n_cs_tokens = 0
        self.r1_lambda = r1_lambda
        self.original_loss = original_loss
        self.noised_loss = noised_loss
        self.enable_r1_loss = enable_r1_loss
        self.keep_boundary_unchanged = keep_boundary_unchanged
        self.r1_on_boundary_only = r1_on_boundary_only
        self.max_seq_length = max_seq_length
        self.noised_max_seq_length = noised_max_seq_length
        self.doc_stride = doc_stride
        self.max_query_length = max_query_length
        self.overall_ratio = overall_ratio

        self.enable_bpe_switch = enable_bpe_switch
        self.bpe_switch_ratio = bpe_switch_ratio / self.overall_ratio
        assert not self.enable_bpe_switch or self.bpe_switch_ratio <= 1.0
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer_languages = tokenizer_languages

        self.enable_bpe_sampling = enable_bpe_sampling
        self.bpe_sampling_ratio = bpe_sampling_ratio / self.overall_ratio
        assert not self.enable_bpe_sampling or self.bpe_sampling_ratio <= 1.0
        self.tokenizer = tokenizer
        self.sampling_alpha = sampling_alpha
        self.sampling_nbest_size = sampling_nbest_size
        self.enable_random_noise = enable_random_noise
        self.noise_detach_embeds = noise_detach_embeds
        self.noise_eps = noise_eps
        self.noise_type = noise_type

        self.enable_code_switch = enable_code_switch
        self.code_switch_ratio = code_switch_ratio / self.overall_ratio
        assert not self.enable_code_switch or self.code_switch_ratio <= 1.0
        self.dict_dir = dict_dir
        self.dict_languages = dict_languages
        self.lang2dict = {}
        for lang in copy.deepcopy(dict_languages):
            dict_path = os.path.join(self.dict_dir, "en-{}.txt".format(lang))
            if not os.path.exists(dict_path):
                logger.info("dictionary en-{} doesn't exist.".format(lang))
                self.dict_languages.remove(lang)
                continue
            logger.info("reading dictionary from {}".format(dict_path))
            assert os.path.exists(dict_path)
            with open(dict_path, "r", encoding="utf-8") as reader:
                raw = reader.readlines()
            self.lang2dict[lang] = {}
            for line in raw:
                line = line.strip()
                try:
                    src, tgt = line.split("\t")
                except:
                    src, tgt = line.split(" ")
                if src not in self.lang2dict[lang]:
                    self.lang2dict[lang][src] = [tgt]
                else:
                    self.lang2dict[lang][src].append(tgt)

        self.lang2tokenizer = {}
        for lang in tokenizer_languages:
            self.lang2tokenizer[lang] = XLMRobertaTokenizer.from_pretrained(
                os.path.join(tokenizer_dir, "{}".format(lang)), do_lower_case=do_lower_case)

        self.translation_path = translation_path
        self.disable_translate_labels = disable_translate_labels
        self.translate_languages = translate_languages
        self.enable_data_augmentation = enable_data_augmentation
        self.augment_ratio = augment_ratio
        self.augment_method = augment_method
        self.r2_lambda = r2_lambda
        self.use_hard_labels = use_hard_labels
        self.id2ex = None
        if self.enable_data_augmentation and self.augment_method == "mt":
            # drop_languages = ["en", "zh-CN", "zh", "ja", "ko", "th", "my", "ml", "ta"]
            drop_languages = ["en"]
            for lang in drop_languages:
                if lang in self.translate_languages:
                    self.translate_languages.remove(lang)
            self.id2ex = {}
            for lang in self.translate_languages:
                if self.task_name == "tydiqa":
                    file_name = "tydiqa.translate.train.en-{}.json".format(lang)
                else:
                    file_name = "squad.translate.train.en-{}.json".format(lang)
                logger.info("Reading translation from {}".format(os.path.join(self.translation_path, file_name)))
                processor = MLQAProcessor()
                examples = processor.get_train_examples(self.translation_path,
                                                        file_name)
                for ex in examples:
                    if ex.qas_id not in self.id2ex:
                        self.id2ex[ex.qas_id] = []
                    if self.disable_translate_labels:
                        ex.is_impossible = True
                    self.id2ex[ex.qas_id].append(ex)

    def augment_examples(self, examples):
        n_augment = math.ceil(len(examples) * self.augment_ratio)

        augment_examples = []

        while n_augment > 0:
            examples = copy.deepcopy(examples)
            augment_examples += examples[:n_augment]
            n_augment -= len(examples[:n_augment])
            random.shuffle(examples)

        return augment_examples

    def get_translate_data(self, examples):
        translate_examples = []
        n_unfound = 0

        qas_ids = list(self.id2ex.keys())
        for ex_idx, example in enumerate(examples):
            qas_id = example.qas_id
            if self.task_name == "tydiqa" or qas_id not in self.id2ex:
                rand_qas_id = qas_ids[random.randint(0, len(qas_ids) - 1)]
                # logger.info(
                #     "qas_id {} is not found in translate data, using {} as replacement.".format(qas_id, rand_qas_id))
                n_unfound += 1
                qas_id = rand_qas_id

            idx = random.randint(0, len(self.id2ex[qas_id]) - 1)
            tgt_ex = self.id2ex[qas_id][idx]
            translate_examples.append(tgt_ex)

        logger.info("{} qas_ids unfound.".format(n_unfound))
        return translate_examples

    def get_noised_dataset(self, examples):
        # maybe do not save augmented examples
        examples = copy.deepcopy(examples)

        is_augmented = [0] * len(examples)
        if self.enable_data_augmentation:
            augment_examples = self.augment_examples(examples)
            if self.augment_method == "mt":
                assert not self.enable_code_switch
                augment_examples = self.get_translate_data(augment_examples)
            is_augmented += [1] * len(augment_examples)
            examples += augment_examples

        if self.enable_code_switch:
            self.n_tokens = 0
            self.n_cs_tokens = 0

        dataset = self.convert_examples_to_dataset(examples, is_augmented)

        if self.enable_code_switch:
            logger.info("{:.2f}% tokens have been code-switched.".format(self.n_cs_tokens / self.n_tokens * 100))
        return dataset

    def tokenize_token(self, token, switch_text=False, can_be_switched=True,
                       enable_code_switch=False,
                       enable_bpe_switch=False,
                       enable_bpe_sampling=False, ):
        switch_token = (random.random() <= self.overall_ratio) and can_be_switched
        is_switched = False
        self.n_tokens += 1
        if enable_code_switch and switch_text and switch_token and random.random() <= self.code_switch_ratio:
            lang = self.dict_languages[random.randint(0, len(self.dict_languages) - 1)]
            if token.lower() in self.lang2dict[lang]:
                self.n_cs_tokens += 1
                token = self.lang2dict[lang][token.lower()][
                    random.randint(0, len(self.lang2dict[lang][token.lower()]) - 1)]
                is_switched = True

        if enable_bpe_switch and switch_text and switch_token and random.random() <= self.bpe_switch_ratio:
            lang = self.tokenizer_languages[random.randint(0, len(self.tokenizer_languages) - 1)]
            tokenizer = self.lang2tokenizer[lang]
            is_switched = True
        else:
            tokenizer = self.tokenizer

        if enable_bpe_sampling and switch_text and switch_token and random.random() <= self.bpe_sampling_ratio:
            sub_tokens = tokenizer.tokenize(token, nbest_size=self.sampling_nbest_size,
                                            alpha=self.sampling_alpha)
            is_switched = True
        else:
            sub_tokens = tokenizer.tokenize(token)

        return sub_tokens, switch_token and is_switched

    def tokenize_sentence(self, sentence, switch_text=False):
        all_sub_tokens = []
        tokens = sentence.split(" ")
        for token in tokens:
            sub_tokens, switch_token = self.tokenize_token(token, switch_text)
            all_sub_tokens += sub_tokens
        return all_sub_tokens

    def convert_examples_to_dataset(self, examples, is_augmented=None, is_training=True):
        all_original_input_ids = []
        all_original_attention_mask = []
        all_original_token_type_ids = []
        all_original_r1_mask = []
        all_original_start_positions = []
        all_original_end_positions = []

        all_noised_input_ids = []
        all_noised_attention_mask = []
        all_noised_token_type_ids = []
        all_noised_r1_mask = []
        all_noised_start_positions = []
        all_noised_end_positions = []

        all_is_augmented = []

        for (ex_index, example) in enumerate(examples):
            if is_training and not example.is_impossible:
                # Get start and end position
                start_position = example.start_position
                end_position = example.end_position

                # If the answer cannot be found in the text, then skip this example.
                actual_text = " ".join(example.doc_tokens[start_position: (end_position + 1)])
                cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                    # exit(0)
            else:
                start_position, end_position = None, None

            if ex_index % 1000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
                # if ex_index == 1000:
                #     break

            # switch all examples
            switch_text = True
            noised_orig_to_tok_index = []
            noised_all_doc_tokens = []
            noised_tok_to_orig_index = []
            original_orig_to_tok_index = []
            original_all_doc_tokens = []
            original_tok_to_orig_index = []
            is_token_switched = [False] * len(example.doc_tokens)

            for (i, token) in enumerate(example.doc_tokens):
                original_orig_to_tok_index.append(len(original_all_doc_tokens))

                can_be_switched = False if self.keep_boundary_unchanged and (
                        i == start_position or i == end_position) else True
                if self.enable_data_augmentation and is_augmented[ex_index]:
                    if self.augment_method == "cs":
                        if start_position <= i <= end_position:
                            can_be_switched = False
                        original_sub_tokens, switch_token = self.tokenize_token(token, switch_text,
                                                                                can_be_switched=can_be_switched,
                                                                                enable_code_switch=True)
                    elif self.augment_method == "ss":
                        original_sub_tokens, switch_token = self.tokenize_token(token, switch_text,
                                                                                can_be_switched=can_be_switched,
                                                                                enable_bpe_sampling=True)
                    elif self.augment_method == "mt" or self.augment_method == "gn":
                        original_sub_tokens, switch_token = self.tokenize_token(token, switch_text=False)
                    else:
                        assert False
                else:
                    original_sub_tokens, switch_token = self.tokenize_token(token, switch_text=False)
                    # original_sub_tokens = self.tokenizer.tokenize(token)

                is_token_switched[i] = is_token_switched[i] or switch_token
                for sub_token in original_sub_tokens:
                    original_tok_to_orig_index.append(i)
                    original_all_doc_tokens.append(sub_token)

            keep_answer_unchanged = False
            if is_training and not example.is_impossible:
                original_tok_start_position = original_orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    original_tok_end_position = original_orig_to_tok_index[example.end_position + 1] - 1
                else:
                    original_tok_end_position = len(original_all_doc_tokens) - 1

                (new_original_tok_start_position, new_original_tok_end_position) = _improve_answer_span(
                    original_all_doc_tokens, original_tok_start_position, original_tok_end_position, self.tokenizer,
                    example.answer_text
                )

                keep_answer_unchanged = (original_tok_start_position != new_original_tok_start_position) or (
                        original_tok_end_position != new_original_tok_end_position)

            for (i, token) in enumerate(example.doc_tokens):
                noised_orig_to_tok_index.append(len(noised_all_doc_tokens))

                can_be_switched = False if self.keep_boundary_unchanged and (
                        i == start_position or i == end_position) else True
                if keep_answer_unchanged and i >= start_position and i <= end_position:
                    can_be_switched = False
                noised_sub_tokens, switch_token = self.tokenize_token(token, switch_text,
                                                                      can_be_switched=can_be_switched,
                                                                      enable_code_switch=self.enable_code_switch,
                                                                      enable_bpe_switch=self.enable_bpe_switch,
                                                                      enable_bpe_sampling=self.enable_bpe_sampling)
                is_token_switched[i] = is_token_switched[i] or switch_token
                for sub_token in noised_sub_tokens:
                    noised_tok_to_orig_index.append(i)
                    noised_all_doc_tokens.append(sub_token)

            if is_training and not example.is_impossible:
                noised_tok_start_position = noised_orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    noised_tok_end_position = noised_orig_to_tok_index[example.end_position + 1] - 1
                else:
                    noised_tok_end_position = len(noised_all_doc_tokens) - 1

                (noised_tok_start_position, noised_tok_end_position) = _improve_answer_span(
                    noised_all_doc_tokens, noised_tok_start_position, noised_tok_end_position, self.tokenizer,
                    example.answer_text
                )

            original_truncated_query = self.tokenizer.encode(example.question_text, add_special_tokens=False,
                                                             truncation=True, max_length=self.max_query_length)
            noised_question_sub_tokens = self.tokenize_sentence(example.question_text, switch_text)
            noised_truncated_query = self.tokenizer.encode(noised_question_sub_tokens, add_special_tokens=False,
                                                           truncation=True, max_length=self.max_query_length)
            sequence_added_tokens = (
                self.tokenizer.max_len - self.tokenizer.max_len_single_sentence + 1
                if "roberta" in str(type(self.tokenizer)) or "camembert" in str(type(self.tokenizer))
                else self.tokenizer.max_len - self.tokenizer.max_len_single_sentence
            )

            sequence_pair_added_tokens = self.tokenizer.max_len - self.tokenizer.max_len_sentences_pair

            spans = []
            span_doc_tokens = original_all_doc_tokens
            while len(spans) * self.doc_stride < len(original_all_doc_tokens):

                original_encoded_dict = self.tokenizer.encode_plus(  # TODO(thom) update this logic
                    original_truncated_query if self.tokenizer.padding_side == "right" else span_doc_tokens,
                    span_doc_tokens if self.tokenizer.padding_side == "right" else original_truncated_query,
                    max_length=self.max_seq_length,
                    return_overflowing_tokens=True,
                    pad_to_max_length=True,
                    stride=self.max_seq_length - self.doc_stride - len(
                        original_truncated_query) - sequence_pair_added_tokens,
                    truncation_strategy="only_second" if self.tokenizer.padding_side == "right" else "only_first",
                )

                paragraph_len = min(
                    len(original_all_doc_tokens) - len(spans) * self.doc_stride,
                    self.max_seq_length - len(original_truncated_query) - sequence_pair_added_tokens,
                )

                if self.tokenizer.pad_token_id in original_encoded_dict["input_ids"]:
                    if self.tokenizer.padding_side == "right":
                        non_padded_ids = original_encoded_dict["input_ids"][
                                         : original_encoded_dict["input_ids"].index(self.tokenizer.pad_token_id)]
                    else:
                        last_padding_id_position = (
                                len(original_encoded_dict["input_ids"]) - 1 - original_encoded_dict["input_ids"][
                                                                              ::-1].index(
                            self.tokenizer.pad_token_id)
                        )
                        non_padded_ids = original_encoded_dict["input_ids"][last_padding_id_position + 1:]

                else:
                    non_padded_ids = original_encoded_dict["input_ids"]

                tokens = self.tokenizer.convert_ids_to_tokens(non_padded_ids)

                original_encoded_dict["tokens"] = tokens
                original_encoded_dict["start"] = len(spans) * self.doc_stride
                original_encoded_dict["length"] = paragraph_len

                noised_tokens = []
                noised_r1_mask = []
                original_r1_mask = []
                token_to_orig_map = {}
                span_start = None
                break_flag = False
                for i in range(paragraph_len):
                    index = len(
                        original_truncated_query) + sequence_added_tokens + i if self.tokenizer.padding_side == "right" else i
                    token_to_orig_map[index] = original_tok_to_orig_index[len(spans) * self.doc_stride + i]

                    original_index = len(spans) * self.doc_stride + i
                    cur_orig_index = original_tok_to_orig_index[original_index]
                    pre_orig_index = original_tok_to_orig_index[original_index - 1] if i > 0 else -1

                    if not is_token_switched[cur_orig_index]:
                        noised_index = original_index - original_orig_to_tok_index[cur_orig_index] + \
                                       noised_orig_to_tok_index[cur_orig_index]
                        assert original_all_doc_tokens[original_index] == noised_all_doc_tokens[noised_index]
                        if span_start is None:
                            span_start = noised_index
                        if len(noised_tokens) + len(
                                noised_truncated_query) + sequence_pair_added_tokens == self.noised_max_seq_length:
                            break
                        noised_tokens.append(noised_all_doc_tokens[noised_index])
                        noised_r1_mask.append(1)
                    elif is_token_switched[cur_orig_index] and cur_orig_index != pre_orig_index:
                        noised_index = noised_orig_to_tok_index[cur_orig_index]
                        while noised_index < len(noised_tok_to_orig_index):
                            if noised_tok_to_orig_index[noised_index] != cur_orig_index:
                                break
                            if span_start is None:
                                span_start = noised_index
                            if len(noised_tokens) + len(
                                    noised_truncated_query) + sequence_pair_added_tokens == self.noised_max_seq_length:
                                break_flag = True
                                break
                            noised_tokens.append(noised_all_doc_tokens[noised_index])
                            noised_r1_mask.append(0)
                            noised_index += 1

                        if break_flag:
                            break

                    original_r1_mask.append(1 if not is_token_switched[cur_orig_index] else 0)

                assert len(noised_tokens) + len(
                    noised_truncated_query) + sequence_pair_added_tokens <= self.noised_max_seq_length

                if self.tokenizer.padding_side == "right":
                    noised_r1_mask = [0] * (len(noised_truncated_query) + 3) + noised_r1_mask + [0]
                    original_r1_mask = [0] * (len(original_truncated_query) + 3) + original_r1_mask + [0]
                else:
                    assert False

                noised_r1_mask += (self.noised_max_seq_length - len(noised_r1_mask)) * [0]
                original_r1_mask += (self.max_seq_length - len(original_r1_mask)) * [0]

                noised_encoded_dict = self.tokenizer.encode_plus(  # TODO(thom) update this logic
                    noised_truncated_query if self.tokenizer.padding_side == "right" else noised_tokens,
                    noised_tokens if self.tokenizer.padding_side == "right" else original_truncated_query,
                    max_length=self.noised_max_seq_length,
                    pad_to_max_length=True,
                    truncation_strategy="only_second" if self.tokenizer.padding_side == "right" else "only_first",
                )

                if self.tokenizer.pad_token_id in noised_encoded_dict["input_ids"]:
                    if self.tokenizer.padding_side == "right":
                        non_padded_ids = noised_encoded_dict["input_ids"][
                                         : noised_encoded_dict["input_ids"].index(self.tokenizer.pad_token_id)]
                    else:
                        last_padding_id_position = (
                                len(noised_encoded_dict["input_ids"]) - 1 - noised_encoded_dict["input_ids"][
                                                                            ::-1].index(
                            self.tokenizer.pad_token_id)
                        )
                        non_padded_ids = noised_encoded_dict["input_ids"][last_padding_id_position + 1:]
                else:
                    non_padded_ids = noised_encoded_dict["input_ids"]

                tokens = self.tokenizer.convert_ids_to_tokens(non_padded_ids)

                noised_encoded_dict["tokens"] = tokens
                noised_encoded_dict["r1_mask"] = noised_r1_mask
                assert span_start is not None
                noised_encoded_dict["start"] = span_start
                noised_encoded_dict["length"] = len(noised_tokens)

                original_encoded_dict["r1_mask"] = original_r1_mask

                spans.append((original_encoded_dict, noised_encoded_dict))

                if "overflowing_tokens" not in original_encoded_dict:
                    break
                span_doc_tokens = original_encoded_dict["overflowing_tokens"]

            for (original_span, noised_span) in spans:
                # Identify the position of the CLS token
                original_cls_index = original_span["input_ids"].index(self.tokenizer.cls_token_id)
                noised_cls_index = noised_span["input_ids"].index(self.tokenizer.cls_token_id)

                # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
                # Original TF implem also keep the classification token (set to 0) (not sure why...)
                original_p_mask = np.array(original_span["token_type_ids"])
                noised_p_mask = np.array(noised_span["token_type_ids"])

                original_p_mask = np.minimum(original_p_mask, 1)
                noised_p_mask = np.minimum(noised_p_mask, 1)

                if self.tokenizer.padding_side == "right":
                    # Limit positive values to one
                    original_p_mask = 1 - original_p_mask
                    noised_p_mask = 1 - noised_p_mask

                original_p_mask[np.where(np.array(original_span["input_ids"]) == self.tokenizer.sep_token_id)[0]] = 1
                noised_p_mask[np.where(np.array(noised_span["input_ids"]) == self.tokenizer.sep_token_id)[0]] = 1

                # Set the CLS index to '0'
                original_p_mask[original_cls_index] = 0
                noised_p_mask[noised_cls_index] = 0

                # TODO cls_index in xlm-r is 0
                assert original_cls_index == 0
                assert noised_cls_index == 0
                original_span["r1_mask"][original_cls_index] = 1
                noised_span["r1_mask"][noised_cls_index] = 1

                span_is_impossible = example.is_impossible
                original_start_position = 0
                original_end_position = 0
                noised_start_position = 0
                noised_end_position = 0
                if is_training and not span_is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    noised_doc_start = noised_span["start"]
                    noised_doc_end = noised_span["start"] + noised_span["length"] - 1
                    noised_out_of_span = False
                    original_doc_start = original_span["start"]
                    original_doc_end = original_span["start"] + original_span["length"] - 1
                    original_out_of_span = False

                    if not (
                            noised_tok_start_position >= noised_doc_start and noised_tok_end_position <= noised_doc_end):
                        noised_out_of_span = True

                    if not (
                            new_original_tok_start_position >= original_doc_start and new_original_tok_end_position <= original_doc_end):
                        original_out_of_span = True

                    if noised_out_of_span:
                        noised_start_position = noised_cls_index
                        noised_end_position = noised_cls_index
                        span_is_impossible = True
                    else:
                        if self.tokenizer.padding_side == "left":
                            doc_offset = 0
                        else:
                            doc_offset = len(noised_truncated_query) + sequence_added_tokens

                        noised_start_position = noised_tok_start_position - noised_doc_start + doc_offset
                        noised_end_position = noised_tok_end_position - noised_doc_start + doc_offset

                    if original_out_of_span:
                        original_start_position = original_cls_index
                        original_end_position = original_cls_index
                        span_is_impossible = True
                    else:
                        if self.tokenizer.padding_side == "left":
                            doc_offset = 0
                        else:
                            doc_offset = len(original_truncated_query) + sequence_added_tokens
                        original_start_position = new_original_tok_start_position - original_doc_start + doc_offset
                        original_end_position = new_original_tok_end_position - original_doc_start + doc_offset

                all_original_input_ids += [original_span["input_ids"]]
                all_original_attention_mask += [original_span["attention_mask"]]
                all_original_token_type_ids += [original_span["token_type_ids"]]
                all_original_r1_mask += [original_span["r1_mask"]]
                all_original_start_positions += [original_start_position]
                all_original_end_positions += [original_end_position]

                all_noised_input_ids += [noised_span["input_ids"]]
                all_noised_attention_mask += [noised_span["attention_mask"]]
                all_noised_token_type_ids += [noised_span["token_type_ids"]]
                all_noised_r1_mask += [noised_span["r1_mask"]]
                all_noised_start_positions += [noised_start_position]
                all_noised_end_positions += [noised_end_position]
                all_is_augmented += [is_augmented[ex_index]]

        # Convert to Tensors and build dataset
        all_original_input_ids = torch.tensor([input_ids for input_ids in all_original_input_ids], dtype=torch.long)
        all_original_attention_mask = torch.tensor([attention_mask for attention_mask in all_original_attention_mask],
                                                   dtype=torch.long)
        all_original_token_type_ids = torch.tensor([token_type_ids for token_type_ids in all_original_token_type_ids],
                                                   dtype=torch.long)
        all_original_r1_mask = torch.tensor([original_r1_mask for original_r1_mask in all_original_r1_mask],
                                            dtype=torch.long)
        all_original_start_positions = torch.tensor([start_position for start_position in all_original_start_positions],
                                                    dtype=torch.long)
        all_original_end_positions = torch.tensor([end_position for end_position in all_original_end_positions],
                                                  dtype=torch.long)

        all_noised_input_ids = torch.tensor([input_ids for input_ids in all_noised_input_ids], dtype=torch.long)
        all_noised_attention_mask = torch.tensor([attention_mask for attention_mask in all_noised_attention_mask],
                                                 dtype=torch.long)
        all_noised_token_type_ids = torch.tensor([token_type_ids for token_type_ids in all_noised_token_type_ids],
                                                 dtype=torch.long)
        all_noised_r1_mask = torch.tensor([noised_r1_mask for noised_r1_mask in all_noised_r1_mask],
                                          dtype=torch.long)
        all_noised_start_positions = torch.tensor([start_position for start_position in all_noised_start_positions],
                                                  dtype=torch.long)
        all_noised_end_positions = torch.tensor([end_position for end_position in all_noised_end_positions],
                                                dtype=torch.long)
        all_is_augmented = torch.tensor([is_augmented for is_augmented in all_is_augmented])
        dataset = TensorDataset(all_original_input_ids, all_original_attention_mask, all_original_token_type_ids,
                                all_original_start_positions, all_original_end_positions, all_original_attention_mask,
                                all_original_attention_mask, all_original_attention_mask,
                                all_noised_input_ids, all_noised_attention_mask, all_noised_token_type_ids,
                                all_noised_r1_mask, all_original_r1_mask, all_noised_start_positions,
                                all_noised_end_positions, all_is_augmented)
        return dataset

    def get_train_steps(self, examples, args):
        if args.max_steps > 0:
            t_total = args.max_steps
        else:
            assert False
        return t_total


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_examples, train_dataset, model, first_stage_model, tokenizer, noised_data_generator=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_log_dir = os.getenv("PHILLY_JOB_DIRECTORY", None)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'w')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # args.warmup_steps == -1 means 0.1 warmup ratio
    if args.warmup_steps == -1:
        args.warmup_steps = int(t_total * 0.1)
    logger.info("Warmup steps: %d" % args.warmup_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss, best_avg_f1 = 0.0, 0.0, 0.0
    tr_original_loss, logging_original_loss = 0.0, 0.0
    tr_noised_loss, logging_noised_loss = 0.0, 0.0
    tr_r1_loss, logging_r1_loss = 0.0, 0.0
    tr_r2_loss, logging_r2_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    def logging(eval=False):
        results = None
        # Only evaluate when single GPU otherwise metrics may not average well
        if args.local_rank in [-1, 0] and args.evaluate_during_training and eval:
            results = evaluate(args, model, tokenizer)
            for key, value in results.items():
                logger.info("eval_{}: {}".format(key, value))
            # for key, value in results.items():
            #     tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            log_writer.write("{0}\t{1}".format(global_step, json.dumps(results)) + '\n')
            log_writer.flush()
        logger.info(
            "global_step: {}, lr: {:.6f}, loss: {:.6f}, original_loss: {:.6f}, noised_loss: {:.6f}, r1_loss: {:.6f}, r2_loss: {:.6f}".format(
                global_step, scheduler.get_lr()[0], (tr_loss - logging_loss) / args.logging_steps,
                                                    (tr_original_loss - logging_original_loss) / args.logging_steps,
                                                    (tr_noised_loss - logging_noised_loss) / args.logging_steps,
                                                    (tr_r1_loss - logging_r1_loss) / args.logging_steps,
                                                    (tr_r2_loss - logging_r2_loss) / args.logging_steps))
        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
        tb_writer.add_scalar("original_loss", (tr_original_loss - logging_original_loss) / args.logging_steps,
                             global_step)
        tb_writer.add_scalar("noised_loss", (tr_noised_loss - logging_noised_loss) / args.logging_steps, global_step)
        tb_writer.add_scalar("r1_loss", (tr_r1_loss - logging_r1_loss) / args.logging_steps, global_step)
        tb_writer.add_scalar("r2_loss", (tr_r2_loss - logging_r2_loss) / args.logging_steps, global_step)
        if results is not None:
            return results["dev_avg"]["f1"]
        else:
            return None

    for _ in train_iterator:
        use_noised_ids = False
        if noised_data_generator is not None:
            assert noised_data_generator.enable_r1_loss or noised_data_generator.noised_loss or noised_data_generator.enable_data_augmentation

            noised_train_dataset = noised_data_generator.get_noised_dataset(train_examples)

            train_sampler = RandomSampler(noised_train_dataset) if args.local_rank == -1 else DistributedSampler(
                noised_train_dataset)
            train_dataloader = DataLoader(noised_train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            if first_stage_model is not None:
                first_stage_model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if first_stage_model is not None:
                with torch.no_grad():
                    inputs["first_stage_model_start_logits"], inputs["first_stage_model_end_logits"] = first_stage_model(**inputs)[1:3]

            if noised_data_generator is not None:
                inputs.update({"noised_input_ids": batch[8], "noised_attention_mask": batch[9],
                               "noised_token_type_ids": batch[10], "noised_r1_mask": batch[11],
                               "original_r1_mask": batch[12], "noised_start_positions": batch[13],
                               "noised_end_positions": batch[14], "is_augmented": batch[15]})

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]
                if use_noised_ids:
                    del inputs["noised_token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                assert False
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if True or noised_data_generator is not None:
                original_loss, noised_loss, r1_loss, r2_loss = outputs[1:5]
                if args.n_gpu > 1:
                    original_loss = original_loss.mean()
                    noised_loss = noised_loss.mean()
                    r1_loss = r1_loss.mean()
                    r2_loss = r2_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    original_loss = original_loss / args.gradient_accumulation_steps
                    noised_loss = noised_loss / args.gradient_accumulation_steps
                    r1_loss = r1_loss / args.gradient_accumulation_steps
                    r2_loss = r2_loss / args.gradient_accumulation_steps

                tr_original_loss += original_loss.item()
                tr_noised_loss += noised_loss.item()
                tr_r1_loss += r1_loss.item()
                tr_r2_loss += r2_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    cur_result = logging(eval=args.evaluate_steps > 0 and global_step % args.evaluate_steps == 0)
                    logging_loss = tr_loss
                    logging_original_loss = tr_original_loss
                    logging_noised_loss = tr_noised_loss
                    logging_r1_loss = tr_r1_loss
                    logging_r2_loss = tr_r2_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0] and args.logging_each_epoch:
            avg_f1 = logging(eval=True)
            logging_loss = tr_loss
            logging_original_loss = tr_original_loss
            logging_noised_loss = tr_noised_loss
            logging_r1_loss = tr_r1_loss
            logging_r2_loss = tr_r2_loss
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                output_dir = os.path.join(args.output_dir, "checkpoint-best")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        log_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    languages = args.language.split(',')
    all_languages_results = {}

    if args.task_name.lower() == "mlqa" or args.task_name == "mlqa_dev":
        processor = MLQAProcessor()
    elif args.task_name.lower() == "xquad":
        processor = XQuADProcessor()
    elif args.task_name.lower() == "tydiqa":
        processor = TyDiQAProcessor()
    elif args.task_name.lower() == "squad":
        processor = SquadV1Processor()
    else:
        assert False

    split_lang_list = []
    # split_lang_list.append(("run_dev", "en"))
    for lang in languages:
        split_lang_list.append(("dev", lang))

    if args.task_name.lower() == "mlqa":
        for lang in languages:
            split_lang_list.append(("test", lang))

    for split, lang in split_lang_list:
        # for split, lang in itertools.product(["dev", "test"], languages):
        print("evaluating on {0} {1}".format(split, lang))
        dataset, examples, features = load_and_cache_examples(args, tokenizer, language=lang, split=split,
                                                              output_examples=True)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_results = []
        start_time = timeit.default_timer()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                    del inputs["token_type_ids"]

                example_indices = batch[3]

                # XLNet and XLM use more arguments for their predictions
                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                        )

                outputs = model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs]

                # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
                # models only use two.
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]

                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )

                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

        # Compute predictions
        output_prediction_file = os.path.join(args.output_dir, "{}.prediction".format(lang))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}_{}_{}.json".format(prefix, split, lang))

        if args.version_2_with_negative:
            output_null_log_odds_file = os.path.join(args.output_dir,
                                                     "null_odds_{}_{}_{}.json".format(prefix, split, lang))
        else:
            output_null_log_odds_file = None

        # XLNet and XLM use a more complex post-processing procedure
        if args.model_type in ["xlnet", "xlm"]:
            start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
            end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

            predictions = compute_predictions_log_probs(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                start_n_top,
                end_n_top,
                args.version_2_with_negative,
                tokenizer,
                args.verbose_logging,
            )
        else:
            predictions = compute_predictions_logits(
                examples,
                features,
                all_results,
                args.n_best_size,
                args.max_answer_length,
                args.do_lower_case,
                output_prediction_file,
                output_nbest_file,
                output_null_log_odds_file,
                args.verbose_logging,
                args.version_2_with_negative,
                args.null_score_diff_threshold,
                tokenizer,
                map_to_origin=not (args.model_type == "xlmr" and (lang == 'zh' or lang == "ko")),
                # map_to_origin=False,
            )

        # Compute the F1 and exact scores.
        if args.task_name.lower() == "mlqa" or args.task_name.lower() == "mlqa_dev":
            results = mlqa_evaluate_with_path(processor.get_dataset_path(args.data_dir, split, lang),
                                              output_prediction_file, lang)
        else:
            results = squad_evaluate_with_path(processor.get_dataset_path(args.data_dir, split, lang),
                                               output_prediction_file)
        # results = squad_evaluate(examples, predictions)
        # results = evaluate_with_path(processor.get_dataset_path(args.data_dir, split, lang), output_prediction_file,
        #                              lang)
        all_languages_results["{0}_{1}".format(split, lang)] = results
    for split in ["dev", "test"]:
        all_languages_results["{0}_avg".format(split)] = average_dic(
            [value for key, value in all_languages_results.items() if split in key])

    return all_languages_results


def average_dic(dic_list):
    if len(dic_list) == 0:
        return {}
    dic_sum = {}
    for dic in dic_list:
        if len(dic_sum) == 0:
            for key, value in dic.items():
                dic_sum[key] = value
        else:
            assert set(dic_sum.keys()) == set(dic.keys()), "sum_keys:{0}, dic_keys:{1}".format(set(dic_sum.keys()),
                                                                                               set(dic.keys()))
            for key, value in dic.items():
                dic_sum[key] += value
    for key in dic_sum:
        dic_sum[key] /= len(dic_list)
    return dic_sum


def load_and_cache_examples(args, tokenizer, language, split="train", output_examples=False):
    if args.local_rank not in [-1, 0] and split == "train":
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    model_name = "xlmr-base-final"
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}_{}".format(
            split,
            language,
            model_name,
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and (
                (split != "train" and not args.predict_file) or (split == "train" and not args.train_file)):
            raise ValueError("data dir can't be empty")
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            # processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if args.task_name.lower() == "mlqa" or args.task_name.lower() == "mlqa_dev":
                processor = MLQAProcessor()
            elif args.task_name.lower() == "xquad":
                processor = XQuADProcessor()
            elif args.task_name.lower() == "tydiqa":
                processor = TyDiQAProcessor()
            elif args.task_name.lower() == "squad":
                processor = SquadV1Processor()
            else:
                assert False


            if split == "run_dev":
                examples = processor.get_dev_examples(args.data_dir)
            elif split == "dev":
                if args.task_name.lower() == "squad":
                    examples = processor.get_dev_examples(args.data_dir)
                else:
                    examples = processor.get_dev_examples_by_language(args.data_dir, language=language)
            elif split == "test":
                examples = processor.get_test_examples_by_language(args.data_dir, language=language)
            else:
                examples = processor.get_train_examples(args.data_dir)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=split == "train",
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and split == "train":
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--reload",
        default="",
        type=str,
        help="path to infoxlm checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--task_name",
        default="mlqa",
        type=str,
        help="task_name",
    )

    # stable fine-tuning paramters
    parser.add_argument("--overall_ratio", default=1.0, type=float, help="overall ratio")
    parser.add_argument("--enable_r1_loss", action="store_true", help="Whether to enable r1 loss.")
    parser.add_argument("--r1_lambda", default=5.0, type=float, help="lambda of r1 loss")
    parser.add_argument("--original_loss", action="store_true",
                        help="Whether to use cross entropy loss on the former example.")
    parser.add_argument("--noised_loss", action="store_true",
                        help="Whether to use cross entropy loss on the latter example.")
    parser.add_argument("--noised_max_seq_length", default=512, type=int, help="noised max sequence length")
    parser.add_argument("--keep_boundary_unchanged", action="store_true",
                        help="Whether to keep the boundary of answer unchanged.")
    parser.add_argument("--r1_on_boundary_only", action="store_true",
                        help="Whether to enable r1 loss on boundary only.")
    parser.add_argument("--enable_bpe_switch", action="store_true", help="Whether to enable bpe-switch.")
    parser.add_argument("--bpe_switch_ratio", default=0.5, type=float, help="bpe_switch_ratio")
    parser.add_argument("--tokenizer_dir", default=None, type=str, help="tokenizer dir")
    parser.add_argument("--tokenizer_languages", default=None, type=str, help="tokenizer languages")
    parser.add_argument("--enable_bpe_sampling", action="store_true", help="Whether to enable bpe sampling.")
    parser.add_argument("--bpe_sampling_ratio", default=0.5, type=float, help="bpe_sampling_ratio")
    parser.add_argument("--sampling_alpha", default=5.0, type=float, help="alpha of sentencepiece sampling")
    parser.add_argument("--sampling_nbest_size", default=-1, type=int, help="nbest_size of sentencepiece sampling")
    parser.add_argument("--enable_random_noise", action="store_true", help="Whether to enable random noise.")
    parser.add_argument("--noise_detach_embeds", action="store_true", help="Whether to detach noised embeddings.")
    parser.add_argument("--noise_eps", default=1e-5, type=float, help="noise eps")
    parser.add_argument('--noise_type', type=str, default='uniform',
                        choices=['normal', 'uniform'],
                        help='type of noises for RXF methods')
    parser.add_argument("--enable_code_switch", action="store_true", help="Whether to enable code switch.")
    parser.add_argument("--code_switch_ratio", default=0.5, type=float, help="code_switch_ratio")
    parser.add_argument("--dict_dir", default=None, type=str, help="dict dir")
    parser.add_argument("--dict_languages", default=None, type=str, help="dict languages")
    parser.add_argument("--enable_translate_data", action="store_true",
                        help="Whether to enable translate data.")
    parser.add_argument("--translation_path", default=None, type=str, help="path to translation")
    parser.add_argument("--disable_translate_labels", action="store_true", help="Whether to disable translate labels.")
    parser.add_argument("--translate_languages", default=None, type=str, help="translate languages")
    parser.add_argument("--translate_augment_ratio", default=0.0, type=float, help="translate augment ratio")
    parser.add_argument("--enable_data_augmentation", action="store_true", help="Whether to enable data augmentation.")
    parser.add_argument("--augment_ratio", default=1.0, type=float, help="augmentation ratio.")
    parser.add_argument("--augment_method", default=None, type=str, required=False, help="augment_method")
    parser.add_argument("--first_stage_model_path", default=None, type=str, required=False,
                        help="stable model path")
    parser.add_argument("--r2_lambda", default=1.0, type=float, required=False,
                        help="r2_lambda")
    parser.add_argument("--use_hard_labels", action="store_true", help="Whether to use hard labels.")

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
             + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
             "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
             "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--evaluate_steps", type=int, default=0, help="Log every X updates steps.")
    parser.add_argument("--logging_each_epoch", action="store_true", help="Whether to log after each epoch.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    # cross-lingual part
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )
    parser.add_argument(
        "--train_language", default=None, type=str, help="Train language if is different of the evaluation language."
    )
    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.enable_r1_loss or args.noised_loss or args.enable_translate_data or args.enable_data_augmentation:
        noised_data_generator = NoisedDataGenerator(
            task_name=args.task_name,
            r1_lambda=args.r1_lambda,
            enable_r1_loss=args.enable_r1_loss,
            original_loss=args.original_loss,
            noised_loss=args.noised_loss,
            keep_boundary_unchanged=args.keep_boundary_unchanged,
            r1_on_boundary_only=args.r1_on_boundary_only,
            noised_max_seq_length=args.noised_max_seq_length,
            max_seq_length=args.max_seq_length,
            max_query_length=args.max_query_length,
            doc_stride=args.doc_stride,
            overall_ratio=args.overall_ratio,
            enable_bpe_switch=args.enable_bpe_switch,
            bpe_switch_ratio=args.bpe_switch_ratio,
            tokenizer_dir=args.tokenizer_dir,
            do_lower_case=args.do_lower_case,
            tokenizer_languages=args.tokenizer_languages.split(',') if args.tokenizer_languages is not None else [],
            enable_bpe_sampling=args.enable_bpe_sampling,
            bpe_sampling_ratio=args.bpe_sampling_ratio,
            tokenizer=tokenizer,
            sampling_alpha=args.sampling_alpha,
            sampling_nbest_size=args.sampling_nbest_size,
            enable_random_noise=args.enable_random_noise,
            noise_detach_embeds=args.noise_detach_embeds,
            noise_eps=args.noise_eps,
            noise_type=args.noise_type,
            enable_code_switch=args.enable_code_switch,
            code_switch_ratio=args.code_switch_ratio,
            dict_dir=args.dict_dir,
            dict_languages=args.dict_languages.split(',') if args.dict_languages is not None else [],
            translation_path=args.translation_path,
            disable_translate_labels=args.disable_translate_labels,
            translate_languages=args.translate_languages.split(
                ',') if args.translate_languages is not None else args.language.split(','),
            enable_data_augmentation=args.enable_data_augmentation,
            augment_ratio=args.augment_ratio,
            augment_method=args.augment_method,
            r2_lambda=args.r2_lambda,
            use_hard_labels=args.use_hard_labels,
        )
    else:
        noised_data_generator = None

    if args.first_stage_model_path is not None:
        first_stage_model = model_class.from_pretrained(args.first_stage_model_path,
                                                   config=config)
    else:
        first_stage_model = None

    state_dict = None
    if args.reload != "":
        from tools.dump_hf_state_dict import convert_pt_to_hf
        state_dict = convert_pt_to_hf(os.path.join(args.model_name_or_path, 'pytorch_model.bin'), args.reload, logger)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        noised_data_generator=noised_data_generator,
        cache_dir=args.cache_dir if args.cache_dir else None,
        state_dict=state_dict,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    if first_stage_model is not None:
        first_stage_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        train_dataset, train_examples, _ = load_and_cache_examples(args, tokenizer, language=args.train_language,
                                                                   split="train", output_examples=True)
        global_step, tr_loss = train(args, train_examples, train_dataset, model, first_stage_model, tokenizer,
                                     noised_data_generator=noised_data_generator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)  # , force_download=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}

    if args.do_eval and args.local_rank in [-1, 0]:
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else "test"
            model = model_class.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'w')
            result = evaluate(args, model, tokenizer, prefix=global_step)
            # result = squad(args, model, tokenizer, prefix=global_step)
            log_writer.write("{0}\t{1}".format(global_step, json.dumps(result)) + '\n')
            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))
    logger.info("Task MLQA Finished!")

    return results


if __name__ == "__main__":
    main()
