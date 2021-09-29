# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
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
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import copy
import json
import random
import math

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_tag import convert_examples_to_features
from utils_tag import get_labels
from utils_tag import read_examples_from_file
from utils_tag import InputExample

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    RobertaConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassificationPoolingStable,
)

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in (RobertaConfig, XLMRobertaConfig)),
    ()
)

MODEL_CLASSES = {
    "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassificationPoolingStable, XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_root(x, parent):
    if x == parent[x]: return x
    parent[x] = get_root(parent[x], parent)
    return parent[x]


class NoisedDataGenerator(object):
    def __init__(self,
                 label_list,
                 pad_token_label_id,
                 r1_lambda=5.0,
                 r1_on_unswitched_tokens=False,
                 enable_r1_loss=False,
                 disable_backward_kl=False,
                 use_sentence_label_probs=False,
                 use_token_label_probs=False,
                 original_loss=True,
                 noised_loss=False,
                 max_seq_length=512,
                 noised_max_seq_length=512,
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
                 detach_embeds=False,
                 noise_eps=1e-5,
                 noise_type='uniform',
                 enable_code_switch=False,
                 code_switch_ratio=0.5,
                 dict_dir=None,
                 dict_languages=None,
                 use_average_representations=False,
                 translation_path=None,
                 translate_languages=None,
                 use_align_label_probs=False,
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
        if enable_r1_loss:
            assert use_token_label_probs or use_sentence_label_probs or (
                    use_align_label_probs and enable_translate_data)

        self.use_average_representations = use_average_representations

        self.n_tokens = 0
        self.n_cs_tokens = 0
        self.r1_lambda = r1_lambda
        self.r1_on_unswitched_tokens = r1_on_unswitched_tokens
        self.original_loss = original_loss
        self.noised_loss = noised_loss
        self.enable_r1_loss = enable_r1_loss
        self.disable_backward_kl = disable_backward_kl
        self.use_align_label_probs = use_align_label_probs
        self.use_sentence_label_probs = use_sentence_label_probs
        self.use_token_label_probs = use_token_label_probs
        self.max_seq_length = max_seq_length
        self.noised_max_seq_length = noised_max_seq_length
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
        self.detach_embeds = detach_embeds
        self.noise_eps = noise_eps
        self.noise_type = noise_type

        self.enable_code_switch = enable_code_switch
        self.code_switch_ratio = code_switch_ratio / self.overall_ratio
        assert not self.enable_code_switch or self.code_switch_ratio <= 1.0
        self.dict_dir = dict_dir
        self.dict_languages = []
        self.lang2dict = {}
        for lang in dict_languages:
            # dict_path = os.path.join(self.dict_dir, "{}2.txt".format(lang))
            dict_path = os.path.join(self.dict_dir, "en-{}.txt".format(lang))
            if not os.path.exists(dict_path):
                logger.info("dictionary en-{} doesn't exist.".format(lang))
                continue
            self.dict_languages.append(lang)
            logger.info("reading dictionary from {}".format(dict_path))
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
        self.translate_languages = translate_languages
        self.augment_method = augment_method
        self.enable_data_augmentation = enable_data_augmentation
        if self.enable_data_augmentation and self.augment_method == "mt":
            drop_languages = ["en", "zh-CN", "zh", "ja", "ko", "th", "my", "ml", "ta"]
            for lang in drop_languages:
                if lang in self.translate_languages:
                    self.translate_languages.remove(lang)
            # self.translate_languages = ["de"]
            self.src2tgt = {}
            logger.info("Reading translation from {}".format(self.translation_path))
            with open(self.translation_path, encoding="utf-8") as f:
                line_cnt = 0
                for line in f:
                    # if line_cnt == 100:
                    #     exit(0)
                    line_cnt += 1
                    if line_cnt % 10000 == 0:
                        print("Reading lines {}".format(line_cnt))
                    items = line.split("\t")
                    if len(items) == 3:
                        src_sent, tgt_lang, tgt_sent = line.split("\t")
                        alignment = None
                    else:
                        src_sent, tgt_lang, tgt_sent, alignment_str = line.split("\t")
                        alignment = []
                        for x in alignment_str.split(" "):
                            alignment.append((int(x.split("/")[0]), int(x.split("/")[1])))

                    if tgt_lang in drop_languages:
                        continue
                    if self.translate_languages is not None and tgt_lang not in self.translate_languages:
                        continue
                    if src_sent not in self.src2tgt:
                        self.src2tgt[src_sent] = []
                    if alignment is not None:
                        n_src = len(src_sent.split(" "))
                        n_tgt = len(tgt_sent.split(" "))
                        parent = list(range(0, n_src + n_tgt))
                        for x in alignment:
                            x_src = x[0]
                            x_tgt = x[1] + n_src
                            if get_root(x_src, parent) != get_root(x_tgt, parent):
                                parent[x_src] = get_root(x_tgt, parent)

                        cnt = [0] * (n_src + n_tgt)
                        for i in range(n_src + n_tgt):
                            cnt[get_root(i, parent)] += 1

                        align_pooling_id = [0] * (n_src + n_tgt)
                        root2id = {}
                        for i in range(n_src + n_tgt):
                            if cnt[get_root(i, parent)] == 1:
                                continue
                            if not get_root(i, parent) in root2id:
                                root2id[get_root(i, parent)] = len(root2id) + 1
                            align_pooling_id[i] = root2id[get_root(i, parent)]
                        # print(align_pooling_id[:n_src], align_pooling_id[n_src:])
                        self.src2tgt[src_sent].append(
                            (tgt_lang, tgt_sent, (align_pooling_id[:n_src], align_pooling_id[n_src:])))
                    else:
                        self.src2tgt[src_sent].append(
                            (tgt_lang, tgt_sent, None))
                    # print(align_pooling_id[:n_src], align_pooling_id[n_src:])

        self.enable_data_augmentation = enable_data_augmentation
        self.augment_ratio = augment_ratio
        self.r2_lambda = r2_lambda
        self.use_hard_labels = use_hard_labels

        self.label_list = label_list
        self.cls_token_at_end = False
        self.cls_token = self.tokenizer.cls_token
        self.cls_token_segment_id = 0
        self.sep_token = self.tokenizer.sep_token
        self.sep_token_extra = True
        self.pad_on_left = False
        self.pad_token = self.tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        self.pad_token_segment_id = 0
        self.pad_token_label_id = pad_token_label_id
        self.sequence_a_segment_id = 0
        self.mask_padding_with_zero = True

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
        for ex_idx, example in enumerate(examples):
            src_sent = " ".join(example.words)
            if src_sent not in self.src2tgt:
                logger.info("sentence || {} || is not found in translate data".format(src_sent))
                tgt_sent = src_sent
                tgt_lang = "en"
                align_pooling_id = (
                    list(range(1, len(src_sent.split(" ")) + 1)), list(range(1, len(src_sent.split(" ")) + 1)))
                n_unfound += 1
            else:
                # assert src_sent in self.src2tgt
                idx = random.randint(0, len(self.src2tgt[src_sent]) - 1)
                tgt_lang, tgt_sent, align_pooling_id = self.src2tgt[src_sent][idx]
            words = tgt_sent.split(" ")
            # print(len(words))
            labels = ['<unk_label>'] * len(words)
            translate_examples.append(InputExample(ex_idx, words, labels, langs=tgt_lang))

        logger.info("{} sentences unfound.".format(n_unfound))
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
        return dataset, None

    def tokenize_token(self, token, switch_text=False, enable_code_switch=False, enable_bpe_switch=False,
                       enable_bpe_sampling=False):
        switch_token = random.random() <= self.overall_ratio
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

        return sub_tokens, is_switched

    def convert_examples_to_dataset(self, examples, is_augmented):
        all_original_input_ids = []
        all_original_input_mask = []
        all_original_segment_ids = []
        all_original_label_ids = []
        all_original_pooling_ids = []
        all_original_r1_mask = []

        all_noised_input_ids = []
        all_noised_input_mask = []
        all_noised_segment_ids = []
        all_noised_label_ids = []
        all_noised_pooling_ids = []
        all_noised_r1_mask = []
        all_is_augmented = []

        label_map = {label: i for i, label in enumerate(self.label_list)}

        for (ex_index, example) in enumerate(examples):
            if ex_index % 1000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len(examples)))
            noised_tokens = []
            original_tokens = []
            noised_label_ids = []
            original_label_ids = []
            noised_pooling_ids = []
            original_pooling_ids = []
            noised_r1_mask = []
            original_r1_mask = []
            switch_text = True

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if self.sep_token_extra else 2

            for word, label in zip(example.words, example.labels):
                noised_word_tokens, noised_is_switched = self.tokenize_token(word, switch_text=switch_text,
                                                                             enable_code_switch=self.enable_code_switch,
                                                                             enable_bpe_switch=self.enable_bpe_switch,
                                                                             enable_bpe_sampling=self.enable_bpe_sampling)
                if self.enable_data_augmentation and is_augmented[ex_index]:
                    if self.augment_method == "cs":
                        original_word_tokens, original_is_switched = self.tokenize_token(word, switch_text=switch_text,
                                                                                         enable_code_switch=True)
                    elif self.augment_method == "ss":
                        original_word_tokens, original_is_switched = self.tokenize_token(word, switch_text=switch_text,
                                                                                         enable_bpe_sampling=True)
                    elif self.augment_method == "mt" or self.augment_method == "gn":
                        original_word_tokens, original_is_switched = self.tokenize_token(word, switch_text=False)
                    else:
                        assert False
                else:
                    original_word_tokens, original_is_switched = self.tokenize_token(word, switch_text=False)

                is_switched = noised_is_switched or original_is_switched

                if len(word) != 0 and len(original_word_tokens) == 0:
                    original_word_tokens = [self.tokenizer.unk_token]

                if len(word) != 0 and len(noised_word_tokens) == 0:
                    noised_word_tokens = [self.tokenizer.unk_token]

                if len(noised_word_tokens) == 0 or len(original_word_tokens) == 0:
                    continue

                noised_tokens.extend(noised_word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                noised_label_ids.extend([label_map.get(label, self.pad_token_label_id)] + [self.pad_token_label_id] * (
                        len(noised_word_tokens) - 1))
                noised_pooling_ids.extend([len(noised_pooling_ids) + 1] * len(noised_word_tokens))

                original_tokens.extend(original_word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                original_label_ids.extend(
                    [label_map.get(label, self.pad_token_label_id)] + [self.pad_token_label_id] * (
                            len(original_word_tokens) - 1))
                original_pooling_ids.extend([len(original_pooling_ids) + 1] * len(original_word_tokens))

                if is_switched and self.r1_on_unswitched_tokens:
                    noised_r1_mask.extend([0] + [0] * (len(noised_word_tokens) - 1))
                else:
                    noised_r1_mask.extend([1] + [0] * (len(noised_word_tokens) - 1))

                if is_switched and self.r1_on_unswitched_tokens:
                    original_r1_mask.extend([0] + [0] * (len(original_word_tokens) - 1))
                else:
                    original_r1_mask.extend([1] + [0] * (len(original_word_tokens) - 1))

                break_flag = False
                if len(noised_tokens) >= self.noised_max_seq_length - special_tokens_count:
                    logger.info('truncate noised token {} {} {}'.format(len(noised_tokens), self.noised_max_seq_length,
                                                                        special_tokens_count))
                    noised_tokens = noised_tokens[:(self.noised_max_seq_length - special_tokens_count)]
                    noised_label_ids = noised_label_ids[:(self.noised_max_seq_length - special_tokens_count)]
                    noised_pooling_ids = noised_pooling_ids[:(self.noised_max_seq_length - special_tokens_count)]
                    noised_r1_mask = noised_r1_mask[:(self.noised_max_seq_length - special_tokens_count)]
                    break_flag = True

                if len(original_tokens) >= self.max_seq_length - special_tokens_count:
                    logger.info('truncate original token {} {} {}'.format(len(original_tokens), self.max_seq_length,
                                                                          special_tokens_count))
                    original_tokens = original_tokens[:(self.max_seq_length - special_tokens_count)]
                    original_label_ids = original_label_ids[:(self.max_seq_length - special_tokens_count)]
                    original_pooling_ids = original_pooling_ids[:(self.max_seq_length - special_tokens_count)]
                    original_r1_mask = original_r1_mask[:(self.max_seq_length - special_tokens_count)]
                    break_flag = True
                if break_flag:
                    break

            assert len(noised_tokens) <= self.noised_max_seq_length - special_tokens_count
            original_tokens += [self.sep_token]
            original_label_ids += [self.pad_token_label_id]
            original_pooling_ids += [0]
            original_r1_mask += [0]

            noised_tokens += [self.sep_token]
            noised_label_ids += [self.pad_token_label_id]
            noised_pooling_ids += [0]
            noised_r1_mask += [0]
            if self.sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                noised_tokens += [self.sep_token]
                noised_label_ids += [self.pad_token_label_id]
                noised_pooling_ids += [0]
                noised_r1_mask += [0]

                original_tokens += [self.sep_token]
                original_label_ids += [self.pad_token_label_id]
                original_pooling_ids += [0]
                original_r1_mask += [0]

            noised_segment_ids = [self.sequence_a_segment_id] * len(noised_tokens)
            original_segment_ids = [self.sequence_a_segment_id] * len(original_tokens)

            if self.cls_token_at_end:
                noised_tokens += [self.cls_token]
                noised_label_ids += [self.pad_token_label_id]
                noised_segment_ids += [self.cls_token_segment_id]
                noised_pooling_ids += [0]
                noised_r1_mask += [0]

                original_tokens += [self.cls_token]
                original_label_ids += [self.pad_token_label_id]
                original_segment_ids += [self.cls_token_segment_id]
                original_pooling_ids += [0]
                original_r1_mask += [0]
            else:
                noised_tokens = [self.cls_token] + noised_tokens
                noised_label_ids = [self.pad_token_label_id] + noised_label_ids
                noised_segment_ids = [self.cls_token_segment_id] + noised_segment_ids
                noised_pooling_ids = [0] + noised_pooling_ids
                noised_r1_mask = [0] + noised_r1_mask

                original_tokens = [self.cls_token] + original_tokens
                original_label_ids = [self.pad_token_label_id] + original_label_ids
                original_segment_ids = [self.cls_token_segment_id] + original_segment_ids
                original_pooling_ids = [0] + original_pooling_ids
                original_r1_mask = [0] + original_r1_mask

            noised_input_ids = self.tokenizer.convert_tokens_to_ids(noised_tokens)
            original_input_ids = self.tokenizer.convert_tokens_to_ids(original_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            noised_input_mask = [1 if self.mask_padding_with_zero else 0] * len(noised_input_ids)
            original_input_mask = [1 if self.mask_padding_with_zero else 0] * len(original_input_ids)

            # Zero-pad up to the sequence length.
            noised_padding_length = self.noised_max_seq_length - len(noised_input_ids)
            original_padding_length = self.max_seq_length - len(original_input_ids)
            if self.pad_on_left:
                noised_input_ids = [self.pad_token] * noised_padding_length + noised_input_ids
                noised_input_mask = [
                                        0 if self.mask_padding_with_zero else 1] * noised_padding_length + noised_input_mask
                noised_segment_ids = [self.pad_token_segment_id] * noised_padding_length + noised_segment_ids
                noised_label_ids = [self.pad_token_label_id] * noised_padding_length + noised_label_ids
                noised_pooling_ids = ([0] * noised_padding_length) + noised_pooling_ids
                noised_r1_mask = [0] * noised_padding_length + noised_r1_mask

                original_input_ids = [self.pad_token] * original_padding_length + original_input_ids
                original_input_mask = [
                                          0 if self.mask_padding_with_zero else 1] * original_padding_length + original_input_mask
                original_segment_ids = [self.pad_token_segment_id] * original_padding_length + original_segment_ids
                original_label_ids = [self.pad_token_label_id] * original_padding_length + original_label_ids
                original_pooling_ids = ([0] * original_padding_length) + original_pooling_ids
                original_r1_mask = [0] * original_padding_length + original_r1_mask
            else:
                noised_input_ids += [self.pad_token] * noised_padding_length
                noised_input_mask += [0 if self.mask_padding_with_zero else 1] * noised_padding_length
                noised_segment_ids += [self.pad_token_segment_id] * noised_padding_length
                noised_label_ids += [self.pad_token_label_id] * noised_padding_length
                noised_pooling_ids += ([0] * noised_padding_length)
                noised_r1_mask += [0] * noised_padding_length

                original_input_ids += [self.pad_token] * original_padding_length
                original_input_mask += [0 if self.mask_padding_with_zero else 1] * original_padding_length
                original_segment_ids += [self.pad_token_segment_id] * original_padding_length
                original_label_ids += [self.pad_token_label_id] * original_padding_length
                original_pooling_ids += ([0] * original_padding_length)
                original_r1_mask += [0] * original_padding_length

            assert sum(noised_r1_mask) == sum(original_r1_mask)
            assert len(noised_input_ids) == self.noised_max_seq_length
            assert len(noised_input_mask) == self.noised_max_seq_length
            assert len(noised_segment_ids) == self.noised_max_seq_length
            assert len(noised_label_ids) == self.noised_max_seq_length
            assert len(noised_pooling_ids) == self.noised_max_seq_length

            assert len(original_input_ids) == self.max_seq_length
            assert len(original_input_mask) == self.max_seq_length
            assert len(original_segment_ids) == self.max_seq_length
            assert len(original_label_ids) == self.max_seq_length
            assert len(original_pooling_ids) == self.max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("original_tokens: %s", " ".join([str(x) for x in original_tokens]))
                logger.info("original_input_ids: %s", " ".join([str(x) for x in original_input_ids]))
                logger.info("original_input_mask: %s", " ".join([str(x) for x in original_input_mask]))
                logger.info("original_segment_ids: %s", " ".join([str(x) for x in original_segment_ids]))
                logger.info("original_label_ids: %s", " ".join([str(x) for x in original_label_ids]))
                logger.info("original_pooling_ids: %s", " ".join([str(x) for x in original_pooling_ids]))
                logger.info("original_r1_mask: %s", " ".join([str(x) for x in original_r1_mask]))
                logger.info("noised_tokens: %s", " ".join([str(x) for x in noised_tokens]))
                logger.info("noised_input_ids: %s", " ".join([str(x) for x in noised_input_ids]))
                logger.info("noised_input_mask: %s", " ".join([str(x) for x in noised_input_mask]))
                logger.info("noised_segment_ids: %s", " ".join([str(x) for x in noised_segment_ids]))
                logger.info("noised_label_ids: %s", " ".join([str(x) for x in noised_label_ids]))
                logger.info("noised_pooling_ids: %s", " ".join([str(x) for x in noised_pooling_ids]))
                logger.info("noised_r1_mask: %s", " ".join([str(x) for x in noised_r1_mask]))

            all_noised_input_ids += [noised_input_ids]
            all_noised_input_mask += [noised_input_mask]
            all_noised_segment_ids += [noised_segment_ids]
            all_noised_label_ids += [noised_label_ids]
            all_noised_pooling_ids += [noised_pooling_ids]
            all_noised_r1_mask += [noised_r1_mask]

            all_original_input_ids += [original_input_ids]
            all_original_input_mask += [original_input_mask]
            all_original_segment_ids += [original_segment_ids]
            all_original_label_ids += [original_label_ids]
            all_original_pooling_ids += [original_pooling_ids]
            all_original_r1_mask += [original_r1_mask]
            all_is_augmented += [is_augmented[ex_index]]

        # Convert to Tensors and build dataset
        all_noised_input_ids = torch.tensor([input_ids for input_ids in all_noised_input_ids], dtype=torch.long)
        all_noised_input_mask = torch.tensor([input_mask for input_mask in all_noised_input_mask], dtype=torch.long)
        all_noised_segment_ids = torch.tensor([segment_ids for segment_ids in all_noised_segment_ids], dtype=torch.long)
        all_noised_label_ids = torch.tensor([label_ids for label_ids in all_noised_label_ids], dtype=torch.long)
        all_noised_pooling_ids = torch.tensor([pooling_ids for pooling_ids in all_noised_pooling_ids], dtype=torch.long)
        all_noised_r1_mask = torch.tensor([noised_r1_mask for noised_r1_mask in all_noised_r1_mask], dtype=torch.long)

        all_original_input_ids = torch.tensor([input_ids for input_ids in all_original_input_ids], dtype=torch.long)
        all_original_input_mask = torch.tensor([input_mask for input_mask in all_original_input_mask], dtype=torch.long)
        all_original_segment_ids = torch.tensor([segment_ids for segment_ids in all_original_segment_ids],
                                                dtype=torch.long)
        all_original_label_ids = torch.tensor([label_ids for label_ids in all_original_label_ids], dtype=torch.long)
        all_original_pooling_ids = torch.tensor([pooling_ids for pooling_ids in all_original_pooling_ids],
                                                dtype=torch.long)
        all_original_r1_mask = torch.tensor([original_r1_mask for original_r1_mask in all_original_r1_mask],
                                            dtype=torch.long)
        all_is_augmented = torch.tensor([is_augmented for is_augmented in all_is_augmented])

        # print(all_noised_r1_mask.sum(), all_original_r1_mask.sum())
        assert all_noised_r1_mask.sum() == all_original_r1_mask.sum()

        dataset = TensorDataset(all_original_input_ids, all_original_input_mask, all_original_segment_ids,
                                all_original_label_ids, all_original_pooling_ids,
                                all_noised_input_ids, all_noised_input_mask, all_noised_segment_ids,
                                all_noised_label_ids, all_noised_pooling_ids, all_noised_r1_mask,
                                all_original_r1_mask, all_is_augmented)

        return dataset

    def load_translate_data(self, examples):
        n_unfound = 0
        translate_examples = []
        all_align_pooling_ids = []
        for ex_idx, example in enumerate(examples):
            src_sent = " ".join(example.words)
            if src_sent not in self.src2tgt:
                logger.info("sentence || {} || is not found in translate data".format(src_sent))
                tgt_sent = src_sent
                tgt_lang = "en"
                align_pooling_id = (
                    list(range(1, len(src_sent.split(" ")) + 1)), list(range(1, len(src_sent.split(" ")) + 1)))
                n_unfound += 1
            else:
                # assert src_sent in self.src2tgt
                idx = random.randint(0, len(self.src2tgt[src_sent]) - 1)
                tgt_lang, tgt_sent, align_pooling_id = self.src2tgt[src_sent][idx]

            words = tgt_sent.split(" ")
            # print(len(words))
            labels = ['O'] * len(words)
            translate_examples.append(InputExample(ex_idx, words, labels, langs=tgt_lang))
            # print(align_pooling_id)
            all_align_pooling_ids.append(align_pooling_id)

        print("{} sentences unfound.".format(n_unfound))
        features = convert_examples_to_features(translate_examples, self.label_list, self.max_seq_length,
                                                self.tokenizer,
                                                cls_token_at_end=self.cls_token_at_end,
                                                cls_token=self.cls_token,
                                                cls_token_segment_id=self.cls_token_segment_id,
                                                sep_token=self.sep_token,
                                                sep_token_extra=self.sep_token_extra,
                                                pad_on_left=self.pad_on_left,
                                                pad_token=self.pad_token,
                                                pad_token_segment_id=self.pad_token_segment_id,
                                                pad_token_label_id=self.pad_token_label_id,
                                                lang=None)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_pooling_ids = torch.tensor([f.pooling_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        # not used under this setting
        all_noised_r1_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_original_r1_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pooling_ids,
                                all_noised_r1_mask, all_original_r1_mask, all_label_ids)

        return dataset, all_align_pooling_ids

    def load_translate_data_by_batch(self, examples, train_batch_size):
        translate_languagse = self.translate_languages
        language_cnt = [0] * len(translate_languagse)
        pass

    def get_train_steps(self, examples, args):
        n_augment_examples = math.ceil(len(examples) * (1 + self.augment_ratio))

        augment_steps = math.ceil(n_augment_examples / args.train_batch_size) // args.gradient_accumulation_steps

        if args.max_steps > 0:
            t_total = args.max_steps
            assert False
        else:
            t_total = augment_steps * args.num_train_epochs
        return t_total


def train(args, train_examples, train_dataset, model, first_stage_model, tokenizer, labels, pad_token_label_id,
          noised_data_generator=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
        tb_log_dir = os.getenv("PHILLY_JOB_DIRECTORY", None)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'w')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if noised_data_generator is not None and noised_data_generator.enable_data_augmentation:
        t_total = noised_data_generator.get_train_steps(train_examples, args)
    else:
        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # args.warmup_steps == -1 means 0.1 warmup ratio
    if args.warmup_steps == -1:
        args.warmup_steps = int(t_total * 0.1)
    logger.info("Warmup steps: %d" % args.warmup_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    best_score = 0.0
    best_checkpoint = None
    patience = 0
    global_step = 0

    tr_loss, logging_loss, best_avg = 0.0, 0.0, 0.0
    tr_original_loss, logging_original_loss = 0.0, 0.0
    tr_noised_loss, logging_noised_loss = 0.0, 0.0
    tr_r1_loss, logging_r1_loss = 0.0, 0.0
    tr_r2_loss, logging_r2_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Add here for reproductibility (even between python 2 and 3)

    def logging(eval=False):
        results = None
        # Only evaluate when single GPU otherwise metrics may not average well
        if args.local_rank in [-1, 0] and args.evaluate_during_training and eval:
            results = evaluate(args, model, tokenizer, labels, pad_token_label_id)
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

        return results

    for _ in train_iterator:
        if noised_data_generator is not None:
            assert noised_data_generator.enable_r1_loss or noised_data_generator.noised_loss or noised_data_generator.enable_data_augmentation

            noised_train_dataset, all_align_pooling_ids = noised_data_generator.get_noised_dataset(train_examples)

            train_sampler = RandomSampler(noised_train_dataset) if args.local_rank == -1 else DistributedSampler(
                noised_train_dataset)
            train_dataloader = DataLoader(noised_train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            # if not args.max_steps > 0:
            #     assert t_total == len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if first_stage_model is not None:
                first_stage_model.eval()
            batch = tuple(t.to(args.device) for t in batch if t is not None)
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3],
                      "pooling_ids": batch[4]}

            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None

            # if args.model_type == "xlm":
            #     inputs["langs"] = batch[5]

            if first_stage_model is not None:
                with torch.no_grad():
                    inputs["first_stage_model_logits"] = first_stage_model(**inputs)[1]

            # if noised_data_generator is not None and noised_data_generator.enable_r1_loss and \
            #         noised_data_generator.enable_translate_data and noised_data_generator.use_align_label_probs:
            #     inputs.update({"src_pooling_ids": batch[-2],
            #                    "tgt_pooling_ids": batch[-1]})
            #     batch = batch[:-2]

            if noised_data_generator is not None:
                inputs.update({"noised_input_ids": batch[5],
                               "noised_attention_mask": batch[6],
                               "noised_token_type_ids": None,
                               "noised_labels": batch[8],
                               "noised_pooling_ids": batch[9],
                               "noised_r1_mask": batch[10],
                               "original_r1_mask": batch[11],
                               "is_augmented": batch[12]})

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel training
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if noised_data_generator is not None:
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

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    do_eval = args.evaluate_steps > 0 and global_step % args.evaluate_steps == 0
                    cur_result = logging(do_eval)
                    logging_loss = tr_loss
                    logging_original_loss = tr_original_loss
                    logging_noised_loss = tr_noised_loss
                    logging_r1_loss = tr_r1_loss
                    logging_r2_loss = tr_r2_loss

                    if do_eval:
                        print(cur_result)
                        if cur_result["dev_avg"]["f1"] > best_score:
                            logger.info(
                                "result['f1']={} > best_score={}".format(cur_result["dev_avg"]["f1"], best_score))
                            best_score = cur_result["dev_avg"]["f1"]
                            # Save the best model checkpoint
                            output_dir = os.path.join(args.output_dir, "checkpoint-best")
                            best_checkpoint = output_dir
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            # Take care of distributed/parallel training
                            model_to_save = model.module if hasattr(model, "module") else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))
                            logger.info("Saving the best model checkpoint to %s", output_dir)

                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to %s", output_dir)
                            logger.info("Reset patience to 0")
                            patience = 0
                        else:
                            patience += 1
                            logger.info("Hit patience={}".format(patience))
                            if args.eval_patience > 0 and patience > args.eval_patience:
                                logger.info("early stop! patience={}".format(patience))
                                epoch_iterator.close()
                                train_iterator.close()
                                if args.local_rank in [-1, 0]:
                                    tb_writer.close()
                                    log_writer.close()
                                return global_step, tr_loss / global_step

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        log_writer.close()

    return global_step, tr_loss / global_step


def predict(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", print_result=True):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3],
                      "pooling_ids": batch[4]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
            if args.model_type == 'xlm':
                inputs["langs"] = batch[5]
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel evaluating
                tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    if nb_eval_steps == 0:
        results = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
    else:
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=2)

        label_map = {i: label for i, label in enumerate(labels)}

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        results = {
            "loss": eval_loss,
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list)
        }

    if print_result:
        logger.info("***** Evaluation result %s in %s *****" % (prefix, lang))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def evaluate(args, model, tokenizer, labels, pad_token_label_id, prefix=""):
    # eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    eval_datasets = []
    eval_langs = args.predict_langs.split(',')

    splits = ["dev", "test"] if args.do_train else ["test"]
    for split in splits:
        for lang in eval_langs:
            eval_datasets.append((split, lang))

    all_languages_results = {}
    # leave interface for multi-task evaluation
    # eval_task = eval_task_names[0]
    eval_output_dir = eval_outputs_dirs[0]

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split, lang in eval_datasets:
        task_name = "{0}-{1}".format(split, lang)
        eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=split, lang=lang)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3],
                          "pooling_ids": batch[4]}
                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None
                if args.model_type == 'xlm':
                    inputs["langs"] = batch[5]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                if args.n_gpu > 1:
                    # mean() to average on multi-gpu parallel evaluating
                    tmp_eval_loss = tmp_eval_loss.mean()

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        if nb_eval_steps == 0:
            results = {k: 0 for k in ["precision", "recall", "f1"]}
            continue
        else:
            eval_loss = eval_loss / nb_eval_steps
            preds = np.argmax(preds, axis=2)

            label_map = {i: label for i, label in enumerate(labels)}

            out_label_list = [[] for _ in range(out_label_ids.shape[0])]
            preds_list = [[] for _ in range(out_label_ids.shape[0])]

            for i in range(out_label_ids.shape[0]):
                for j in range(out_label_ids.shape[1]):
                    if out_label_ids[i, j] != pad_token_label_id:
                        out_label_list[i].append(label_map[out_label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])

            results = {
                "precision": precision_score(out_label_list, preds_list),
                "recall": recall_score(out_label_list, preds_list),
                "f1": f1_score(out_label_list, preds_list)
            }

        all_languages_results["{0}_{1}".format(split, lang)] = results

    for split in splits:
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


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, few_shot=-1,
                            return_examples=False):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    model_name = "xlm-roberta-base"
    if args.word_dropout_rate > 0:
        assert mode != "train"
        cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}_wdr{}".format(mode, lang,
                                                                                             model_name,
                                                                                             str(args.max_seq_length),
                                                                                             str(
                                                                                                 args.word_dropout_rate)))
    else:
        cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
                                                                                       model_name,
                                                                                       str(args.max_seq_length)))

    cached_features_file += "_pooling"

    if args.languages_without_spaces is not None and lang in args.languages_without_spaces.split(','):
        cached_features_file += "_lws"

    data_file = os.path.join(args.data_dir, lang, "{}.{}".format(mode, model_name))
    logger.info("Creating features from dataset file at {} in language {}".format(data_file, lang))
    examples = read_examples_from_file(data_file, lang)

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("all languages = {}".format(lang))
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                    0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id,
                                                lang=lang,
                                                languages_without_spaces=args.languages_without_spaces.split(
                                                    ',') if args.languages_without_spaces is not None else None,
                                                word_dropout_rate=args.word_dropout_rate,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info(
                "Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_pooling_ids = torch.tensor([f.pooling_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if args.model_type == 'xlm' and features[0].langs is not None:
        all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
        logger.info('all_langs[0] = {}'.format(all_langs[0]))
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pooling_ids,
                                all_langs)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pooling_ids)
    if return_examples:
        return dataset, examples
    else:
        return dataset


def ConcatDataset(dataset_list):
    all_input_ids = torch.cat([dataset.tensors[0] for dataset in dataset_list], dim=0)
    all_input_mask = torch.cat([dataset.tensors[1] for dataset in dataset_list], dim=0)
    all_segment_ids = torch.cat([dataset.tensors[2] for dataset in dataset_list], dim=0)
    all_label_ids = torch.cat([dataset.tensors[3] for dataset in dataset_list], dim=0)
    all_pooling_ids = torch.cat([dataset.tensors[4] for dataset in dataset_list], dim=0)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_pooling_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the NER/POS task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # stable fine-tuning paramters
    parser.add_argument("--overall_ratio", default=1.0, type=float, help="overall ratio")
    parser.add_argument("--enable_r1_loss", action="store_true", help="Whether to enable r1 loss.")
    parser.add_argument("--disable_backward_kl", action="store_true", help="Whether to disable backward kl loss.")
    parser.add_argument("--r1_lambda", default=5.0, type=float, help="lambda of r1 loss")
    parser.add_argument("--r1_on_unswitched_tokens", action="store_true",
                        help="Whether to enable r1 loss only on unswitched tokens.")
    parser.add_argument("--original_loss", action="store_true",
                        help="Whether to use cross entropy loss on the former example.")
    parser.add_argument("--noised_loss", action="store_true",
                        help="Whether to use cross entropy loss on the latter example.")
    parser.add_argument("--noised_max_seq_length", default=256, type=int, help="noised max sequence length")
    parser.add_argument("--enable_bpe_switch", action="store_true", help="Whether to enable bpe-switch.")
    parser.add_argument("--bpe_switch_ratio", default=0.5, type=float, help="bpe_switch_ratio")
    parser.add_argument("--tokenizer_dir", default=None, type=str, help="tokenizer dir")
    parser.add_argument("--tokenizer_languages", default=None, type=str, help="tokenizer languages")
    parser.add_argument("--enable_bpe_sampling", action="store_true", help="Whether to enable bpe sampling.")
    parser.add_argument("--bpe_sampling_ratio", default=0.5, type=float, help="bpe_sampling_ratio")
    parser.add_argument("--sampling_alpha", default=5.0, type=float, help="alpha of sentencepiece sampling")
    parser.add_argument("--sampling_nbest_size", default=-1, type=int, help="nbest_size of sentencepiece sampling")
    parser.add_argument("--enable_random_noise", action="store_true", help="Whether to enable random noise.")
    parser.add_argument("--detach_embeds", action="store_true", help="Whether to detach noised embeddings.")
    parser.add_argument("--noise_eps", default=1e-5, type=float, help="noise eps")
    parser.add_argument('--noise_type', type=str, default='uniform',
                        choices=['normal', 'uniform'],
                        help='type of noises for RXF methods')
    parser.add_argument("--enable_code_switch", action="store_true", help="Whether to enable code switch.")
    parser.add_argument("--code_switch_ratio", default=0.5, type=float, help="code_switch_ratio")
    parser.add_argument("--dict_dir", default=None, type=str, help="dict dir")
    parser.add_argument("--dict_languages", default=None, type=str, help="dict languages")
    parser.add_argument("--hidden_dropout_prob", default=None, type=float, help="hidden_dropout_prob")
    parser.add_argument("--attention_probs_dropout_prob", default=None, type=float, help="attention_probs_dropout_prob")
    parser.add_argument("--use_pooling_strategy", action="store_true", help="Whether to use pooling strategy.")
    parser.add_argument("--use_sentence_label_probs", action="store_true",
                        help="Whether to use r1 loss on sentence-level label probs.")
    parser.add_argument("--use_token_label_probs", action="store_true",
                        help="Whether to use r1 loss on token-level label probs.")
    parser.add_argument("--use_average_representations", action="store_true",
                        help="Whether to use average representation.")
    parser.add_argument("--translation_path", default=None, type=str, help="path to translation")
    parser.add_argument("--translate_languages", default=None, type=str, help="translate languages")
    parser.add_argument("--languages_without_spaces", default=None, type=str, help="languages without spaces")
    parser.add_argument("--use_align_label_probs", action="store_true",
                        help="Whether to use r1 loss on align label probs.")
    parser.add_argument("--enable_data_augmentation", action="store_true", help="Whether to enable data augmentation.")
    parser.add_argument("--augment_ratio", default=1.0, type=float, help="augmentation ratio.")
    parser.add_argument("--augment_method", default=None, type=str, required=False,
                        help="augment method")
    parser.add_argument("--first_stage_model_path", default=None, type=str, required=False,
                        help="stable model path")
    parser.add_argument("--r2_lambda", default=1.0, type=float, required=False,
                        help="r2_lambda")
    parser.add_argument("--use_hard_labels", action="store_true", help="Whether to use hard labels.")
    parser.add_argument("--word_dropout_rate", default=0.0, type=float, required=False, help="test word dropout rate")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_predict_dev", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--few_shot", default=-1, type=int,
                        help="num of few-shot exampes")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--evaluate_steps", type=int, default=0, help="Log every X updates steps.")
    parser.add_argument("--save_only_best_checkpoint", action="store_true",
                        help="Save only the best checkpoint during training")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--predict_langs", type=str, default="en", help="prediction languages")
    parser.add_argument("--train_langs", default="en", type=str,
                        help="The languages in the training sets.")
    parser.add_argument("--log_file", type=str, default=None, help="log file")
    parser.add_argument("--eval_patience", type=int, default=-1,
                        help="wait N times of decreasing dev score before early stop during training")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    # logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
    #                     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    #                     datefmt='%m/%d/%Y %H:%M:%S',
    #                     level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logging.info("Input args: %r" % args)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare NER/POS task
    labels = get_labels(args.labels)
    logger.info(labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = args.hidden_dropout_prob
    if args.attention_probs_dropout_prob is not None:
        config.attention_probs_dropout_prob = args.attention_probs_dropout_prob

    if args.noised_loss or args.enable_r1_loss or args.enable_data_augmentation:
        noised_data_generator = NoisedDataGenerator(
            label_list=labels,
            pad_token_label_id=pad_token_label_id,
            r1_lambda=args.r1_lambda,
            r1_on_unswitched_tokens=args.r1_on_unswitched_tokens,
            enable_r1_loss=args.enable_r1_loss,
            disable_backward_kl=args.disable_backward_kl,
            use_sentence_label_probs=args.use_sentence_label_probs,
            use_token_label_probs=args.use_token_label_probs,
            original_loss=args.original_loss,
            noised_loss=args.noised_loss,
            max_seq_length=args.max_seq_length,
            noised_max_seq_length=args.noised_max_seq_length,
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
            detach_embeds=args.detach_embeds,
            noise_eps=args.noise_eps,
            noise_type=args.noise_type,
            enable_code_switch=args.enable_code_switch,
            code_switch_ratio=args.code_switch_ratio,
            dict_dir=args.dict_dir,
            dict_languages=args.dict_languages.split(',') if args.dict_languages is not None else [],
            use_average_representations=args.use_average_representations,
            translation_path=args.translation_path,
            translate_languages=args.translate_languages.split(
                ',') if args.translate_languages is not None else args.predict_langs.split(','),
            use_align_label_probs=args.use_align_label_probs,
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
                                                   config=config,
                                                   use_pooling_strategy=args.use_pooling_strategy, )
    else:
        first_stage_model = None

    if args.init_checkpoint:
        logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
        model = model_class.from_pretrained(args.init_checkpoint,
                                            config=config,
                                            noised_data_generator=noised_data_generator,
                                            use_pooling_strategy=args.use_pooling_strategy,
                                            cache_dir=args.init_checkpoint)
    else:
        logger.info("loading from cached model = {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config,
                                            noised_data_generator=noised_data_generator,
                                            use_pooling_strategy=args.use_pooling_strategy,
                                            cache_dir=args.cache_dir if args.cache_dir else None)

    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    if first_stage_model is not None:
        first_stage_model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        train_langs = args.train_langs.split(',')

        dataset_list = []
        train_examples = []

        for lang in train_langs:
            lg_train_dataset, lg_train_examples = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id,
                                                                          mode="train", lang=lang,
                                                                          few_shot=args.few_shot, return_examples=True)
            dataset_list.append(lg_train_dataset)
            train_examples += lg_train_examples
        train_dataset = ConcatDataset(dataset_list)
        global_step, tr_loss = train(args, train_examples, train_dataset, model, first_stage_model, tokenizer, labels,
                                     pad_token_label_id, noised_data_generator=noised_data_generator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use default names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Initialization for evaluation
    results = {}
    if args.init_checkpoint:
        best_checkpoint = args.init_checkpoint
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        checkpoints = [best_checkpoint]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, use_pooling_strategy=args.use_pooling_strategy)
            model.to(args.device)
            results = evaluate(args, model, tokenizer, labels, pad_token_label_id)
            for key, value in results.items():
                logger.info("eval_{}: {}".format(key, value))
            log_writer = open(os.path.join(args.output_dir, "evaluate_wdr{}_logs.txt".format(args.word_dropout_rate)),
                              'w')
            log_writer.write("{0}\t{1}".format("evaluate", json.dumps(results)) + '\n')
            exit(0)

            result, _ = predict(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step,
                                lang=args.train_langs)
            if result["f1"] > best_f1:
                best_checkpoint = checkpoint
                best_f1 = result["f1"]
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint, use_pooling_strategy=args.use_pooling_strategy)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "a") as result_writer:
            for lang in args.predict_langs.split(','):
                if not os.path.exists(os.path.join(args.data_dir, lang, 'test.{}'.format(args.model_name_or_path))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = predict(args, model, tokenizer, labels, pad_token_label_id, mode="test",
                                              lang=lang)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                # Save predictions
                output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
                infile = os.path.join(args.data_dir, lang, "test.{}".format(args.model_name_or_path))
                idxfile = infile + '.idx'
                save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)

    # Predict dev set
    if args.do_predict_dev and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint, use_pooling_strategy=args.use_pooling_strategy)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "dev_results.txt")
        with open(output_test_results_file, "w") as result_writer:
            for lang in args.predict_langs.split(','):
                if not os.path.exists(os.path.join(args.data_dir, lang, 'dev.{}'.format(args.model_name_or_path))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = predict(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                              lang=lang)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                # Save predictions
                output_test_predictions_file = os.path.join(args.output_dir, "dev_{}_predictions.txt".format(lang))
                infile = os.path.join(args.data_dir, lang, "dev.{}".format(args.model_name_or_path))
                idxfile = infile + '.idx'
                save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)


def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
    # Save predictions
    with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
        text = text_reader.readlines()
        index = idx_reader.readlines()
        assert len(text) == len(index)

    # Sanity check on the predictions
    with open(output_file, "w") as writer:
        example_id = 0
        prev_id = int(index[0])
        for line, idx in zip(text, index):
            if line == "" or line == "\n":
                example_id += 1
            else:
                cur_id = int(idx)
                output_line = '\n' if cur_id != prev_id else ''
                if output_word_prediction:
                    output_line += line.split()[0] + '\t'
                output_line += predictions[example_id].pop(0) + '\n'
                writer.write(output_line)
                prev_id = cur_id


if __name__ == "__main__":
    main()
