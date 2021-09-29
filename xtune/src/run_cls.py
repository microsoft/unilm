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
""" Finetuning multi-lingual models on classification (Bert, DistilBERT, XLM, XLM-R). Adapted from `examples/run_glue.py`"""

import argparse
import glob
import logging
import os
import random
import json
import copy
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassificationStable,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import xtreme_convert_examples_to_features as convert_examples_to_features
from transformers import xtreme_compute_metrics as compute_metrics
from transformers import xtreme_output_modes as output_modes
from transformers import xtreme_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in
     (BertConfig, DistilBertConfig, XLMConfig, XLMRobertaConfig)), ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassificationStable, XLMRobertaTokenizer)
}


class NoisedDataGenerator(object):
    def __init__(self,
                 task_name="xnli",
                 enable_r1_loss=False,
                 r1_lambda=5.0,
                 original_loss=True,
                 noised_loss=False,
                 max_length=512,
                 overall_ratio=1.0,
                 enable_bpe_switch=False,
                 bpe_switch_ratio=0.5,
                 tokenizer_dir=None,
                 do_lower_case=False,
                 tokenizer_languages=None,
                 enable_bpe_sampling=False,
                 tokenizer=None,
                 bpe_sampling_ratio=0.5,
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
                 enable_word_dropout=False,
                 word_dropout_rate=0.1,
                 enable_translate_data=False,
                 translation_path=None,
                 train_language=None,
                 data_dir=None,
                 translate_different_pair=False,
                 translate_en_data=False,
                 enable_data_augmentation=False,
                 augment_method=None,
                 augment_ratio=0.0,
                 r2_lambda=1.0,
                 use_hard_labels=False):
        if enable_code_switch:
            assert dict_dir is not None
            assert dict_languages is not None
        assert tokenizer is not None
        if enable_random_noise:
            assert noise_type in ['uniform', 'normal']

        self.task_name = task_name
        self.n_tokens = 0
        self.n_cs_tokens = 0
        self.enable_r1_loss = enable_r1_loss
        self.r1_lambda = r1_lambda
        self.original_loss = original_loss
        self.noised_loss = noised_loss
        self.max_length = max_length
        self.overall_ratio = overall_ratio

        self.enable_bpe_switch = enable_bpe_switch
        self.bpe_switch_ratio = bpe_switch_ratio / self.overall_ratio
        assert self.bpe_switch_ratio <= 1.0
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer_languages = tokenizer_languages

        self.enable_bpe_sampling = enable_bpe_sampling
        self.bpe_sampling_ratio = bpe_sampling_ratio / self.overall_ratio
        assert self.bpe_sampling_ratio <= 1.0
        self.tokenizer = tokenizer
        self.sampling_alpha = sampling_alpha
        self.sampling_nbest_size = sampling_nbest_size

        self.enable_random_noise = enable_random_noise
        self.noise_detach_embeds = noise_detach_embeds
        self.noise_eps = noise_eps
        self.noise_type = noise_type

        self.enable_word_dropout = enable_word_dropout
        self.word_dropout_rate = word_dropout_rate

        self.enable_translate_data = enable_translate_data
        self.train_languages = train_language.split(',')
        self.data_dir = data_dir
        self.translate_different_pair = translate_different_pair
        self.translate_en_data = translate_en_data

        if "en" in self.train_languages:
            self.train_languages.remove("en")
        self.translate_train_dicts = []
        self.tgt2src_dict = {}
        self.tgt2src_cnt = {}
        self.translation_path = translation_path
        self.enable_code_switch = enable_code_switch
        self.code_switch_ratio = code_switch_ratio / self.overall_ratio
        assert self.code_switch_ratio <= 1.0
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

        self.enable_data_augmentation = enable_data_augmentation
        self.augment_method = augment_method
        self.augment_ratio = augment_ratio
        self.r2_lambda = r2_lambda
        self.use_hard_labels = use_hard_labels

    def augment_examples(self, examples):
        n_augment = math.ceil(len(examples) * self.augment_ratio)

        augment_examples = []

        while n_augment > 0:
            examples = copy.deepcopy(examples)
            augment_examples += examples[:n_augment]
            n_augment -= len(examples[:n_augment])
            random.shuffle(examples)

        return augment_examples

    def get_noised_dataset(self, examples):
        # maybe do not save augmented examples
        examples = copy.deepcopy(examples)

        if (self.enable_data_augmentation and self.augment_method == "mt") or self.enable_translate_data:
            self.load_translate_data()

        is_augmented = [0] * len(examples)
        if self.enable_data_augmentation:
            augment_examples = self.augment_examples(examples)
            is_augmented += [1] * len(augment_examples)
            examples += augment_examples

        if self.enable_code_switch:
            self.n_tokens = 0
            self.n_cs_tokens = 0

        dataset = self.convert_examples_to_dataset(examples, is_augmented)

        if self.enable_code_switch:
            logger.info("{:.2f}% tokens have been code-switched.".format(self.n_cs_tokens / self.n_tokens * 100))
        return dataset

    def encode_sentence(self, text, switch_text=False, enable_code_switch=False, enable_bpe_switch=False,
                        enable_bpe_sampling=False, enable_word_dropout=False, ):
        if text is None:
            return None
        ids = []
        tokens = text.split(" ")
        for token in tokens:
            switch_token = random.random() <= self.overall_ratio
            self.n_tokens += 1
            if enable_code_switch and switch_text and switch_token and random.random() <= self.code_switch_ratio:
                lang = self.dict_languages[random.randint(0, len(self.dict_languages) - 1)]
                if token.lower() in self.lang2dict[lang]:
                    self.n_cs_tokens += 1
                    token = self.lang2dict[lang][token.lower()][
                        random.randint(0, len(self.lang2dict[lang][token.lower()]) - 1)]

            if enable_bpe_switch and switch_text and switch_token and random.random() <= self.bpe_switch_ratio:
                lang = self.tokenizer_languages[random.randint(0, len(self.tokenizer_languages) - 1)]
                tokenizer = self.lang2tokenizer[lang]
            else:
                tokenizer = self.tokenizer

            if enable_bpe_sampling and switch_text and switch_token and random.random() <= self.bpe_sampling_ratio:
                token_ids = tokenizer.encode_plus(token, add_special_tokens=True,
                                                  nbest_size=self.sampling_nbest_size,
                                                  alpha=self.sampling_alpha)["input_ids"]
            else:
                token_ids = tokenizer.encode_plus(token, add_special_tokens=True)["input_ids"]

            if enable_word_dropout:
                for token_id in token_ids[1:-1]:
                    if random.random() <= self.word_dropout_rate:
                        ids += [tokenizer.unk_token_id]
                    else:
                        ids += [token_id]
            else:
                ids += token_ids[1:-1]
        return ids

    def encode_plus(self, text_a, text_b, switch_text=False, enable_code_switch=False, enable_bpe_switch=False,
                    enable_bpe_sampling=False, enable_word_dropout=False, ):
        # switch all sentences
        ids = self.encode_sentence(text_a, switch_text, enable_code_switch, enable_bpe_switch, enable_bpe_sampling,
                                   enable_word_dropout)
        pair_ids = self.encode_sentence(text_b, switch_text, enable_code_switch, enable_bpe_switch, enable_bpe_sampling,
                                        enable_word_dropout)

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}

        # Handle max sequence length
        total_len = len_ids + len_pair_ids + (self.tokenizer.num_added_tokens(pair=pair))
        if self.max_length and total_len > self.max_length:
            ids, pair_ids, overflowing_tokens = self.tokenizer.truncate_sequences(
                ids,
                pair_ids=pair_ids,
                num_tokens_to_remove=total_len - self.max_length,
                truncation_strategy="longest_first",
                stride=0,
            )

        # Handle special_tokens
        sequence = self.tokenizer.build_inputs_with_special_tokens(ids, pair_ids)
        token_type_ids = self.tokenizer.create_token_type_ids_from_sequences(ids, pair_ids)

        encoded_inputs["input_ids"] = sequence
        encoded_inputs["token_type_ids"] = token_type_ids

        return encoded_inputs

    def convert_examples_to_dataset(
            self,
            examples,
            is_augmented=None,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True
    ):

        processor = processors[self.task_name](language="en", train_language="en")
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, self.task_name))
        label_map = {label: i for i, label in enumerate(label_list)}

        output_mode = output_modes[self.task_name]
        logger.info("Using output mode %s for task %s" % (output_mode, self.task_name))

        all_original_input_ids = []
        all_original_attention_mask = []
        all_original_token_type_ids = []
        all_labels = []

        all_noised_input_ids = []
        all_noised_attention_mask = []
        all_noised_token_type_ids = []

        all_r1_mask = []
        all_is_augmented = []

        for (ex_index, example) in enumerate(examples):
            len_examples = len(examples)
            if ex_index % 10000 == 0:
                logger.info("Writing example %d/%d" % (ex_index, len_examples))
                # if ex_index == 10000: break

            if is_augmented[ex_index]:
                if self.augment_method == "mt":
                    example.text_a, example.text_b = self.get_translation_pair(example.text_a, example.text_b)
                    original_inputs = self.encode_plus(example.text_a, example.text_b, switch_text=False)
                    all_r1_mask.append(1)
                elif self.augment_method == "gn":
                    original_inputs = self.encode_plus(example.text_a, example.text_b, switch_text=False)
                    all_r1_mask.append(1)
                elif self.augment_method == "cs":
                    original_inputs = self.encode_plus(example.text_a, example.text_b, switch_text=True,
                                                       enable_code_switch=True)
                    all_r1_mask.append(1)
                elif self.augment_method == "ss":
                    original_inputs = self.encode_plus(example.text_a, example.text_b, switch_text=True,
                                                       enable_bpe_sampling=True)
                    all_r1_mask.append(1)
                else:
                    assert False
            else:
                original_inputs = self.encode_plus(example.text_a, example.text_b, switch_text=False)
                all_r1_mask.append(1)

            all_is_augmented.append(is_augmented[ex_index])

            original_input_ids, original_token_type_ids = original_inputs["input_ids"], original_inputs[
                "token_type_ids"]

            original_attention_mask = [1 if mask_padding_with_zero else 0] * len(original_input_ids)

            original_padding_length = self.max_length - len(original_input_ids)

            if pad_on_left:
                original_input_ids = ([pad_token] * original_padding_length) + original_input_ids
                original_attention_mask = ([0 if mask_padding_with_zero else 1] * original_padding_length) + \
                                          original_attention_mask
                original_token_type_ids = ([pad_token_segment_id] * original_padding_length) + original_token_type_ids
            else:
                original_input_ids = original_input_ids + ([pad_token] * original_padding_length)
                original_attention_mask = original_attention_mask + (
                        [0 if mask_padding_with_zero else 1] * original_padding_length)
                original_token_type_ids = original_token_type_ids + ([pad_token_segment_id] * original_padding_length)

            assert len(original_input_ids) == self.max_length, "Error with input length {} vs {}".format(
                len(original_input_ids), self.max_length)
            assert len(original_attention_mask) == self.max_length, "Error with input length {} vs {}".format(
                len(original_attention_mask), self.max_length)
            assert len(original_token_type_ids) == self.max_length, "Error with input length {} vs {}".format(
                len(original_token_type_ids), self.max_length)

            if output_mode == "classification":
                label = label_map[example.label]
            elif output_mode == "regression":
                label = float(example.label)
            else:
                raise KeyError(output_mode)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("original text a: %s" % (example.text_a))
                logger.info("original text b: %s" % (example.text_b))
                logger.info("original_input_ids: %s" % " ".join([str(x) for x in original_input_ids]))
                logger.info("original_attention_mask: %s" % " ".join([str(x) for x in original_attention_mask]))
                logger.info("original_token_type_ids: %s" % " ".join([str(x) for x in original_token_type_ids]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            all_original_input_ids.append(original_input_ids)
            all_original_attention_mask.append(original_attention_mask)
            all_original_token_type_ids.append(original_token_type_ids)
            all_labels.append(label)

            if not self.enable_r1_loss:
                continue

            if self.enable_translate_data:
                noised_text_a, noised_text_b = self.get_translation_pair(example.text_a, example.text_b)
            else:
                noised_text_a, noised_text_b = example.text_a, example.text_b

            noised_inputs = self.encode_plus(noised_text_a, noised_text_b, switch_text=True,
                                             enable_code_switch=self.enable_code_switch,
                                             enable_bpe_switch=self.enable_bpe_switch,
                                             enable_bpe_sampling=self.enable_bpe_sampling,
                                             enable_word_dropout=self.enable_word_dropout)
            noised_input_ids, noised_token_type_ids = noised_inputs["input_ids"], noised_inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.

            noised_attention_mask = [1 if mask_padding_with_zero else 0] * len(noised_input_ids)

            # Zero-pad up to the sequence length.

            noised_padding_length = self.max_length - len(noised_input_ids)
            if pad_on_left:
                noised_input_ids = ([pad_token] * noised_padding_length) + noised_input_ids
                noised_attention_mask = ([0 if mask_padding_with_zero else 1] * noised_padding_length) + \
                                        noised_attention_mask
                noised_token_type_ids = ([pad_token_segment_id] * noised_padding_length) + noised_token_type_ids
            else:
                noised_input_ids = noised_input_ids + ([pad_token] * noised_padding_length)
                noised_attention_mask = noised_attention_mask + (
                        [0 if mask_padding_with_zero else 1] * noised_padding_length)
                noised_token_type_ids = noised_token_type_ids + ([pad_token_segment_id] * noised_padding_length)

            assert len(noised_input_ids) == self.max_length, "Error with input length {} vs {}".format(
                len(noised_input_ids), self.max_length)
            assert len(noised_attention_mask) == self.max_length, "Error with input length {} vs {}".format(
                len(noised_attention_mask), self.max_length)
            assert len(noised_token_type_ids) == self.max_length, "Error with input length {} vs {}".format(
                len(noised_token_type_ids), self.max_length)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("noised text a: %s" % (noised_text_a))
                logger.info("noised text b: %s" % (noised_text_b))
                logger.info("noised_input_ids: %s" % " ".join([str(x) for x in noised_input_ids]))
                logger.info("noised_attention_mask: %s" % " ".join([str(x) for x in noised_attention_mask]))
                logger.info("noised_token_type_ids: %s" % " ".join([str(x) for x in noised_token_type_ids]))

            all_noised_input_ids.append(noised_input_ids)
            all_noised_attention_mask.append(noised_attention_mask)
            all_noised_token_type_ids.append(noised_token_type_ids)

        all_original_input_ids = torch.tensor([input_ids for input_ids in all_original_input_ids], dtype=torch.long)
        all_original_attention_mask = torch.tensor([attention_mask for attention_mask in all_original_attention_mask],
                                                   dtype=torch.long)
        all_original_token_type_ids = torch.tensor([token_type_ids for token_type_ids in all_original_token_type_ids],
                                                   dtype=torch.long)
        all_labels = torch.tensor([label for label in all_labels], dtype=torch.long)
        is_augmented = torch.tensor([is_augmented for is_augmented in all_is_augmented], dtype=torch.long)

        if self.enable_r1_loss:
            all_noised_input_ids = torch.tensor([input_ids for input_ids in all_noised_input_ids], dtype=torch.long)
            all_noised_attention_mask = torch.tensor([attention_mask for attention_mask in all_noised_attention_mask],
                                                     dtype=torch.long)
            all_noised_token_type_ids = torch.tensor([token_type_ids for token_type_ids in all_noised_token_type_ids],
                                                     dtype=torch.long)
            all_r1_mask = torch.tensor([r1_mask for r1_mask in all_r1_mask],
                                               dtype=torch.long)

            dataset = TensorDataset(all_original_input_ids, all_original_attention_mask, all_original_token_type_ids,
                                    all_labels, is_augmented, all_noised_input_ids, all_noised_attention_mask,
                                    all_noised_token_type_ids, all_r1_mask)
        else:
            dataset = TensorDataset(all_original_input_ids, all_original_attention_mask, all_original_token_type_ids,
                                    all_labels, is_augmented)
        return dataset

    def get_translation_pair(self, text_a, text_b):
        if text_a.strip() in self.tgt2src_dict and text_b.strip() in self.tgt2src_dict:
            # tgt to {en, tgt}
            en_text_a = self.tgt2src_dict[text_a.strip()]
            en_text_b = self.tgt2src_dict[text_b.strip()]
            lang_id_a = random.randint(0, len(self.train_languages) - 1)
            if self.translate_different_pair:
                lang_id_b = random.randint(0, len(self.train_languages) - 1)
            else:
                lang_id_b = lang_id_a

            if text_a == self.translate_train_dicts[lang_id_a][en_text_a.strip()]:
                text_a = en_text_a
            else:
                text_a = self.translate_train_dicts[lang_id_a][en_text_a.strip()]

            if text_b == self.translate_train_dicts[lang_id_b][en_text_b.strip()]:
                text_b = en_text_b
            else:
                text_b = self.translate_train_dicts[lang_id_b][en_text_b.strip()]
        else:
            # en to tgt
            lang_id_a = random.randint(0, len(self.train_languages) - 1)
            if self.translate_different_pair:
                lang_id_b = random.randint(0, len(self.train_languages) - 1)
            else:
                lang_id_b = lang_id_a

            assert text_a.strip() in self.translate_train_dicts[lang_id_a]
            assert text_b.strip() in self.translate_train_dicts[lang_id_b]

            text_a = self.translate_train_dicts[lang_id_a][text_a.strip()]
            text_b = self.translate_train_dicts[lang_id_b][text_b.strip()]

        return text_a, text_b

    def load_translate_data(self):
        self.translate_train_dicts = []
        self.tgt2src_dict = {}
        self.tgt2src_cnt = {}
        for i, language in enumerate(self.train_languages):
            logger.info("reading training data from lang {}".format(language))
            processor = processors[self.task_name](language=language, train_language=language)
            src2tgt_dict = processor.get_translate_train_dict(self.translation_path, self.tgt2src_dict, self.tgt2src_cnt)
            self.translate_train_dicts.append(src2tgt_dict)

    def get_train_steps(self, dataloader_size, args):
        n_augment_batch = math.ceil(dataloader_size * (1 + self.augment_ratio))
        augment_steps = n_augment_batch // args.gradient_accumulation_steps
        if args.max_steps > 0:
            t_total = args.max_steps
            assert False
        else:
            t_total = augment_steps * args.num_train_epochs
        return t_total


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def ConcatDataset(dataset_list):
    all_input_ids = torch.cat([dataset.tensors[0] for dataset in dataset_list], dim=0)
    all_attention_mask = torch.cat([dataset.tensors[1] for dataset in dataset_list], dim=0)
    all_token_type_ids = torch.cat([dataset.tensors[2] for dataset in dataset_list], dim=0)
    all_labels = torch.cat([dataset.tensors[3] for dataset in dataset_list], dim=0)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def train(args, train_examples, train_dataset, model, first_stage_model, tokenizer, noised_data_generator=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, "tb-log"))
        log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'w')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if noised_data_generator is not None and noised_data_generator.enable_data_augmentation:
        t_total = noised_data_generator.get_train_steps(len(train_dataloader), args)
    else:
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
    logger.info("  Logging steps = %d", args.logging_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and False:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss, best_avg = 0.0, 0.0, 0.0
    tr_original_loss, logging_original_loss = 0.0, 0.0
    tr_noised_loss, logging_noised_loss = 0.0, 0.0
    tr_r1_loss, logging_r1_loss = 0.0, 0.0
    tr_r2_loss, logging_r2_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility

    def logging(eval=False):
        results = None
        if args.evaluate_during_training and eval:
            results = evaluate(args, model, tokenizer, single_gpu=True)
            for task, result in results.items():
                for key, value in result.items():
                    tb_writer.add_scalar("eval_{}_{}".format(task, key), value, global_step)
                    logger.info("eval_%s_%s: %s" % (task, key, value))
            log_writer.write("{0}\t{1}\n".format(global_step, json.dumps(results)))
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

    def save_checkpoint_best(result):
        task_metric = "acc"
        if args.task_name == "rel":
            task_metric = "ndcg"
        if result is not None and best_avg < result["valid_avg"][task_metric]:
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
            return result["valid_avg"][task_metric]
        else:
            return best_avg

    for _ in train_iterator:
        if noised_data_generator is not None:
            assert noised_data_generator.enable_r1_loss or noised_data_generator.noised_loss or noised_data_generator.enable_data_augmentation
            noised_train_dataset = noised_data_generator.get_noised_dataset(train_examples)

            train_sampler = RandomSampler(noised_train_dataset) if args.local_rank == -1 else DistributedSampler(
                noised_train_dataset)
            train_dataloader = DataLoader(noised_train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            # if not args.max_steps > 0:
            #     assert t_total == len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            if first_stage_model is not None:
                first_stage_model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            if len(batch) == 4:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
            elif len(batch) == 5:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                inputs["is_augmented"] = batch[4]
            else:
                assert len(batch) == 9
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3],
                          "is_augmented": batch[4],
                          "noised_input_ids": batch[5],
                          "noised_attention_mask": batch[6],
                          "r1_mask": batch[8]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                    inputs["noised_token_type_ids"] = (
                        batch[7] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids

            if first_stage_model is not None:
                first_stage_model_inputs = {"input_ids": inputs["input_ids"],
                                       "attention_mask": inputs["attention_mask"],
                                       "token_type_ids": inputs["token_type_ids"],
                                       "labels": inputs["labels"]}
                with torch.no_grad():
                    inputs["first_stage_model_logits"] = first_stage_model(**first_stage_model_inputs)[1]

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
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

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    do_eval = args.evaluate_steps > 0 and global_step % args.evaluate_steps == 0
                    cur_result = logging(eval=do_eval)
                    logging_loss = tr_loss
                    logging_original_loss = tr_original_loss
                    logging_noised_loss = tr_noised_loss
                    logging_r1_loss = tr_r1_loss
                    logging_r2_loss = tr_r2_loss
                    best_avg = save_checkpoint_best(cur_result)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.local_rank in [-1, 0] and args.logging_each_epoch:
            cur_result = logging(eval=True)
            logging_loss = tr_loss
            logging_original_loss = tr_original_loss
            logging_noised_loss = tr_noised_loss
            logging_r1_loss = tr_r1_loss
            logging_r2_loss = tr_r2_loss
            best_avg = save_checkpoint_best(cur_result)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        log_writer.close()

    return global_step, tr_loss / (global_step + 1)


def predict(args, model, tokenizer, label_list, prefix="", single_gpu=False, verbose=True):
    if single_gpu:
        args = copy.deepcopy(args)
        args.local_rank = -1
        args.n_gpu = 1
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    eval_datasets = []
    eval_langs = args.language.split(',')
    for split in ["test"]:
        for lang in eval_langs:
            eval_datasets.append((split, lang))
    results = {}

    # leave interface for multi-task evaluation
    eval_task = eval_task_names[0]
    eval_output_dir = eval_outputs_dirs[0]

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split, lang in eval_datasets:
        task_name = "{0}-{1}".format(split, lang)
        eval_dataset, guids = load_and_cache_examples(args, eval_task, tokenizer, lang, split=split)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        guids = np.array(guids)
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                outputs = model(**inputs)
                logits = outputs[0]

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XGLUE.")
        results[lang] = preds

    for lang in results.keys():
        output_eval_file = os.path.join(eval_output_dir, prefix, "{}.prediction".format(lang))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            print("results:", results)
            for item in results[lang]:
                writer.write(str(label_list[item]) + "\n")


def evaluate(args, model, tokenizer, prefix="", single_gpu=False, verbose=True):
    if single_gpu:
        args = copy.deepcopy(args)
        args.local_rank = -1
        args.n_gpu = 1
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    eval_datasets = []
    eval_langs = args.language.split(',')
    splits = ["valid", "test"] if args.do_train else ["test"]
    for split in splits:
        for lang in eval_langs:
            eval_datasets.append((split, lang))
    results = {}

    # leave interface for multi-task evaluation
    eval_task = eval_task_names[0]
    eval_output_dir = eval_outputs_dirs[0]

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for split, lang in eval_datasets:
        task_name = "{0}-{1}".format(split, lang)
        eval_dataset, guids = load_and_cache_examples(args, eval_task, tokenizer, lang, split=split)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        guids = np.array(guids)
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XGLUE.")
        # print("pred:" + split + str([i for i in preds[:500]]), flush=True)
        # print("label:" + split + str([i for i in out_label_ids[:500]]), flush=True)
        result = compute_metrics(eval_task, preds, out_label_ids, guids)
        results[task_name] = result

    if args.do_train:
        results["valid_avg"] = average_dic([value for key, value in results.items() if key.startswith("valid")])
    results["test_avg"] = average_dic([value for key, value in results.items() if key.startswith("test")])
    return results


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


def load_and_cache_examples(args, task, tokenizer, language, split="train", return_examples=False):
    assert split in ["train", "valid", "test"]
    if args.local_rank not in [-1, 0] and evaluate == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task](language=language, train_language=language)
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    # data_cache_name = list(filter(None, args.model_name_or_path.split("/"))).pop()
    data_cache_name = "xlmr-base-final"
    if args.data_cache_name is not None:
        data_cache_name = args.data_cache_name

    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            split,
            data_cache_name,
            str(args.max_seq_length),
            str(task),
            str(language),
        ),
    )

    if split == "test":
        examples = processor.get_test_examples(args.data_dir)
    elif split == "valid":
        examples = processor.get_valid_examples(args.data_dir)
    else:  # train
        examples = processor.get_train_examples(args.data_dir)

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_guids = [f.guid for f in features]

    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    # if output_mode == "classification" and (not split == "test") :
    #     all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    # else:
    #     all_labels = torch.tensor([0 for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    if return_examples:
        return dataset, all_guids, examples
    else:
        return dataset, all_guids


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
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
        "--data_cache_name",
        default=None,
        type=str,
        help="The name of cached data",
    )
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
    parser.add_argument(
        "--sample_ratio", default=0.0, type=float, help="The training sample ratio of each language"
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # stable fine-tuning paramters
    parser.add_argument("--overall_ratio", default=1.0, type=float, help="overall ratio")
    parser.add_argument("--enable_r1_loss", action="store_true", help="Whether to enable r1 loss.")
    parser.add_argument("--r1_lambda", default=5.0, type=float, help="lambda of r1 loss")
    parser.add_argument("--original_loss", action="store_true",
                        help="Whether to use cross entropy loss on the former example.")
    parser.add_argument("--noised_loss", action="store_true",
                        help="Whether to use cross entropy loss on the latter example.")
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
    parser.add_argument("--enable_word_dropout", action="store_true", help="Whether to enable word dropout.")
    parser.add_argument("--word_dropout_rate", default=0.1, type=float, help="word dropout rate.")
    parser.add_argument("--enable_translate_data", action="store_true", help="Whether to enable translate data.")
    parser.add_argument("--translation_path", default=None, type=str, help="translation path")
    parser.add_argument("--translate_languages", default=None, type=str, help="translate languages")
    parser.add_argument("--translate_different_pair", action="store_true", help="Whether to translate different pair.")
    parser.add_argument("--translate_en_data", action="store_true", help="Whether to translate en data.")
    parser.add_argument("--enable_data_augmentation", action="store_true", help="Whether to enable data augmentation.")
    parser.add_argument("--augment_method", default=None, type=str, help="augment method")
    parser.add_argument("--augment_ratio", default=1.0, type=float, help="augmentation ratio.")
    parser.add_argument("--first_stage_model_path", default=None, type=str, required=False,
                        help="stable model path")
    parser.add_argument("--r2_lambda", default=1.0, type=float, required=False,
                        help="r2_lambda")
    parser.add_argument("--use_hard_labels", action="store_true", help="Whether to use hard labels.")

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--gpu_id", default="", type=str, help="GPU id"
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
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on the test set.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
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

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--evaluate_steps", type=int, default=5000, help="Log every X updates steps.")
    parser.add_argument("--logging_each_epoch", action="store_true", help="Whether to log after each epoch.")
    parser.add_argument("--logging_steps_in_sample", type=int, default=-1, help="log every X samples.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

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
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--train_cut_ratio", type=float, default=1.0, help="Cut training data to the ratio")
    args = parser.parse_args()

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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
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

    # preprocess args
    if args.train_language is None or args.train_language == "all":
        args.train_language = args.language

    assert not (
            args.logging_steps != -1 and args.logging_steps_in_sample != -1), "these two parameters can't both be setted"
    if args.logging_steps == -1 and args.logging_steps_in_sample != -1:
        total_batch_size = args.n_gpu * args.per_gpu_train_batch_size * args.gradient_accumulation_steps
        args.logging_steps = args.logging_steps_in_sample // total_batch_size

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name](language=args.language, train_language=args.train_language)
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.enable_r1_loss or args.noised_loss or args.enable_data_augmentation:
        noised_data_generator = NoisedDataGenerator(
            task_name=args.task_name,
            enable_r1_loss=args.enable_r1_loss,
            r1_lambda=args.r1_lambda,
            original_loss=args.original_loss,
            noised_loss=args.noised_loss,
            max_length=args.max_seq_length,
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
            enable_word_dropout=args.enable_word_dropout,
            word_dropout_rate=args.word_dropout_rate,
            enable_translate_data=args.enable_translate_data,
            translation_path=args.translation_path,
            train_language=args.language if args.translate_languages is None else args.translate_languages,
            data_dir=args.data_dir,
            translate_different_pair=args.translate_different_pair,
            translate_en_data=args.translate_en_data,
            enable_data_augmentation=args.enable_data_augmentation,
            augment_method=args.augment_method,
            augment_ratio=args.augment_ratio,
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
        # state_dict = torch.load(args.reload)

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        noised_data_generator=noised_data_generator,
        cache_dir=args.cache_dir if args.cache_dir else None,
        state_dict=state_dict,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    if first_stage_model is not None:
        first_stage_model.to(args.device)
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        train_langs = args.train_language.split(',')
        dataset_list = []
        train_examples = []
        for lang in train_langs:
            lg_train_dataset, guids, lg_examples = load_and_cache_examples(args, args.task_name, tokenizer, lang,
                                                                           split="train", return_examples=True)
            dataset_list.append(lg_train_dataset)
            train_examples += lg_examples
        train_dataset = ConcatDataset(dataset_list)

        global_step, tr_loss = train(args, train_examples, train_dataset, model, first_stage_model, tokenizer,
                                     noised_data_generator)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.init_checkpoint:
        best_checkpoint = args.init_checkpoint
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint = best_checkpoint
        tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
        logger.info("Evaluate the following checkpoints: %s", checkpoint)
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        for key, value in result.items():
            logger.info("eval_{}: {}".format(key, value))
        log_writer = open(os.path.join(args.output_dir, "evaluate_logs.txt"), 'w')
        log_writer.write("{0}\t{1}".format("evaluate", json.dumps(result)) + '\n')

    if args.do_predict and args.local_rank in [-1, 0]:
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoint = best_checkpoint
        tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        predict(args, model, tokenizer, label_list)

    logger.info("Task {0} finished!".format(args.task_name))
    return results


if __name__ == "__main__":
    main()
