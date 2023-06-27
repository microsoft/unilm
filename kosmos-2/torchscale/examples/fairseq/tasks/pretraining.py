# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import json
import logging
import os
from argparse import Namespace

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field

import sentencepiece as spm
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from omegaconf import II, MISSING

from .data.mlm_loader import MLMLoader

logger = logging.getLogger(__name__)

SAMPLE_BREAK_MODE_CHOICES = ChoiceEnum(["none", "complete", "complete_doc", "eos"])
SHORTEN_METHOD_CHOICES = ChoiceEnum(["none", "truncate", "random_crop"])


@dataclass
class PretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING,
        metadata={
            "help": "colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner"
        },
    )
    sample_break_mode: SAMPLE_BREAK_MODE_CHOICES = field(
        default="complete",
        metadata={
            "help": 'If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.'
        },
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    mask_prob: float = field(
        default=0.15,
        metadata={"help": "probability of replacing a token with mask"},
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    random_token_prob: float = field(
        default=0.1,
        metadata={"help": "probability of replacing a token with a random token"},
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=False,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: SHORTEN_METHOD_CHOICES = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed --tokens-per-sample"
        },
    )
    shorten_data_split_list: str = field(
        default="",
        metadata={
            "help": "comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)'
        },
    )
    seed: int = II("common.seed")
    span_length: float = field(
        default=3.0,
        metadata={"help": "average span length for masking"},
    )
    remove_source_sentinel: bool = field(
        default=False,
        metadata={"help": "remove the source sentinel for the span corruption task"},
    )
    remove_target_sentinel: bool = field(
        default=False,
        metadata={"help": "remove the target sentinel for the span corruption task"},
    )
    batch_read_ahead: int = field(
        default=100000,
        metadata={"help": "batch read ahead size for infinibatch"},
    )
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")
    spm_model: str = field(
        default="",
        metadata={"help": "sentencepice model to tokenize the data"},
    )
    dict_file: str = field(
        default="",
        metadata={"help": ""},
    )


@register_task("pretraining", dataclass=PretrainingConfig)
class PLMTask(FairseqTask):
    def __init__(self, cfg, dictionary, tokenizer):
        super().__init__(cfg)
        self.cfg = cfg
        self.dictionary = dictionary
        self.tokenizer = tokenizer
        self.seed = cfg.seed
        self.mask_idx = dictionary.index("<mask>")

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        if cfg.dict_file != "":
            dictionary = Dictionary.load(cfg.dict_file)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

        # add mask token
        dictionary.add_symbol("<mask>")
        for i in range(100):
            dictionary.add_symbol(f"<mask_{i}>")

        dictionary.pad_to_multiple_(cfg.required_batch_size_multiple)
        logger.info("dictionary: {} types".format(len(dictionary)))

        # tokenizer = SentencepieceBPE(Namespace(sentencepiece_model=cfg.spm_model))
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load(cfg.spm_model)
        return cls(cfg, dictionary, tokenizer)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = {
            "data": json.load(open(f"{self.cfg.data}/json/{split}.json")),
            "data_dir": self.cfg.data,
            "shuffle": True if split == "train" else False,
        }
        self.datasets[split] = Namespace(**self.datasets[split])

    def dataset(self, split):
        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)

        return self.datasets[split]

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        return MLMLoader(
            self.cfg,
            dataset,
            self.dictionary,
            self.tokenizer,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            max_positions=max_positions,
            ignore_invalid_inputs=ignore_invalid_inputs,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
        )

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
