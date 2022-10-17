# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
from argparse import Namespace
from collections import OrderedDict

import torch
from dataclasses import dataclass, field
from fairseq.data import (
    Dictionary,
    encoders,
    data_utils,
    StripTokenDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    DenoisingDataset,
    ConcatDataset,
    FairseqDataset,
    iterators,
    ResamplingDataset,
    MaskTokensDataset,
    LanguagePairDataset,
)
from fairseq.data.audio.speech_to_text_joint_dataset import S2TJointDataConfig
from fairseq.data.shorten_dataset import maybe_shorten_dataset
# from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from fairseq.dataclass.constants import ChoiceEnum
from omegaconf import MISSING

from speechlm.data.multimodal_corpus_dataset import MultiCorpusDataset
from speechlm.data.load_langpair_dataset  import load_langpair_dataset
from speechlm.data.language_trible_dataset import LanguageTripleDataset, load_langtriple_dataset
from speechlm.data.hubert_dataset import HubertDataset

logger = logging.getLogger(__name__)

TOKENIZER_CHOICES = ChoiceEnum(["sentencepiece", "hubert_letters", "none"])

def _lang_token(lang: str):
    return "<lang:{}>".format(lang)

def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx


class LabelEncoder(object):
    def __init__(self, dictionary: Dictionary) -> None:
        self.dictionary = dictionary

    def __call__(self, label: str) -> List[str]:
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False,
        )


### wrap the initial get_whole_word_mask which needs bpe_tokenizer,
### here we just assume words are splited by "|" or "<SIL>"
def get_whole_word_mask(args, dictionary):
    def is_beginning_of_word(i):
        if i < dictionary.nspecial:
            # special elements are always considered beginnings
            return True
        tok = dictionary[i]
        if tok.startswith("madeupword"):
            return True
        elif tok in ["<unk>", "<s>", "</s>", "<pad>", "|", "<eps>"]:
            return True
        else:
            return False

    mask_whole_words = torch.ByteTensor(
        list(map(is_beginning_of_word, range(len(dictionary))))
    )
    return mask_whole_words

def get_repeative_start(tokens):
    """
    tokens: torch.Tensor with repeative tokens
    """
    length = len(tokens)
    rep_start_id = tokens[:-1] != tokens[1:]
    return torch.cat([torch.tensor([True]), rep_start_id])

@dataclass
class TextPretrainingConfig(FairseqDataclass):    
    ### added for joint pretraining
    text_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, path to text data directory",
        },
    )
    seed: Optional[int] = field(
        default=1,
        metadata={
            "help": "for ordered_indices in MulticorpusDataset",
        },
    )
    tokens_per_sample: Optional[int] = field(
        default=512,
        metadata={
            "help": "max number of total tokens over all segments per sample for dataset",
        },
    )
    tokens_per_sample_tgt: Optional[int] = field(
        default=512,
        metadata={
            "help": "max number of total tokens over all segments per target sample for dataset",
        },
    )
    sample_break_mode: Optional[str] = field(
        default="eos",
        metadata={
            "help": "mode for breaking sentence",
        },
    )
    mask: Optional[float] = field(
        default=0.3,
        metadata={
            "help": "fraction of words/subwords that will be masked",
        },
    )
    leave_unmasked_prob: float = field(
        default=0.1,
        metadata={"help": "probability that a masked token is unmasked"},
    )
    mask_random: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "instead of using [MASK], use random token this often",
        },
    )
    freq_weighted_replacement: bool = field(
        default=False,
        metadata={"help": "sample random replacement words based on word frequencies"},
    )
    mask_whole_words: bool = field(
        default=True,
        metadata={"help": "mask whole words; you may also want to set --bpe"},
    )
    mask_repeative_tokens: bool = field(
        default=True,
        metadata={"help": "mask repeative_tokens; if mask_whole_words=False"},
    )
    mask_multiple_length: int = field(
        default=1,
        metadata={"help": "repeat the mask indices multiple times"},
    )
    mask_stdev: float = field(
        default=0.0,
        metadata={"help": "stdev of the mask length"},
    )
    shorten_method: Optional[str] = field(
        default="none",
        metadata={
            "help": "if not none, shorten sequences that exceed tokens_per_sample",
            "choices": "none/truncate/random_crop"
        },
    )
    shorten_data_split_list: Optional[str] = field(
        default="",
        metadata={
            "help": "comma_separated list of dataset splits to apply shortening to, e.g., train,valid (default: all dataset splits)",
        },
    )

    ### below hypra-parameters is used in bart
    insert: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "insert this percentage of additional random tokens",
        },
    )
    permute: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "take this proportion of subwords and permute them",
        },
    )
    rotate: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "rotate this proportion of inputs",
        },
    )
    poisson_lambda: Optional[float] = field(
        default=3.5,
        metadata={
            "help": "randomly shuffle sentences for this proportion of inputs",
        },
    )
    permute_sentences: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "shuffle this proportion of sentences in all inputs",
        },
    )
    mask_length: Optional[str] = field(
        default="span-poisson",
        metadata={
            "help": "mask length to choose",
            "choice": "subword/word/span-poisson"
        },
    )
    replace_length: Optional[int] = field(
        default=1,
        metadata={
            "help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
        },
    )
    shuffle_instance: Optional[bool] = field(
        default=False,
        metadata={"help": "shuffle instance"},
    )
    max_source_positions: Optional[int] = field(
        default=1024,
        metadata={"help": "max number of tokens in the source sequence"},
    )
    max_target_positions: Optional[int] = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"},
    )
    bpe: Optional[str] = field(
        default="",
        metadata={
            "help": "will wrapped by the text_data_config yaml",
        },
    )
    data_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "a config yaml specify the bpe model of text data",
        },
    )
    text_maxtokens_ratio: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "for text, max_tokens = max_tokens * text_maxtokens_ratio / 320 ",
        },
    )
    prepend_tgt_lang_tag: bool = field(
        default=False,
        metadata={"help": "prepend tgt_lang_tag to replace <eos>"},
    )
    mask_text_ratio: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "mask_text_ratio, for paired data",
        },
    )
    truncate_mono_source: bool = field(
        default=True,
        metadata={"help": "truncate mono source-side examples that exceed max-positions"},
    )


@dataclass
class JointPretrainingConfig(FairseqDataclass):
    data: str = field(
        default=MISSING, metadata={"help": "path to speech data directory"}
    )
    fine_tuning: bool = field(
        default=False, metadata={"help": "set to true if fine-tuning Hubert"}
    )
    labels: List[str] = field(
        default_factory=lambda: ["ltr"],
        metadata={
            "help": (
                "extension of the label files to load, frame-level labels for"
                " pre-training, and sequence-level label for fine-tuning"
            )
        },
    )
    label_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "if set, looks for labels in this directory instead",
        },
    )
    label_rate: int = field(
        default=-1,
        metadata={"help": "label frame rate. -1 for sequence label"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
            "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={
            "help": "if set, normalizes input to have 0 mean and unit variance"
        },
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    single_target: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset"
        },
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )
    store_labels: Optional[bool] = field(
        default=True,
        metadata={"help": "store spm labels in memory, should be true when fine-tune with bpe"},
    )
    add_decoder_target: bool = field(
        default=False,
        metadata={"help": "contral the model architecture, if set True, load reduced unit as target"},
    )
    split_modality_batch: bool = field(
        default=False,
        metadata={"help": "whether create all samples of different modalities in a batch"},
    )
    speech_tgt_lang: str = field(
        default="",
        metadata={"help": "prepend <tgt-id> to prev_output_tokens to replace <eos>, only used for decoder"},
    )
    speech_sampling_alpha: float = field(
        default=0.2,
        metadata={
            "help": "Hyper-parameter alpha = 1/T for temperature-based speech resampling."
            "(alpha = 1 for no resampling)"
        },
    )
    text_sampling_alpha: float = field(
        default=0.2,
        metadata={
            "help": "Hyper-parameter alpha = 1/T for temperature-based text resampling."
            "(alpha = 1 for no resampling)"
        },
    )
    hubert_tokenizer: Optional[TOKENIZER_CHOICES] = field(
        default="none",
        metadata={"help": "which tokenizer for processing text"},
    )
    sp_path: Optional[str] = field(
        default=None,
        metadata={"help": "sentencepiece model path if using bpe tokenizer"},
    )

    text_cfg: TextPretrainingConfig = TextPretrainingConfig()


@register_task("joint_sc2t_pretraining", dataclass=JointPretrainingConfig)
class Jsc2tPretrainingTask(FairseqTask):

    cfg: JointPretrainingConfig

    def __init__(
        self,
        cfg: JointPretrainingConfig,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"JSTPretrainingTask Config {cfg}")

        self.cfg = cfg
        self.fine_tuning = cfg.fine_tuning
        self.blank_symbol = "<s>"

        self.state.add_factory("hubert_tokenizer", self.build_tokenizer)
        if self.cfg.text_cfg.text_data is not None and os.path.exists(self.cfg.text_cfg.text_data):
            self.state.add_factory("text_dictionary", self.load_text_dictionary)
            self.state.add_factory("text_src_dictionary", self.load_text_src_dictionary)
        if cfg.fine_tuning:
            self.state.add_factory("target_dictionary", self.load_dictionaries)
        else:
            self.state.add_factory("dictionaries", self.load_dictionaries)

        if cfg.text_cfg.data_config is not None:
            self.text_data_cfg = S2TJointDataConfig(Path(f"{cfg.text_cfg.text_data}/{cfg.text_cfg.data_config}"))
            self.cfg.text_cfg.bpe = self.text_data_cfg.bpe_tokenizer["bpe"]
        else:
            self.text_data_cfg = None

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return self.state.target_dictionary

    @property
    def dictionaries(self) -> List[Dictionary]:
        return self.state.dictionaries

    @property
    def text_dictionary(self) -> Optional[Dictionary]:
        return self.state.text_dictionary

    @property
    def text_src_dictionary(self) -> Optional[Dictionary]:
        return self.state.text_src_dictionary

    @property
    def hubert_tokenizer(self):
        return self.state.hubert_tokenizer

    def load_dictionaries(self):
        label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
        dictionaries = [Dictionary.load(f"{label_dir}/dict.{label}.txt") for label in self.cfg.labels]
        if not self.cfg.fine_tuning:
            for dictionary in dictionaries:
                dictionary.add_symbol("<mask>")
        return dictionaries[0] if self.cfg.fine_tuning else dictionaries
    
    def load_text_dictionary(self):
        tgt_dict_path = f"{self.cfg.text_cfg.text_data}/{self.text_data_cfg.vocab_filename if self.text_data_cfg is not None else 'dict.txt'}"
        if not os.path.isfile(tgt_dict_path):
            raise FileNotFoundError(f"Dict not found: {tgt_dict_path}")
        text_dictionary = Dictionary.load(tgt_dict_path)
        self.mask_idx = text_dictionary.add_symbol("<mask>")
        return text_dictionary

    def load_text_src_dictionary(self):
        src_dict_path = f"{self.cfg.text_cfg.text_data}/{self.text_data_cfg.src_vocab_filename if self.text_data_cfg is not None else 'dict.txt'}"
        if not os.path.isfile(src_dict_path):
            raise FileNotFoundError(f"Dict not found: {src_dict_path}")
        src_text_dictionary = Dictionary.load(src_dict_path)
        self.mask_idx = src_text_dictionary.add_symbol("<mask>")
        return src_text_dictionary

    @classmethod
    def setup_task(
        cls, cfg: JointPretrainingConfig, **kwargs
    ) -> "Jsc2tPretrainingTask":
        return cls(cfg)

    def get_label_dir(self) -> str:
        if self.cfg.label_dir is None:
            return self.cfg.data
        return self.cfg.label_dir
    
    def load_paired_dataset(self, text_split, truncate_source=False):
        text_split, lp = text_split.rsplit('.', 1)       # e.g. "libritext.ltr-ltr"
        if len(lp.split("-")) == 2:
            src, tgt = lp.split("-")
            if src == tgt:
                logger.warn(f"| trying to load monolingual dataset {text_split}.{lp}, please check your task is right.")
                paired_dataset = self.load_char_bart_dataset(f"{text_split}.{lp}.{tgt}")
                return paired_dataset
            paired_dataset = load_langpair_dataset(
                self.cfg.text_cfg.text_data,
                text_split,
                src,
                self.text_src_dictionary,
                tgt,
                self.text_dictionary,
                combine=True,
                dataset_impl=None,
                upsample_primary=1,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.cfg.text_cfg.tokens_per_sample,
                max_target_positions=self.cfg.text_cfg.tokens_per_sample,
                truncate_source=truncate_source,
                prepend_bos=False,
                load_alignments=False,
                append_source_id=True if self.cfg.text_cfg.prepend_tgt_lang_tag else False,
                lang_format="<lang:{}>" if self.cfg.text_cfg.prepend_tgt_lang_tag else "[{}]",
                input_feeding=self.cfg.add_decoder_target,
            )
            if self.cfg.text_cfg.mask_text_ratio > 0:
                # add mask
                self.mask_idx = self.text_src_dictionary.index("<mask>")
                mask_whole_words = None
                if self.cfg.text_cfg.mask_whole_words:
                    mask_whole_words = get_whole_word_mask(self.cfg.text_cfg, self.text_src_dictionary)
                elif self.cfg.text_cfg.mask_repeative_tokens:
                    mask_whole_words = get_repeative_start
                
                src_dataset, src_unmasked_dataset = MaskTokensDataset.apply_mask(
                    paired_dataset.src,
                    self.text_src_dictionary,
                    pad_idx=self.text_src_dictionary.pad(),
                    mask_idx=self.mask_idx,
                    seed=self.cfg.text_cfg.seed,
                    mask_prob=self.cfg.text_cfg.mask_text_ratio,
                    leave_unmasked_prob=self.cfg.text_cfg.leave_unmasked_prob,
                    random_token_prob=self.cfg.text_cfg.mask_random,
                    freq_weighted_replacement=self.cfg.text_cfg.freq_weighted_replacement,
                    mask_whole_words=mask_whole_words,
                    mask_multiple_length=self.cfg.text_cfg.mask_multiple_length,
                    mask_stdev=self.cfg.text_cfg.mask_stdev,
                )
                tgt_dataset = paired_dataset.tgt if paired_dataset.tgt is not None else src_unmasked_dataset
                paired_dataset = LanguageTripleDataset(
                    src_dataset,
                    src_dataset.sizes,
                    self.text_src_dictionary,
                    src_unmasked_dataset,
                    src_unmasked_dataset.sizes,
                    self.text_src_dictionary,
                    tgt_dataset,
                    tgt_dataset.sizes,
                    self.text_dictionary,
                    left_pad_source=False,
                    left_pad_target=False,
                    align_dataset=None,
                    eos=None,
                    num_buckets=0,
                    shuffle=True,
                    pad_to_multiple=1,
                )
        else:
            src, ref, tgt = lp.split("-")
            paired_dataset = load_langtriple_dataset(
                self.cfg.text_cfg.text_data,
                text_split,
                src,
                self.text_src_dictionary,
                ref,
                self.dictionaries[-1],
                tgt,
                self.text_dictionary,
                combine=True,
                dataset_impl=None,
                upsample_primary=1,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.cfg.text_cfg.tokens_per_sample,
                max_target_positions=self.cfg.text_cfg.tokens_per_sample,
                truncate_source=truncate_source,
                prepend_bos=False,
                load_alignments=False,
                append_source_id=True if self.cfg.text_cfg.prepend_tgt_lang_tag else False,
                lang_format="<lang:{}>" if self.cfg.text_cfg.prepend_tgt_lang_tag else "[{}]",
            )
        return paired_dataset

    def load_dataset(self, split: str, epoch=1, **kwargs) -> None:
        """
            Create Wav dataset for audio, and Index dataset for phonemized text, 
            then concatenate them to by fairseq.data.multi_corpus_dataset.MultiCorpusDataset.
        """
        speech_splits = split.split('+')[0].split(',')
        ### 1st, create a speech dataset using STSpeechDataset (modified from HubertDataset)
        dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
        pad_list = [dict.pad() for dict in dicts]
        eos_list = [dict.eos() for dict in dicts]
        procs = [LabelEncoder(dict) for dict in dicts]
        if self.cfg.speech_tgt_lang != "":
            tgt_lang_idx = _lang_token_index(dicts[0], self.cfg.speech_tgt_lang)
            logger.info(f"Will prepend <{tgt_lang_idx}> at the beginning of prev_output_tokens to replace <eos>")
        else:
            tgt_lang_idx = None


        # hubert v1: pad_audio=True, random_crop=False;
        speech_datasets = []
        for speech_split in speech_splits:
            paths = [
                f"{self.get_label_dir()}/{speech_split}.{l}" for l in self.cfg.labels
            ]
            speech_datasets.append(
                HubertDataset(
                    f"{self.cfg.data}/{speech_split}.tsv",
                    sample_rate=self.cfg.sample_rate,
                    label_paths=paths,
                    label_rates=self.cfg.label_rate,
                    pad_list=pad_list,
                    eos_list=eos_list,
                    label_processors=procs,
                    max_keep_sample_size=self.cfg.max_keep_size,
                    min_keep_sample_size=self.cfg.min_sample_size,
                    max_sample_size=self.cfg.max_sample_size,
                    pad_audio=self.cfg.pad_audio,
                    normalize=self.cfg.normalize,
                    store_labels=self.cfg.store_labels,
                    random_crop=self.cfg.random_crop,
                    single_target=self.cfg.single_target,
                    tgt_dict=dicts[0],
                    add_decoder_target=self.cfg.add_decoder_target,
                    fine_tuning=self.cfg.fine_tuning,
                    tgt_lang_idx=tgt_lang_idx,
                    tokenizer=self.hubert_tokenizer,
                )
            )
        if len(speech_datasets) > 1:
            speech_dataset = ConcatDataset(speech_datasets)
        else:
            speech_dataset = speech_datasets[0]

        has_text = len(split.split('+')) > 1
        if not has_text:
            assert speech_dataset is not None
            self.datasets[split] = speech_dataset
            return

        ### 2nd, create paired/mono text datasets using Langpairdataset
        if split.split('+')[1] != '':
            paired_splits = [paired_split for paired_split in split.split('+')[1].split(',') if paired_split != '']
            paired_datasets = [self.load_paired_dataset(paired_split) for paired_split in paired_splits]
        else:
            paired_splits, paired_datasets = [], []

        if len(split.split('+')) > 2 and split.split('+')[2] != '':
            mono_splits = [mono_split for mono_split in split.split('+')[2].split(',') if mono_split != '']
            mono_datasets = [self.load_paired_dataset(mono_split, truncate_source=self.cfg.text_cfg.truncate_mono_source) for mono_split in mono_splits]
        else:
            mono_splits, mono_datasets = [], []
        
        assert len(mono_datasets + paired_datasets) > 0, f"split {split} has no text! you should check out for that"

        ### 3rd, if provided, create a supervised dataset with labeled data
        if len(split.split('+')) > 3 and split.split('+')[3] != '':
            assert len(paired_splits) > 0, f"supervised dataset can not be loaded without text paired dataset!"
            tgt = paired_splits[0].rsplit('.', 1)[1].split("-")[1]
            sup_split = split.split('+')[3]

            sup_dataset = HubertDataset(
                f"{self.cfg.data}/{sup_split}.tsv",
                sample_rate=self.cfg.sample_rate,
                label_paths=[f"{self.get_label_dir()}/{sup_split}.{tgt}"],
                label_rates=[-1],
                pad_list=[self.text_dictionary.pad()],
                eos_list=[self.text_dictionary.eos()],
                label_processors=[LabelEncoder(self.text_dictionary)],
                max_keep_sample_size=self.cfg.max_keep_size,
                min_keep_sample_size=None,
                max_sample_size=None,
                pad_audio=True,
                normalize=self.cfg.normalize,
                store_labels=self.cfg.store_labels,
                random_crop=False,
                single_target=True,
                tgt_dict=self.text_dictionary,
                add_decoder_target=self.cfg.add_decoder_target,
                fine_tuning=True,
                tgt_lang_idx=None,
                tokenizer=None,
            )
        else:
            sup_dataset = None

        ### 4th, compose a MultiCorpusDataset
        dataset_dict, max_positions_dict, distributions, max_tokens_ratios = self.resample_multi_modality_dataset(
        speech_dataset, sup_dataset, mono_datasets, paired_datasets, mono_splits, paired_splits, epoch=epoch,
        )
        self.datasets[split] = MultiCorpusDataset(
            dataset_dict,
            max_positions=max_positions_dict,
            distribution=distributions,
            max_tokens_ratio=max_tokens_ratios,
            seed=self.cfg.text_cfg.seed,
            sort_indices=True,
        )


    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self, indices: np.array, *args, **kwargs
    ) -> np.array:
        return indices

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
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.
        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
            grouped_shuffling (bool, optional): group batches with each groups
                containing num_shards batches and shuffle groups. Reduces difference
                between sequence lengths among workers for batches sorted by length.
            update_epoch_batch_itr (bool optional): if true then donot use the cached
                batch iterator for the epoch
            
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        if self.fine_tuning or not isinstance(dataset, MultiCorpusDataset):
            return super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                grouped_shuffling=grouped_shuffling,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )
        
        can_reuse_epoch_itr = (
            not disable_iterator_cache
            and not update_epoch_batch_itr
            and self.can_reuse_epoch_itr(dataset)
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            )

        # create mini-batches with given size constraints
        batch_sampler = dataset.get_batch_sampler(
            indices,
            num_shards,
            seed,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            split_modality_batch=self.cfg.split_modality_batch,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
            disable_shuffling=True,
            grouped_shuffling=grouped_shuffling,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()

    def resample_multi_modality_dataset(self, speech_dataset, sup_dataset, mono_datasets, paired_datasets, mono_splits, paired_splits, epoch=1, train=True):
        assert len(mono_datasets+paired_datasets) > 0, f"No text data loaded!"

        if len(mono_datasets) > 1 and self.cfg.text_sampling_alpha != 1.0:
            size_ratios = self._get_size_ratios(
                mono_splits, [len(s) for s in mono_datasets], alpha=self.cfg.text_sampling_alpha
            )
            mono_datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=0, epoch=epoch, replace=(r >= 1.0)
                ) for d, r in zip(mono_datasets, size_ratios)
            ]
            
        if len(paired_datasets) > 1 and self.cfg.text_sampling_alpha != 1.0:
            size_ratios = self._get_size_ratios(
                paired_splits, [len(s) for s in paired_datasets], alpha=self.cfg.text_sampling_alpha
            )
            paired_datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=0, epoch=epoch, replace=(r >= 1.0)
                ) for d, r in zip(paired_datasets, size_ratios)
            ]

        dataset_list = [speech_dataset, sup_dataset]
        for datasets in [mono_datasets, paired_datasets]:
            if len(datasets) > 1:
                dataset_list.append(ConcatDataset(datasets))
            elif len(datasets) == 1:
                dataset_list.append(datasets[0])
            else:
                dataset_list.append(None)

        ### match speech/text datasets according to modality
        dataset_dict = OrderedDict((name, d) for name, d in zip(["speech", "speech_sup", "text_mono", "text_paired"], dataset_list) if d is not None)
        max_positions_dict = {
            "speech": None,
            "speech_sup": None,
            "text_mono": (self.cfg.text_cfg.tokens_per_sample, self.cfg.text_cfg.tokens_per_sample),
            "text_paired": (self.cfg.text_cfg.tokens_per_sample, self.cfg.text_cfg.tokens_per_sample),
        }
        max_positions_dict = OrderedDict((name, max_positions_dict[name]) for name in dataset_dict.keys())
        max_tokens_ratios_dict = {
            "speech": 1.0,
            "speech_sup": 1.0,
            "text_mono": 1.0 / 320 / self.cfg.text_cfg.text_maxtokens_ratio,
            "text_paired": 1.0 / 320 / self.cfg.text_cfg.text_maxtokens_ratio,
        }
        max_tokens_ratios = [max_tokens_ratios_dict[name] for name in dataset_dict.keys()]
        dataset_lens = np.array([len(dataset) for dataset in dataset_dict.values()])
        dataset_avg_sample_lens = np.array([
            sum([dataset.num_tokens(i) for i in np.random.randint(low=0, high=len(dataset), size=10000)]) / 10000.0 
            for dataset in dataset_dict.values()
        ])

        if not "speech" in dataset_dict:
            distributions = [l / sum(dataset_lens) for l in dataset_lens]
        else:
            ## we just keep the batches of speech and non-speech the same, expand_coef is to ensure speech batches is less than others
            first_ratio = dataset_lens[0] / sum(dataset_lens)
            expand_coef = 1.8 if sup_dataset is None else 1.1 * sum(dataset_lens[0:2]) / dataset_lens[0]
            distributions = [expand_coef * max_tokens_ratios[i] * dataset_avg_sample_lens[0] / l for (i, l) in enumerate(dataset_avg_sample_lens)]
            distributions[0] = 1.0
            if sup_dataset is not None:
                distributions[1] = dataset_lens[1] / dataset_lens[0]
            distributions = [first_ratio * d for d in distributions]

        logging.info(f"Number samples of datasets is {dataset_lens}")
        logging.info(f"Avg sample length of datasets is {dataset_avg_sample_lens}")
        logging.info(f"Sampling distributions is {distributions}")
        logging.info(f"Maxtokens ratio is {max_tokens_ratios}")
        return dataset_dict, max_positions_dict, distributions, max_tokens_ratios

    def build_tokenizer(self, cfg=None):
        logger.info(f"tokenizer: {self.cfg.hubert_tokenizer}")
        if self.cfg.hubert_tokenizer != "none":
            return encoders.build_bpe(Namespace(**{"bpe": self.cfg.hubert_tokenizer, "sentencepiece_model": self.cfg.sp_path}))
        else:
            return None

    def load_char_bart_dataset(self, split):
        mono_dataset = data_utils.load_indexed_dataset(
            f"{self.cfg.text_cfg.text_data}/{split}",
            self.text_dictionary,
        )
        mono_dataset = StripTokenDataset(mono_dataset, self.text_dictionary.eos())
        mono_dataset = maybe_shorten_dataset(
            mono_dataset,
            split,
            self.cfg.text_cfg.shorten_data_split_list,
            self.cfg.text_cfg.shorten_method,
            self.cfg.text_cfg.tokens_per_sample - 2,
            self.cfg.text_cfg.seed,
        )
        logger.info("loaded {} samples from: {}".format(len(mono_dataset), mono_dataset))
        ### prepend bos and eos to dataset
        mono_dataset = PrependTokenDataset(mono_dataset, self.text_dictionary.bos())
        mono_dataset = AppendTokenDataset(mono_dataset, self.text_dictionary.eos())
        mask_whole_words = (
            get_whole_word_mask(None, self.text_dictionary)
            if self.cfg.text_cfg.mask_whole_words
            else None
        )
        lang=self.cfg.speech_tgt_lang
        mono_dataset = DenoisingDataset(
            mono_dataset,
            mono_dataset.sizes,
            self.text_dictionary,
            self.mask_idx,
            mask_whole_words,
            shuffle=self.cfg.text_cfg.shuffle_instance,
            seed=self.cfg.text_cfg.seed,
            args=self.cfg.text_cfg,
            tgt_lang_idx=_lang_token_index(self.text_dictionary, lang) if self.cfg.text_cfg.prepend_tgt_lang_tag else None,
        )
        
        return mono_dataset
