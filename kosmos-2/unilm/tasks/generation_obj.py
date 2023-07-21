import logging
import os

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    FairseqDataset,
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    RawLabelDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq import utils
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask
from fairseq.data import Dictionary, data_utils
from omegaconf import II
from fairseq import metrics, search, tokenizer, utils
from unilm.data.utils import SPECIAL_SYMBOLS, add_location_symbols

import pdb

logger = logging.getLogger(__name__)

@dataclass
class GenerationObjConfig(LanguageModelingConfig):
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")
    dict_path: str = field(
        default="",
        metadata={
            "help": "dictionary path"
        },
    )
    image_feature_length: int = field(
        default=0,
        metadata={
            "help": "image feature length"
        },
    ) 
    input_resolution: int = field(default=224, metadata={"help": ""})
    # newsetting
    location_bin_size: int = field(
        default=16, 
        metadata={
            "help": "used to discrete the continuous coordinates"
        },
    )  
    locate_special_token: int = field(
        default=0, 
        metadata={"help": "used to discrete the continuous coordinates"}
    )


class RawImageDataset(FairseqDataset):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels

    def __getitem__(self, index):
        return self.labels[index]

    def __len__(self):
        return len(self.labels)

    def collater(self, samples):
        return torch.stack(samples)


@register_task("generation_obj", dataclass=GenerationObjConfig)
class GenerationObjTask(LanguageModelingTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @classmethod
    def setup_dictionary(cls, args, **kwargs):
        dictionary = None
        output_dictionary = None
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        
        if len(args.dict_path) > 0:
            dictionary = Dictionary.load(args.dict_path)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))

        dictionary.add_symbol("<mask>")
        # location task, add patch index token as special symbols
        for special_symbol in add_location_symbols(args.location_bin_size, args.locate_special_token):
            dictionary.add_symbol(special_symbol)

        dictionary.pad_to_multiple_(args.required_batch_size_multiple)
        
        # pdb.set_trace()
        output_dictionary = dictionary
        logger.info("dictionary from {}: {} types".format(args.dict_path, len(dictionary)))
        return (dictionary, output_dictionary)


    def build_dataset_for_caption_inference(self, src_tokens, src_lengths, img_src_tokens, img_gpt_input_mask, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        img_gpt_input_mask = StripTokenDataset(
            TokenBlockDataset(
                img_gpt_input_mask,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = dataset

        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'img_src_tokens': RawImageDataset(
                        img_src_tokens,
                    ),
                    'img_gpt_input_mask': PadDataset(
                        img_gpt_input_mask,
                        pad_idx=0,
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )


    def build_dataset_for_speech_inference(self, src_tokens, src_lengths, aud_src_tokens, aud_gpt_input_mask, audio_masks, **kwargs):
        """
        Generate batches for inference. We prepend an eos token to src_tokens
        (or bos if `--add-bos-token` is set) and we append a <pad> to target.
        This is convenient both for generation with a prefix and LM scoring.
        """
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        aud_gpt_input_mask = StripTokenDataset(
            TokenBlockDataset(
                aud_gpt_input_mask,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = dataset

        tgt_dataset = AppendTokenDataset(dataset, token=self.source_dictionary.pad())
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        src_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    'aud_src_tokens': RawImageDataset(
                        aud_src_tokens,
                    ),
                    'aud_gpt_input_mask': PadDataset(
                        aud_gpt_input_mask,
                        pad_idx=0,
                        left_pad=False,
                    ),
                    'aud_masks': RawImageDataset(
                        audio_masks,
                    ),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False
                ),
            },
            sizes=[np.array(src_lengths)],
        )


    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            if constraints is not None:
                raise NotImplementedError(
                    "Constrained decoding with the language_modeling task is not supported"
                )

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                # if prefix_tokens[:, 0].eq(bos_token).all():
                #     prefix_tokens = prefix_tokens[:, 1:]

            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token
            )