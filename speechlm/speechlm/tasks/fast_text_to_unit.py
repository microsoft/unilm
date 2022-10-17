# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import torch
import numpy as np
import logging
from pathlib import Path
from argparse import Namespace

from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_joint_dataset import S2TJointDataConfig

from speechlm.unit_generator import NonAutoregressiveUnitGenerator
from speechlm.data.text_to_unit_dataset import Text2UnitDatasetCreator

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@register_task("fast_text_to_unit")
class FastTextToUnitTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=2048,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument("--n-frames-per-step", type=int, default=1)
        parser.add_argument("--eos-prob-threshold", type=float, default=0.5)
        parser.add_argument("--eval-inference", action="store_true")
        parser.add_argument("--eval-tb-nsample", type=int, default=8)
        parser.add_argument("--vocoder", type=str, default="griffin_lim")
        parser.add_argument("--spec-bwd-max-iter", type=int, default=8)

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if not src_dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {src_dict_path.as_posix()}")
        src_dict = Dictionary.load(src_dict_path.as_posix())
        logger.info(
            f"Source dictionary size ({data_cfg.src_vocab_filename}): " f"{len(src_dict):,}"
        )
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        if not tgt_dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {tgt_dict_path.as_posix()}")
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())
        logger.info(
            f"Target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, src_dict, tgt_dict)
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = Text2UnitDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.src_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            n_frames_per_step=self.args.n_frames_per_step,
            speaker_to_id=self.speaker_to_id,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.args.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def get_speaker_embeddings(cls, args):
        # It Will be used in FastText2UnitModel model, insdead of nn.Embedding on speaker-id, we default to use x-vectors extracted ahead.
        # This is for varying the speaker information when generating units from text.
        if args.speaker_to_id is not None:
            embed_speaker = torch.nn.Embedding(
                len(args.speaker_to_id), args.speaker_embed_dim
            )
        elif args.speaker_embedding_type == "x-vector":
            # return LayerNorm(args.speaker_embed_dim)
            return lambda x: x.unsqueeze(1)
        elif args.speaker_embedding_type == "i-vector":
            # return LayerNorm(args.speaker_embed_dim)
            return lambda x: x
        else:
            embed_speaker = None
        return embed_speaker

    def build_model(self, cfg):
        cfg.pitch_min = self.data_cfg.config["features"].get("pitch_min", None)
        cfg.pitch_max = self.data_cfg.config["features"].get("pitch_max", None)
        cfg.energy_min = self.data_cfg.config["features"].get("energy_min", None)
        cfg.energy_max = self.data_cfg.config["features"].get("energy_max", None)
        cfg.speaker_to_id = self.speaker_to_id
        cfg.speaker_embedding_type = self.data_cfg.config.get("speaker_embedding_type", None)
        model = super().build_model(cfg)
        self.generator = None
        if getattr(cfg, "eval_inference", False):
            self.generator = self.build_generator([model], cfg)
        return model

    def build_generator(self, models, cfg, vocoder=None, **unused):
        model = models[0]
        assert getattr(model, "NON_AUTOREGRESSIVE") is True
        return NonAutoregressiveUnitGenerator(model, vocoder, self.data_cfg)


    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))
