# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import os.path as op
from argparse import Namespace
from collections import OrderedDict

import torch
from fairseq.data import (
    Dictionary, 
    encoders, 
    PrependTokenDataset,
    AppendTokenDataset, 
    data_utils, 
    StripTokenDataset,
    TokenBlockDataset,
)
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq import utils
from speecht5.data.multitask_dataset import MultitaskDataset
from speecht5.data.speech_to_text_dataset import SpeechToTextDataset
from speecht5.data.text_to_speech_dataset import TextToSpeechDataset
from speecht5.data.speech_to_speech_dataset import SpeechToSpeechDataset
from speecht5.data.speech_dataset import SpeechPretrainDataset
from speecht5.data.text_dataset import TextPretrainDataset
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.hubert_pretraining import LabelEncoder 

logger = logging.getLogger(__name__)

TASK_NAME = ["s2t", "t2s", "s2s", "s2c", "pretrain"]

@register_task("speecht5")
class SpeechT5Task(LegacyFairseqTask):
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
            "--max-speech-sample-size",
            default=None,
            type=int,
            metavar="N",
            help="max speech sample size",
        )
        parser.add_argument(
            "--min-speech-sample-size",
            default=None,
            type=int,
            metavar="N",
            help="min speech sample size",
        )
        parser.add_argument(
            "--max-speech-positions",
            default=4000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-text-positions",
            default=450,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            '--t5-task',
            choices=TASK_NAME,
            help='task for training'
        )
        parser.add_argument(
            "--bpe-tokenizer",
            type=str,
            default=None,
            help="bpe tokenizer for s2t",
        )
        # Speaker Identification (SID)
        parser.add_argument(
            "--finetune-from-modules",
            default=None,
            # choices=[
            #     "encoder-decoder", "encoder", "decoder",
            #     "speech_encoder_prenet-encoder-decoder-text_decoder_prenet-text_decoder_postnet",     # ASR, T5 SID
            #     "speech_encoder_prenet-encoder-decoder-text_decoder_prenet-speaker_decoder_postnet",  # SID
            #     "speech_encoder_prenet-encoder-decoder-speech_decoder_prenet-speech_decoder_postnet", # VC, SE
            #     "text_encoder_prenet-encoder-decoder-speech_decoder_prenet-speech_decoder_postnet",   # TTS
            # ],
            help="If set, using part modules of finetune model.",
        )
        parser.add_argument(
            "--finetune-out-of-modules",
            default=None,
            # choices=[
            #     "speaker_decoder_postnet", # SID
            #     "speech_decoder_postnet", # SE with reduction factor 1
            # ],
            help="If set, remove part modules of finetune model.",
        )
        # BART
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )

        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments"
            " per sample for dataset",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="eos",
            type=str,
            help="mode for breaking sentence",
        )
        parser.add_argument(
            "--mask",
            default=0.3,
            type=float,
            help="fraction of words/subwords that will be masked",
        )
        parser.add_argument(
            "--mask-random",
            default=0.1,
            type=float,
            help="instead of using [MASK], use random token this often",
        )
        parser.add_argument(
            "--insert",
            default=0.0,
            type=float,
            help="insert this percentage of additional random tokens",
        )
        parser.add_argument(
            "--permute",
            default=0.0,
            type=float,
            help="take this proportion of subwords and permute them",
        )
        parser.add_argument(
            "--rotate",
            default=0.0,
            type=float,
            help="rotate this proportion of inputs",
        )
        parser.add_argument(
            "--poisson-lambda",
            default=3.5,
            type=float,
            help="randomly shuffle sentences for this proportion of inputs",
        )
        parser.add_argument(
            "--permute-sentences",
            default=0.0,
            type=float,
            help="shuffle this proportion of sentences in all inputs",
        )
        parser.add_argument(
            "--mask-length",
            default="span-poisson",
            type=str,
            choices=["subword", "word", "span-poisson"],
            help="mask length to choose",
        )
        parser.add_argument(
            "--replace-length",
            default=1,
            type=int,
            help="when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)",
        )
        parser.add_argument(
            "--iid-noise-target",
            action="store_true",
            help="whether to use t5 form target",
        )
        # Hubert
        parser.add_argument(
            "--hubert-labels",
            nargs="*",
            type=str,
            default=['km'],
            help="extension of the label files to load, frame-level labels for pre-training, and sequence-level label for fine-tuning",
        )
        parser.add_argument(
            "--hubert-label-dir",
            type=str,
            default=None,
            help="if set, looks for labels in this directory instead",
        )
        parser.add_argument(
            "--sample-rate",
            default=100,
            type=float,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--label-rates",
            default=-1,
            type=float,
            help="if set, looks for labels in this directory instead",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--enable-padding",
            action="store_true",
            help="pad shorter samples instead of cropping",
        )
        parser.add_argument(
            "--pad-audio",
            action="store_true",
            help="pad audio to the longest one in the batch if true",
        )
        parser.add_argument(
            "--random-crop",
            action="store_true",
            help="always crop from the beginning if false",
        )
        parser.add_argument(
            "--single-target",
            action="store_true",
            help="if set, AddTargetDatasets outputs same keys "
            "as AddTargetDataset",
        )
        parser.add_argument(
            "--batch-ratio",
            default=None,
            type=str,
            help="ratio of bach size for each dataset",
        )
        parser.add_argument(
            "--sample-ratios",
            default=None,
            type=str,
            help="ratio of sample for each dataset",
        )
        parser.add_argument(
            "--ctc-weight",
            type=float,
            default=0.0,
            help="ctc weight for inference",
        )

    def __init__(self, args, dicts, config):
        super().__init__(args)
        self.dicts = dicts
        self.config = config
        self.t5_task = args.t5_task
        # Used for filter size
        if self.t5_task in ['s2t', 't2s', 's2s']:
            self.max_pos = [self.args.max_speech_positions * 256]
        elif self.t5_task == 'pretrain':
            self.max_pos = [self.args.max_speech_positions * 256, self.args.max_text_positions]

        self.mask_idx = self.dicts["text"].add_symbol("<mask>")
        # add blank token for ctc
        # if args.ctc_weight > 0:
        self.blank_symbol_idx = self.dicts["text"].add_symbol("<ctc_blank>")
        self.blank_symbol = "<ctc_blank>"

        # add mask token
        if hasattr(args, "iid_noise_target") and args.iid_noise_target:
            self.uni_mask_idxs = []
            for i in range(600):
                self.uni_mask_idxs.append(self.dicts["text"].add_symbol("<mask>" + str(i)))
            self.uni_mask_idxs = torch.tensor(self.uni_mask_idxs)

        self.seed = args.seed

    @classmethod
    def setup_task(cls, args, **kwargs):
        # load dictionaries and config
        dicts = OrderedDict()
        if args.t5_task == 'pretrain' and not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False

        # Prepare config
        config = None
        logger.info('No config file for ' + args.t5_task)

        if args.t5_task == "pretrain":
            dicts["hubert"] = [Dictionary.load(f"{args.hubert_label_dir}/dict.{label}.txt") for label in args.hubert_labels]
            dicts["text"] = Dictionary.load(op.join(args.data, "dict.txt"))
        else:
            if config is None:
                dicts["text"] = Dictionary.load(op.join(args.data, "dict.txt"))
            else:
                dicts["text"] = Dictionary.load(op.join(args.data, config.vocab_filename))

        return cls(args, dicts, config)

    def build_criterion(self, args):
        from fairseq import criterions
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        sample_ratios = []
        if self.t5_task == "s2t":
            ## For speech to text task
            bpe_tokenizer = self.build_bpe(self.args)
            manifest = f"{self.args.data}/{split}.tsv"
            procs = [LabelEncoder(self.dicts["text"])]
            paths = [f"{self.args.hubert_label_dir}/{split}.txt"]
            self.datasets[split] = SpeechToTextDataset(
                manifest,
                sample_rate=self.args.sample_rate,
                label_paths=paths,
                label_processors=procs,
                max_keep_sample_size=self.max_pos[0] if self.args.max_speech_sample_size is None else self.args.max_speech_sample_size,
                min_keep_sample_size=self.args.min_speech_sample_size,
                normalize=self.args.normalize,
                store_labels=False,
                tgt_dict=self.dicts["text"],
                tokenizer=bpe_tokenizer,
            )
        elif self.t5_task == "t2s":
            ## For text to speech task
            from fairseq.data import ConcatDataset
            bpe_tokenizer = self.build_bpe(self.args)
            procs = [LabelEncoder(self.dicts["text"])]
            t2s_datasets = [
                TextToSpeechDataset(
                    manifest_path=f"{self.args.data}/{name}.tsv",
                    sample_rate=self.args.sample_rate,
                    label_paths=[f"{self.args.hubert_label_dir}/{name}.txt"],
                    label_processors=procs,
                    max_keep_sample_size=self.max_pos[0],
                    normalize=self.args.normalize,
                    store_labels=False,
                    src_dict=self.dicts["text"],
                    tokenizer=bpe_tokenizer,
                    reduction_factor=self.args.reduction_factor,
                )
                for name in split.split(",")
            ]
            self.datasets[split] = ConcatDataset(t2s_datasets) if len(t2s_datasets) > 1 else t2s_datasets[0]
        elif self.t5_task == "s2s":
            manifest = f"{self.args.data}/{split}.tsv"
            self.datasets[split] = SpeechToSpeechDataset(
                manifest_path=manifest,
                sample_rate=self.args.sample_rate,
                max_keep_sample_size=self.max_pos[0] if self.args.max_speech_sample_size is None else self.args.max_speech_sample_size,
                min_keep_sample_size=self.args.min_speech_sample_size,
                normalize=self.args.normalize,
                reduction_factor=self.args.reduction_factor,
            )
        elif self.t5_task == "pretrain":
            is_train_split = ("train" in split)
            pretrain_datasets = []
            speech_split, text_split = split.split('|')

            ## Speech pre-train
            manifest = f"{self.args.data}/{speech_split}.tsv"
            dicts = self.dicts["hubert"]
            pad_list = [dict.pad() for dict in dicts]
            eos_list = [dict.eos() for dict in dicts]
            procs = [LabelEncoder(dict) for dict in dicts]
            paths = [
                f"{self.args.hubert_label_dir}/{speech_split}.{l}" for l in self.args.hubert_labels
            ]
            # hubert v1: pad_audio=True, random_crop=False;
            self.args.dec_weight = getattr(self.args, "dec_weight", 1.0)
            pretrain_datasets.append(
                SpeechPretrainDataset(
                    manifest,
                    sample_rate=self.args.sample_rate,
                    label_paths=paths,
                    label_rates=self.args.label_rates,
                    pad_list=pad_list,
                    eos_list=eos_list,
                    label_processors=procs,
                    max_keep_sample_size=None,
                    min_keep_sample_size=32000,
                    max_sample_size=self.args.max_speech_sample_size,
                    pad_audio=self.args.pad_audio,
                    normalize=self.args.normalize,
                    store_labels=False,
                    random_crop=self.args.random_crop,
                    single_target=self.args.single_target,
                    reduction_factor=self.args.reduction_factor,
                )
            )
            sample_ratios.append(sum([pretrain_datasets[0].size(i) for i in range(len(pretrain_datasets[0]))]))

            ## Text pre-train
            paths = utils.split_paths(self.args.data)
            assert len(paths) > 0
            data_path = paths[(epoch - 1) % len(paths)]
            split_path = op.join(data_path, text_split)
            bart_dataset = data_utils.load_indexed_dataset(
                split_path,
                self.dicts["text"],
                self.args.dataset_impl,
                combine=combine,
            )
            if bart_dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(text_split, split_path)
                )
            bart_dataset = StripTokenDataset(bart_dataset, self.dicts["text"].eos())
            bart_dataset = maybe_shorten_dataset(
                bart_dataset,
                text_split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.tokens_per_sample,
                self.args.seed,
            )
            # create continuous blocks of tokens
            bart_dataset = TokenBlockDataset(
                bart_dataset,
                bart_dataset.sizes,
                self.args.tokens_per_sample - 2,  # one less for <s> and one for </s>
                pad=self.dicts["text"].pad(),
                eos=self.dicts["text"].eos(),
                break_mode=self.args.sample_break_mode,
                document_sep_len=0,
            )
            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            bart_dataset = PrependTokenDataset(bart_dataset, self.dicts["text"].bos())
            bart_dataset = AppendTokenDataset(bart_dataset, self.dicts["text"].eos())
            mask_whole_words = (
                get_whole_word_mask(self.args, self.dicts["text"])
                if self.args.mask_length != "subword"
                else None
            )
            self.args.bert_weight = getattr(self.args, "bert_weight", 0.0)
            pretrain_datasets.append(
                TextPretrainDataset(
                    bart_dataset,
                    bart_dataset.sizes,
                    self.dicts["text"],
                    self.mask_idx,
                    mask_whole_words,
                    shuffle=self.args.shuffle_instance,
                    seed=self.seed,
                    args=self.args,
                    iid_noise_target=self.args.iid_noise_target,
                    uni_mask_idxs=self.uni_mask_idxs if self.args.iid_noise_target else None,
                )
            )
            sample_ratios.append(sum(pretrain_datasets[1].sizes))
            logger.info(
                "Task: {0}, Loaded {1} samples of denoising_dataset".format(
                    'bart',
                    len(pretrain_datasets[1]),
                )
            )

            logger.info('token ratio is ' + str(sample_ratios))
            if self.args.batch_ratio is not None:
                batch_ratio = eval(self.args.batch_ratio)
                assert len(batch_ratio) == len(sample_ratios)
                sample_ratios = [sample_ratios[i] / batch_ratio[i] for i in range(len(sample_ratios))]
            else:
                batch_ratio = None
            max_size = max(sample_ratios)
            sample_ratios = [max_size / r for r in sample_ratios]
            if hasattr(self.args, "sample_ratios") and self.args.sample_ratios is not None:
                sample_ratios = eval(self.args.sample_ratios)
            if is_train_split:
                self.datasets[split] = MultitaskDataset(
                    pretrain_datasets, sample_ratios, batch_ratio
                )
            else:
                self.datasets[split] = MultitaskDataset(
                    pretrain_datasets, batch_ratio=batch_ratio
                )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)

        # Junyi: not use sample_size, but normalize the loss locally
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 1.0, {}
        agg_logging_output['sample_size'] = 1

        def forward_backward(model, samples, weight=1.0):
            nonlocal agg_loss, agg_logging_output
            if samples is None or len(samples) == 0:
                return
            loss, sample_size, logging_output = criterion(model, samples)
            if ignore_grad:
                loss *= 0
            else:
                loss *= weight
            loss = loss / sample_size
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # # TODO make summing of the sample sizes configurable
            for k in logging_output:
                if k == 'ntokens' or k == 'nsentences':
                    if k not in agg_logging_output:
                        agg_logging_output[k] = 0
                    agg_logging_output[k] += logging_output[k]
                    # continue
                # agg_logging_output[k] += logging_output[k]
                # agg_logging_output[task_name] += logging_output[k]
            agg_logging_output[samples['task_name']] = logging_output

        forward_backward(model, sample)

        agg_logging_output["loss"] = agg_loss

        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            from collections import defaultdict

            agg_loss, agg_sample_size, agg_logging_output = 0.0, 1.0, defaultdict(float)
            agg_logging_output['sample_size'] = 1
            loss, sample_size, logging_output = criterion(model, sample)
            loss = loss / sample_size
            # agg_loss += loss.data.item() if isinstance(loss, torch.Tensor) else loss
            agg_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
            agg_logging_output[sample['task_name']] = logging_output
            agg_logging_output["loss"] = agg_loss
        return agg_loss, agg_sample_size, agg_logging_output

    @property
    def target_dictionary(self):
        return self.dicts["text"]

    @property
    def source_dictionary(self):
        return None

    def build_model(self, args):
        try:
            args.input_feat_per_channel = self.config.input_feat_per_channel
            args.input_channels = self.config.input_channels
        except Exception as e:
            args.input_feat_per_channel = 80
            args.input_channels = 1
            logger.info(f"Cannot set input_feat_per_channel, input_channels, since: ")
            logger.warn(e)
            logger.info(f"Set to: {args.input_feat_per_channel} and {args.input_channels}")

        args.speech_odim = args.input_feat_per_channel * args.input_channels

        args.label_rates = self.args.label_rates
        args.sample_rate = self.args.sample_rate
        self.args.reduction_factor = args.reduction_factor
        return super(SpeechT5Task, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        from speecht5.sequence_generator import SequenceGenerator
        extra_gen_cls_kwargs = {
            "ctc_weight": self.args.ctc_weight,
            **extra_gen_cls_kwargs
        }
        return super().build_generator(
            models, args, seq_gen_cls=SequenceGenerator, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        if self.config is None:
            logger.info(f"pre-tokenizer: None")
            return encoders.build_tokenizer(Namespace(**{"tokenizer": None}))
        else:
            logger.info(f"pre-tokenizer: {self.config.pre_tokenizer}")
            return encoders.build_tokenizer(Namespace(**self.config.pre_tokenizer))

    def build_bpe(self, args):
        if self.config is not None:
            logger.info(f"tokenizer: {self.config.bpe_tokenizer}")
            return encoders.build_bpe(Namespace(**self.config.bpe_tokenizer))
        else:
            logger.info(f"tokenizer: {self.args.bpe_tokenizer}")
            return encoders.build_bpe(Namespace(**{"bpe": "sentencepiece", "sentencepiece_model": self.args.bpe_tokenizer}))

    def generate_speech(self, models, net_input, **kwargs):
        with torch.no_grad():
            encoder_input = {
                k: v for k, v in net_input.items() if k != "prev_output_tokens" and k != "task_name"
            }
            encoder_input.update(kwargs)
            return models[0].generate_speech(**encoder_input)

    def inference_t2s(
        self, models, sample
    ):
        with torch.no_grad():
            xs = sample['net_input']['src_tokens']
            spkemb = sample['net_input']['spkembs']
            return models[0].inference(xs, spkemb)

    def inference_s2s(
        self, models, sample, force_equal_length=False
    ):
        with torch.no_grad():
            x = sample['net_input']['src_tokens']
            xlen = sample['net_input']['src_lengths']
            spkemb = sample['net_input']['spkembs']
            prev_output_tokens = sample['net_input']['prev_output_tokens']
            padding_mask = sample['net_input']['padding_mask']
            tgt_lengths = sample['net_input']['tgt_lengths']
            return models[0].inference_s2s(x, xlen, spkemb, prev_output_tokens, tgt_lengths, force_equal_length=force_equal_length, padding_mask=padding_mask)

    def inference_s2c(
        self, models, sample
    ):
        with torch.no_grad():
            x = sample['net_input']['src_tokens']
            xlen = sample['net_input']['src_lengths']
            prev_output_tokens = sample['net_input']['prev_output_tokens']
            padding_mask = sample['net_input']['padding_mask']
            assert prev_output_tokens.size(1) == 1, prev_output_tokens.size()
            return models[0].inference_s2c(x, xlen, prev_output_tokens, padding_mask=padding_mask)

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """

        indices, ignored = dataset.filter_indices_by_size(
            indices,
            self.max_pos
        )
        return indices
