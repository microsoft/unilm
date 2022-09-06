import os
from fairseq import search

from fairseq import scoring, utils, metrics
from fairseq.data import Dictionary, encoders
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.fairseq_task import FairseqTask

try:
    from .data import SROIETextRecognitionDataset, Receipt53KDataset, SyntheticTextRecognitionDataset
    from .data_aug import build_data_aug, OptForDataAugment, DataAugment
except ImportError:
    from data import SROIETextRecognitionDataset, Receipt53KDataset, SyntheticTextRecognitionDataset
    from data_aug import build_data_aug, OptForDataAugment, DataAugment

import logging
import torch


logger = logging.getLogger(__name__)


@register_task('text_recognition')
class TextRecognitionTask(LegacyFairseqTask):
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('data', metavar='DIR',
                            help='the path to the data dir')         

        parser.add_argument('--reset-dictionary', action='store_true',
                            help='if reset dictionary and related parameters')
        parser.add_argument('--adapt-dictionary', action='store_true',
                            help='if adapt dictionary and related parameters')
        parser.add_argument('--adapt-encoder-pos-embed', action='store_true',
                            help='if adapt encoder pos embed')

        parser.add_argument('--add-empty-sample', action='store_true',
                            help='add empty samples to the dataset (for multilingual dataset).')
        parser.add_argument('--preprocess', default='ResizeNormalize', type=str,
                            help='the image preprocess methods (ResizeNormalize|DeiT)')     
        parser.add_argument('--decoder-pretrained', default=None, type=str,
                            help='seted to load the RoBERTa parameters to the decoder.')    
        parser.add_argument('--decoder-pretrained-url', default=None, type=str,
                            help='the ckpt url for decoder pretraining (only unilm for now)')     
        parser.add_argument('--dict-path-or-url', default=None, type=str,
                            help='the local path or url for dictionary file')                          
        parser.add_argument('--input-size', type=int, nargs='+', help='images input size', required=True)
        parser.add_argument('--data-type', type=str, default='SROIE',
                            help='the dataset type used for the task (SROIE or Receipt53K)')        

        # Augmentation parameters
        parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                            help='Color jitter factor (default: 0.4)')
        parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                            help='Use AutoAugment policy. "v0" or "original". " + \
                                "(default: rand-m9-mstd0.5-inc1)'),
        parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
        parser.add_argument('--train-interpolation', type=str, default='bicubic',
                            help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

        parser.add_argument('--repeated-aug', action='store_true')
        parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
        parser.set_defaults(repeated_aug=True)

        # * Random Erase params
        parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                            help='Random erase prob (default: 0.25)')
        parser.add_argument('--remode', type=str, default='pixel',
                            help='Random erase mode (default: "pixel")')
        parser.add_argument('--recount', type=int, default=1,
                            help='Random erase count (default: 1)')
        parser.add_argument('--resplit', action='store_true', default=False,
                            help='Do not random erase first (clean) augmentation split')
                   
    @classmethod
    def setup_task(cls, args, **kwargs):
        import urllib.request
        import io            

        if getattr(args, "dict_path_or_url", None) is not None:
            if args.dict_path_or_url.startswith('http'):
                logger.info('Load dictionary from {}'.format(args.dict_path_or_url))  
                dict_content = urllib.request.urlopen(args.dict_path_or_url).read().decode()
                dict_file_like = io.StringIO(dict_content)
                target_dict = Dictionary.load(dict_file_like)
            else:
                target_dict = Dictionary.load(args.dict_path_or_url)      
        elif getattr(args, "decoder_pretrained", None) is not None:
            if args.decoder_pretrained == 'unilm':            
                url = 'https://layoutlm.blob.core.windows.net/trocr/dictionaries/unilm3.dict.txt'
                logger.info('Load unilm dictionary from {}'.format(url))            
                dict_content = urllib.request.urlopen(url).read().decode()
                dict_file_like = io.StringIO(dict_content)
                target_dict = Dictionary.load(dict_file_like)
            elif args.decoder_pretrained.startswith('roberta'):
                url = 'https://layoutlm.blob.core.windows.net/trocr/dictionaries/gpt2_with_mask.dict.txt'
                logger.info('Load gpt2 dictionary from {}'.format(url))            
                dict_content = urllib.request.urlopen(url).read().decode()
                dict_file_like = io.StringIO(dict_content)
                target_dict = Dictionary.load(dict_file_like)
            else:
                raise ValueError('Unknown decoder_pretrained: {}'.format(args.decoder_pretrained))
        else:
            raise ValueError('Either dict_path_or_url or decoder_pretrained should be set.')
          
        logger.info('[label] load dictionary: {} types'.format(len(target_dict)))

        return cls(args, target_dict)

    def __init__(self, args, target_dict):

        super().__init__(args)
        self.args = args
        self.data_dir = args.data            
        self.target_dict = target_dict
        if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] != '0':
            torch.distributed.barrier()
        self.bpe = self.build_bpe(args)
        if 'LOCAL_RANK' in os.environ and os.environ['LOCAL_RANK'] == '0':
            torch.distributed.barrier()

    def load_dataset(self, split, **kwargs):

        input_size = self.args.input_size         
        if isinstance(input_size, list):
            if len(input_size) == 1:
                input_size = (input_size[0], input_size[0])
            else:
                input_size = tuple(input_size)
        elif isinstance(input_size, int):
            input_size = (input_size, input_size)

        logger.info('The input size is {}, the height is {} and the width is {}'.format(input_size, input_size[0], input_size[1]))

        if self.args.preprocess == 'DA2':            
            tfm = build_data_aug(input_size, mode=split)     
        elif self.args.preprocess == 'RandAugment':
            opt = OptForDataAugment(eval= (split != 'train'), isrand_aug=True, imgW=input_size[1], imgH=input_size[0], intact_prob=0.5, augs_num=3, augs_mag=None)
            tfm = DataAugment(opt)
        else:
            raise Exception('Undeined image preprocess method.')        
        
        # load the dataset
        if self.args.data_type == 'SROIE':
            root_dir = os.path.join(self.data_dir, split)
            self.datasets[split] = SROIETextRecognitionDataset(root_dir, tfm, self.bpe, self.target_dict)        
        elif self.args.data_type == 'Receipt53K':
            gt_path = os.path.join(self.data_dir, 'gt_{}.txt'.format(split))            
            self.datasets[split] = Receipt53KDataset(gt_path, tfm, self.bpe, self.target_dict)
        elif self.args.data_type == 'STR':
            gt_path = os.path.join(self.data_dir, 'gt_{}.txt'.format(split))            
            self.datasets[split] = SyntheticTextRecognitionDataset(gt_path, tfm, self.bpe, self.target_dict)
        else:
            raise Exception('Not defined dataset type: ' + self.args.data_type)
    
    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):        
        return self.target_dict

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        try:
            from .generator import TextRecognitionGenerator
        except:
            from generator import TextRecognitionGenerator

        try:
            from fairseq.fb_sequence_generator import FBSequenceGenerator
        except ModuleNotFoundError:
            pass

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            elif getattr(args, "fb_seq_gen", False):
                seq_gen_cls = FBSequenceGenerator
            else:
                seq_gen_cls = TextRecognitionGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            **extra_gen_cls_kwargs,
        )

    def filter_indices_by_size(
        self, indices, dataset, max_positions=None, ignore_invalid_inputs=False
    ):
        return indices