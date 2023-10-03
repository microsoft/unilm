import os
import json
from argparse import Namespace
import torch

from fairseq import utils
from fairseq.data import Dictionary
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from dataclasses import dataclass, field
import sentencepiece

from .data.spm_lm_loader import SpmLmLoader as LMLoader
from .data.laion_loader import LaionLoader 
from .data.wild_loader import WildLoader
from .data.utils import EOL_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, image_code_to_token
from .data.basic_loader import ConcatLoader
from .gpt_base import GPTLanguageModelingConfig, GPTPretrainingTask

DEFAULT_ENCODER_JSON = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json"
DEFAULT_VOCAB_BPE = "https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe"
IMAGE_COOEBOOK_SIZE = 8192

@dataclass
class VLGPTLanguageModelingConfig(GPTLanguageModelingConfig):
    wild_data_dir: str = field(default="", metadata={"help": ""})
    wild_batch_size: int = field(default=4, metadata={"help": ""})
    laion_data_dir: str = field(default="", metadata={"help": ""})
    laion_batch_size: int = field(default=32, metadata={"help": ""})



@register_task('vl_gpt_pretraining', dataclass=VLGPTLanguageModelingConfig)
class VLGPTPretrainingTask(LanguageModelingTask):
    def __init__(self, args, dictionary, tokenizer, output_dictionary=None, targets=None):
        super().__init__(args, dictionary, output_dictionary=output_dictionary, targets=targets)
        self.cfg = args
        self.tokenizer = tokenizer
    
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        if len(cfg.dict_path) > 0:
            dictionary = Dictionary.load(cfg.dict_path)
        else:
            dictionary = Dictionary.load(os.path.join(paths[0], "dict.txt"))
        dictionary.add_symbol(EOL_SYMBOL)
        dictionary.add_symbol(BOI_SYMBOL)
        dictionary.add_symbol(EOI_SYMBOL)
        for i in range(IMAGE_COOEBOOK_SIZE):
            dictionary.add_symbol(image_code_to_token(i))

        print('| dictionary: {} types'.format(len(dictionary)))
        output_dictionary = dictionary

        args = cfg
        # upgrade old checkpoints
        if getattr(args, "exclude_self_target", False):
            args.self_target = False

        targets = []
        if getattr(args, "self_target", False):
            targets.append("self")
        if getattr(args, "future_target", False):
            targets.append("future")
        if getattr(args, "past_target", False):
            targets.append("past")
        if len(targets) == 0:
            # standard language modeling
            targets = ["future"]

        if len(cfg.spm_model) > 0:
            tokenizer = sentencepiece.SentencePieceProcessor(model_file=cfg.spm_model)
        else:
            tokenizer = GPT2BPE(Namespace(
                gpt2_vocab_bpe=cfg.gpt2_vocab_bpe,
                gpt2_encoder_json=cfg.gpt2_encoder_json))

        return cls(cfg, dictionary, tokenizer, output_dictionary, targets=targets)
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        if "tnlg" in self.cfg.data and split == "train":
            self.datasets[split] = {
                # 'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                # 'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub-noarvix-nopubmed.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data': json.load(open(f'{self.cfg.data}/json/{split}-nogithub-noarvix-nopubmed-mtnlg.json')) if split == 'train' else json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
            }
        else:
            self.datasets[split] = {
                'data': json.load(open(f'{self.cfg.data}/json/{split}.json')),
                'data_dir': self.cfg.data,
                'shuffle': True if split == 'train' else False,
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
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False
    ):  
        data_loader_list = []

        disable_prefetching = False
        config_split = 'train'
        if not dataset.shuffle: # for valid and test
            shard_id = 0
            disable_prefetching = True
            config_split = 'valid'

        if self.cfg.wild_data_dir:
            wild_dataset = Namespace(**{
                'data': json.load(open(f'{self.cfg.wild_data_dir}/json/{config_split}.json')),
                'data_dir': self.cfg.wild_data_dir,
                'shuffle': dataset.shuffle})

            wild_vl_loader = WildLoader(
                self.cfg,
                wild_dataset,
                self.dictionary,
                self.tokenizer,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
                disable_prefetching=disable_prefetching,
                data_name='wild'
            )
            data_loader_list.append(wild_vl_loader)

        if self.cfg.laion_data_dir:
            laion_dataset = Namespace(**{
                'data': json.load(open(f'{self.cfg.laion_data_dir}/json/{config_split}.json')),
                'data_dir': self.cfg.laion_data_dir,
                'shuffle': dataset.shuffle})

            lain_vl_loader = LaionLoader(
                self.cfg,
                laion_dataset,
                self.dictionary,
                self.tokenizer,
                max_tokens=max_tokens,
                max_sentences=self.cfg.laion_batch_size,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
                disable_prefetching=disable_prefetching,
                data_name='laion'
            )
            data_loader_list.append(lain_vl_loader)

        lm_loader = LMLoader(
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
                epoch=epoch,
                num_shards=num_shards,
                shard_id=shard_id,
                disable_prefetching=disable_prefetching,
        )
        data_loader_list.append(lm_loader)

        concat_loader = ConcatLoader(data_loader_list)
        return concat_loader

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample['gpt'], loss_name='gpt')
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        
        agg_loss += loss.detach().item()
        agg_sample_size += sample_size
        agg_logging_output.update(logging_output)
        
        if 'laion' in sample:
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, sample['laion'], loss_name='laion')
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
            
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            for key, value in logging_output.items():
                if key not in agg_logging_output:
                    agg_logging_output[key] = value
                else:
                    agg_logging_output[key] += value

        if 'wild' in sample:
            with torch.autograd.profiler.record_function("forward"):
                loss, sample_size, logging_output = criterion(model, sample['wild'], loss_name='wild')
            if ignore_grad:
                loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(loss)
            
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            for key, value in logging_output.items():
                if key not in agg_logging_output:
                    agg_logging_output[key] = value
                else:
                    agg_logging_output[key] += value

        return agg_loss, agg_sample_size, agg_logging_output


    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample['gpt'])
        return loss, sample_size, logging_output
