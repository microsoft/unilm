import os
from typing import Optional
import json
from argparse import Namespace
import torch

from fairseq.tasks import register_task, FairseqDataclass, FairseqTask
from dataclasses import dataclass, field
from omegaconf import II

from .data.lm_loader import LMLoader
from .data.tiktoken_tokenizer import TiktokenTokenizer
from .data.llama_tokenizer import LLaMATokenizer


@dataclass
class GPTLanguageModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    tokens_per_sample: int = field(
        default=1024,
        metadata={"help": "max number of tokens per sample for LM dataset"},
    )
    max_target_positions: Optional[int] = field(
        default=None, metadata={"help": "max number of tokens in the target sequence"}
    )
    llama_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to load tokenizer and config"},
    )
    tiktoken_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "tiktoken model to tokenize the data"
        },
    )
    batch_read_ahead: int = field(
        default=10000,
        metadata={"help": "batch read ahead size for infinibatch"},
    )
    pad_to_max_len: bool = field(
        default=False,
        metadata={"help": "pad each sentence to max length"},
    )
    absolute_path: bool = field(
        default=False,
        metadata={"help": "use absolute path in data config"},
    )
    tokenizer_pad_to_multiple: int = field(
        default=8,
        metadata={"help": "pad to multiple of this value"},
    )
    seed: int = II("common.seed")
    batch_size: Optional[int] = II("dataset.batch_size")


@register_task('gpt', dataclass=GPTLanguageModelingConfig)
class GPTPretrainingTask(FairseqTask):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.cfg = args
        self.tokenizer = tokenizer
    
    @classmethod
    def setup_task(cls, cfg, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        if cfg.llama_model is not None:
            tokenizer = LLaMATokenizer(os.path.join(cfg.llama_model, "tokenizer.model"))
        elif cfg.tiktoken_model is not None:
            tokenizer = TiktokenTokenizer(cfg.tiktoken_model, cfg.tokenizer_pad_to_multiple)
        else:
            raise ValueError("No tokenizer model provided")

        return cls(cfg, tokenizer)
    
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
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
        return LMLoader(
                self.cfg,
                dataset,
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
        )

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
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
    
    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    @property
    def target_dictionary(self):
        padding_idx = self.tokenizer.pad_id
        class Dict:
            def pad(self):
               return padding_idx
        dictionary = Dict()
        return dictionary
        
