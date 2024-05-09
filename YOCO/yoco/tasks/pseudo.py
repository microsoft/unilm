import os
from typing import Optional
import torch

from fairseq.data import FairseqDataset
from fairseq.tasks import register_task, FairseqDataclass, LegacyFairseqTask
from dataclasses import dataclass, field
from omegaconf import II

from .data.tiktoken_tokenizer import TiktokenTokenizer
from .data.llama_tokenizer import LLaMATokenizer


class PseudoIterator(FairseqDataset):
    def __init__(self, batch_size, length, vocab_size):
        super().__init__()
        self.batch_size = batch_size
        self.length = length
        self.vocab_size = vocab_size

        self.epoch = 1
        self.next_epoch_idx = 1
        self.sharded_checkpoint = True
        self.should_close_after_finished = True

    def __iter__(self):
        while True:
            yield self.__next__()

    def __next__(self):
        net_input = torch.randint(size=(self.batch_size, self.length), dtype=torch.long, low=0, high=self.vocab_size - 1)
        return {
            "net_input": {"src_tokens": net_input},
            "target": net_input,
            "ntokens": self.batch_size * self.length,
        }

    def __len__(self) -> int:
        return 819200000

    def next_epoch_itr(self, **kwargs):
        return self

    @property
    def first_batch(self):
        return "DUMMY"

    def end_of_epoch(self) -> bool:
        return False

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

    def setstate(self, value):
        pass
    
    def getstate(self):
        pass
    
    def close(self):
        pass

@dataclass
class PseudoConfig(FairseqDataclass):
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


@register_task('pseudo', dataclass=PseudoConfig)
class PseudoTask(LegacyFairseqTask):
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

    def load_dataset(self, split, **kwargs):
        pass
        # self.datasets[split] = None

    def dataset(self, split):
        return None
    
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
        return PseudoIterator(max_sentences, self.cfg.tokens_per_sample, 10000)

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