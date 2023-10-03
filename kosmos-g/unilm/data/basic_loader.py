import math
import re
import sys
import time
import torch
from infinibatch.iterators import CheckpointableIterator
from unilm.data import utils
from unilm.data.utils import ConcatIterator, MixIterator


class BaseBatchGen(CheckpointableIterator):
    """
    This is a base class for batch generators that use infinibatch
    """

    def __init__(self):
        self._iter = None
        self.epoch = 1
        self.next_epoch_idx = 1
        self.sharded_checkpoint = False
        self.should_close_after_finished = True

    def _build_iter(self):
        """
        Build infinibatch iterator and assign to self._iter
        """
        raise NotImplementedError()
    
    def _move_to_tensor(self, batch):

        def to_tensor(x):
            return torch.tensor(x)

        return utils.apply_to_sample(to_tensor, batch)

    @property
    def iterator(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __iter__(self):
        if self._iter is None:
            raise NotImplementedError("_build_iter() must called first")
        return self._iter

    def __next__(self):
        return next(self._iter)
    
    def setstate(self, value):
        self._iter.setstate(value)
    
    def getstate(self):
        return self._iter.getstate()
    
    def close(self):
        self._iter.close()
    
    def __len__(self) -> int:
        return 819200000

    def next_epoch_itr(
        self, shuffle=True, fix_batches_to_gpus=False, set_dataset_epoch=True
    ):
        return self

    def end_of_epoch(self) -> bool:
        return False

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return self.getstate()

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.setstate(state_dict)

    @property
    def first_batch(self):
        return "DUMMY"


class ConcatLoader(BaseBatchGen):
    def __init__(self, dataloaders):
        super().__init__()
        self.dataloaders = list(dataloaders)
        self._build_iter()

    def _build_iter(self):
        """
        Build infinibatch iterator and assign to self._iter
        """
        self._iter = ConcatIterator([dataloader.iterator for dataloader in self.dataloaders])


class MixLoader(BaseBatchGen):
    def __init__(self, dataloaders, weights):
        super().__init__()
        self.dataloaders = list(dataloaders)
        self.weights = weights
        self._build_iter()

    def _build_iter(self):
        """
        Build infinibatch iterator and assign to self._iter
        """
        self._iter = MixIterator([dataloader.iterator for dataloader in self.dataloaders], self.weights)
