import collections
from random import Random
from typing import Dict, Iterable, Optional

import torch
import numpy as np
from infinibatch import iterators
from infinibatch.iterators import CheckpointableIterator, FixedBatchIterator, SelectManyIterator, MapIterator

from fairseq.data import BaseWrapperDataset, FairseqDataset, data_utils

def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if isinstance(x, np.ndarray):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict(
                (key, _apply(value)) for key, value in x.items()
            )
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


class NativeCheckpointableIterator(iterators.CheckpointableIterator):
    def __init__(self, iterable: Iterable):
        self._input_iterable = iterable
        self.setstate(None)

    def getstate(self) -> Dict:
        return {"num_items_yielded": self._num_items_yielded}

    def setstate(self, checkpoint: Optional[Dict]):
        self._iterator = iter(self._input_iterable)
        self._num_items_yielded = (
            iterators._advance_iterator(self._iterator, checkpoint["num_items_yielded"])
            if checkpoint is not None
            else 0
        )

    def __next__(self):
        item = next(self._iterator)
        self._num_items_yielded += 1
        return item

    def close(self):
        pass


class WeightIterator(object):
    def __init__(self, weights, seed):
        self.weights = weights
        self.seed = seed
        self.control_index = list(range(len(weights)))
        self.setstate(None)

    def __iter__(self):
        return self

    def getstate(self):
        return {"random_state": self._random_state}

    def setstate(self, checkpoint):
        self._random_state = checkpoint["random_state"] if checkpoint else None
        self._random = (
            None  # this will trigger the lazy initialization in self.__next__
        )

    def __next__(self):
        if self._random is None:
            self._random = Random(self.seed)
            if self._random_state is not None:
                self._random.setstate(self._random_state)
        idx = self._random.choices(self.control_index, self.weights)[0]
        self._random_state = self._random.getstate()
        return idx

    def close(self):
        pass


def FixedBlockwiseShuffleIterator(source_iterator: CheckpointableIterator, block_size: int, seed: int=0):
    """
    Shuffles a sequence of items by grouping consecutive items in blocks of fixed size, shuffling
    each block, and yielding the shuffled items of all blocks as a flat sequence.

    E.g. [1, 2, 3, 4, 5, 6, 7, 8] with block_size = 3 may yield [3, 1, 2, 4, 6, 5, 8, 7].

    Args:
        source_iterator: checkpointable iterator or restartable iterable over input items to shuffle
        block_size: size of the buffer in number of items used for shuffling
        seed: random seed used for shuffling (or None)
    """
    # This is implemented as a pipeline:
    #  - group N consecutive items together
    #  - shuffle them
    #  - flatten the result
    blocks = FixedBatchIterator(source_iterator, batch_size=block_size)
    def shuffle_block_fn(block):
        _random = Random(seed)
        _random.shuffle(block)
        return block
    shuffled_blocks = MapIterator(blocks, transform=shuffle_block_fn)
    # samples = SelectManyNoSkipIterator(shuffled_blocks, collection_selector=lambda shuffled_block: iter(shuffled_block))
    samples = SelectManyIterator(shuffled_blocks, collection_selector=lambda shuffled_block: iter(shuffled_block))
    return samples


class IndexIterator(object):
    def __init__(self, num):
        self.num = num
        self.setstate(None)

    def __iter__(self):
        return self

    def getstate(self):
        return {'num_items_yielded': self._num_items_yielded}

    def setstate(self, checkpoint):
        self._num_items_yielded =checkpoint['num_items_yielded'] if checkpoint is not None else 0

    def __next__(self):
        item = self._num_items_yielded % self.num
        self._num_items_yielded += 1
        return item

    def close(self):
        pass


class WeightNoRandomStateIterator(object):
    def __init__(self, weights, seed):
        self.weights = weights
        self.seed = seed
        self.control_index = list(range(len(weights)))
        self.setstate(None)

    def __iter__(self):
        return self

    def getstate(self):
        return {'num_items_yielded': self._num_items_yielded}

    def setstate(self, checkpoint):
        self._num_items_yielded =checkpoint['num_items_yielded'] if checkpoint is not None else 0

    def __next__(self):
        self._random = Random(int(self.seed) + self._num_items_yielded)
        idx = self._random.choices(self.control_index, self.weights)[0]
        self._num_items_yielded += 1
        return idx

    def close(self):
        pass


class SelectManyNoSkipIterator(CheckpointableIterator):
    """
    Projects each element of a source sequence to a sequence and flattens the resulting sequences into one sequence.
    """
    def __init__(self, source_iterator: CheckpointableIterator, collection_selector=None):
        """
        Args:
            source_iterator: iterator over the items to pass to collection_selector()
            collection_selector: user callback that maps an item into an Iterable, whose items will be yielded.
                                 The returned Iterator is used only once. Hence, it is also allowed to
                                 return self-iterables, such as iterators and generator expressions.
                                 If None is given, no callback is applied.
        """
        if not isinstance(source_iterator, CheckpointableIterator):
            raise ValueError('source_iterator has to be a CheckpointableIterator')
        self._source_iterator = source_iterator          # type: CheckpointableIterator
        self._collection_selector = collection_selector  
        self.setstate(None)

    def getstate(self) -> Dict:
        return {'source_state':            self._source_state,
                'flattened_items_yielded': self._flattened_items_yielded}

    def setstate(self, checkpoint: Optional[Dict]):
        self._source_state            = checkpoint['source_state']            if checkpoint else None
        self._flattened_items_yielded =  0
        self._source_iterator.setstate(self._source_state)
        def _generate():
            skip_to_checkpoint = self._flattened_items_yielded
            # main loop over source source_items
            for source_item in self._source_iterator:
                if self._collection_selector is not None:
                    data = iter(self._collection_selector(source_item))
                else:
                    data = iter(source_item)
                self._flattened_items_yielded = 0
                # if skip_to_checkpoint:
                #     #print("Skipping to index", skip_to_checkpoint, file=sys.stderr)
                #     self._flattened_items_yielded += _advance_iterator(data, skip_to_checkpoint)
                #     skip_to_checkpoint = 0
                # main loop over lines
                for item in data:
                    self._flattened_items_yielded += 1
                    yield item
                self._source_state = self._source_iterator.getstate()
        self._iterator = _generate()

    def __next__(self):
        return next(self._iterator)

    def close(self):
        self._source_iterator.close()


class RawArrayDataset(FairseqDataset):

    def __init__(self, dataset, datatype="token"):
        super().__init__()
        self.dataset = dataset
        self.datatype = datatype
        if hasattr(dataset, 'sizes'):
            self._sizes = dataset.sizes
        else:
            try:
                self._sizes = np.array([len(x) for x in self.dataset])
            except:
                self._sizes =  np.array([1 for x in self.dataset])

    def __getitem__(self, index):
        if type(self.dataset[index][0]) != list:
            if self.datatype == "token":
                return torch.Tensor(self.dataset[index]).long()
            else:
                return torch.Tensor(self.dataset[index]).bool()
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if hasattr(self.dataset, 'collater'):
            return self.dataset.collater(samples)
        else:
            raise NotImplementedError()

    @property
    def sizes(self):
        return self._sizes

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)
