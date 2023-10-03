import os
import gzip
from sre_parse import SPECIAL_CHARS
import numpy as np
from random import Random
from typing import Any, Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple, Union
import collections
from infinibatch import iterators

EOD_SYMBOL = "</doc>"
BOI_SYMBOL = "<image>"
EOI_SYMBOL = "</image>"
EOC_SYMBOL = "</chunk>"
EOL_SYMBOL = "</line>"

SPECIAL_SYMBOLS = [EOD_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, EOC_SYMBOL, EOL_SYMBOL]


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if isinstance(x, np.ndarray):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict((key, _apply(value)) for key, value in x.items())
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
        return {'num_items_yielded': self._num_items_yielded}

    def setstate(self, checkpoint: Optional[Dict]):
        self._iterator = iter(self._input_iterable)
        self._num_items_yielded = iterators._advance_iterator(self._iterator, checkpoint['num_items_yielded']) if checkpoint is not None else 0

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
        self._random = None  # this will trigger the lazy initialization in self.__next__
        
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


class ConcatIterator(iterators.CheckpointableIterator):
    """
    Concat items from all given iterators.
    """
    def __init__(self, source_iterators):
        """
        Args:
                source_iterators: list of iterators to zip, item by item
        """
        # TODO: Use all function?
        for source_iterator in source_iterators:
            if not isinstance(source_iterator, iterators.CheckpointableIterator):
                raise ValueError('all iterators in source_iterators have to be CheckpointableIterator')
        self._source_iterators = source_iterators        # type: List[CheckpointableIterator]

    def getstate(self):
        return {'input_states': tuple(iterator.getstate() for iterator in self._source_iterators)}

    def setstate(self, checkpoint):
        if checkpoint is None:
            for iterator in self._source_iterators:
                iterator.setstate(None)
        else:
            # TODO: Add check that both lists have the same length?
            for iterator, state in zip(self._source_iterators, checkpoint['input_states']):
                iterator.setstate(state)

    def __next__(self):
        res = {}    # (note: can't use a generator expression, as it gets confused when a next() call raises StopIteration)
        for iterator in self._source_iterators:
            res.update(next(iterator))
        return res
    
    def close(self):
        for it in self._source_iterators:
            it.close()


class MixIterator(iterators.CheckpointableIterator):
    """
    Concat items from all given iterators.
    """
    def __init__(self, source_iterators, weights):
        """
        Args:
                source_iterators: list of iterators to zip, item by item
        """
        # TODO: Use all function?
        for source_iterator in source_iterators:
            if not isinstance(source_iterator, iterators.CheckpointableIterator):
                raise ValueError('all iterators in source_iterators have to be CheckpointableIterator')
        self._source_iterators = source_iterators        # type: List[CheckpointableIterator]
        assert len(weights) == len(source_iterators)
        self.weights = weights
        self.population = list(range(len(source_iterators)))

    def getstate(self):
        return {'input_states': tuple(iterator.getstate() for iterator in self._source_iterators)}

    def setstate(self, checkpoint):
        if checkpoint is None:
            for iterator in self._source_iterators:
                iterator.setstate(None)
        else:
            # TODO: Add check that both lists have the same length?
            for iterator, state in zip(self._source_iterators, checkpoint['input_states']):
                iterator.setstate(state)

    def __next__(self):
        _random = Random()
        res = {}    # (note: can't use a generator expression, as it gets confused when a next() call raises StopIteration)
        idx = _random.choices(self.population, self.weights)[0]
        res.update(next(self._source_iterators[idx]))
        return res
    
    def close(self):
        for it in self._source_iterators:
            it.close()
