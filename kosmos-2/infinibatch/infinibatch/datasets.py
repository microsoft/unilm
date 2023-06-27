from .iterators import create_source_iterator, CheckpointableIterator, SelectManyIterator, PrefetchIterator, BufferedShuffleIterator, BlockwiseShuffleIterator, MapIterator
from typing import List, Union, Iterable, Iterator, Callable, Any, Optional, Dict
import os, sys

"""
This module contains common datasets, which are implemented as convenience functions that compose underlying Infinibatch iterators.
"""

def bump_seed(seed: Optional[int], step = 1):
    """
    Helper to bump a random seed if not None.
    """
    return None if seed is None else seed + 1


def chunked_dataset_iterator(chunk_refs: List, read_chunk_fn: Callable[[Any], Iterator], buffer_size: int,
                             train: bool=True,
                             seed: Optional[int]=None, shuffle: bool=True, use_windowed: bool=False,
                             transform: Callable[[Any],Any]=None,
                             prefetch: bool=False,
                             num_instances: int=1, instance_rank: int=0) -> CheckpointableIterator:
    """
    Dataset reading data from gzipped chunks.

    If train=True, this chunks are strided assigned to instances in strides and the data is infinitely repeated in permutations.
    Otherwise, the chunks are split among the instances in consecutive blocks and the data is not repeated.
    This way, when using this dataset for inference on multiple GPUs, to order the outputs in a way that corresponds
    to the original order of the data items in the dataset, one simply has to collect the lists of outputs from each GPU
    and then concatenate these lists in order of increasing rank.
    When using MPI, this can be achieved by a gather-operation to get a list of lists of outputs, one list per GPU,
    followed by flattening the lists back into a single list.

    Args:
        chunk_refs: references (such as path names) to chunk files
        read_chunk_fn: function(chunk_ref) -> Iterator to read a chunk's content into an iterator over its items, e.g. read a file and split into text lines
        train: see above
        shuffle: if true, the data is shuffled. If train is False then shuffle must be False as well.
        buffer_size: size of the buffer in number of samples / data items used for shuffling (default: 2**20)
        transform: transform to be applied to each data item (transform(Any) -> Any)
        prefetch: if True, insert a prefetch iterator with buffer_size
        seed: random seed (or None)
        num_instances: number of instances of this dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        instance_rank: rank of this instance of the dataset. Meant for use with multi-process data loading, e.g., in distributed training.
        use_windowed: temporary option to switch back to the WindowedShuffleIterator (default False). Will go away once shown that we don't need it anymore.
    """
    if not train and shuffle:
        raise ValueError('shuffling is not supported when train=False')
    # set up the chunk reader
    chunks = create_source_iterator(chunk_refs, train=train, seed=seed, shuffle=shuffle, num_instances=num_instances, instance_rank=instance_rank)
    # set up the item reader
    samples = SelectManyIterator(source_iterator=chunks, collection_selector=read_chunk_fn)  # type: CheckpointableIterator
    # wrap the I/O operation in a prefetch iterator
    if prefetch:
        samples = PrefetchIterator(samples, buffer_size)
    # set up the item randomizer
    if shuffle:
        if use_windowed:
            samples = BufferedShuffleIterator(samples, buffer_size, bump_seed(seed, 1))
        else:
            samples = BlockwiseShuffleIterator(samples, buffer_size, bump_seed(seed, 1))
    # apply transform, if given
    if transform is not None:
        samples = MapIterator(samples, transform)
    # this is what we are serving out
    return samples
