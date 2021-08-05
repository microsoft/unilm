# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import contextlib
import itertools
import os
import sys
import types

import numpy as np


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


def load_indexed_dataset(path, dictionary, dataset_impl=None, combine=False, default='cached'):
    """A helper function for loading indexed datasets.

    Args:
        path (str): path to indexed dataset (e.g., 'data-bin/train')
        dictionary (~fairseq.data.Dictionary): data dictionary
        dataset_impl (str, optional): which dataset implementation to use. If
            not provided, it will be inferred automatically. For legacy indexed
            data we use the 'cached' implementation by default.
        combine (bool, optional): automatically load and combine multiple
            datasets. For example, if *path* is 'data-bin/train', then we will
            combine 'data-bin/train', 'data-bin/train1', ... and return a
            single ConcatDataset instance.
    """
    from fairseq.data.concat_dataset import ConcatDataset
    import fairseq.data.indexed_dataset as indexed_dataset

    datasets = []
    for k in itertools.count():
        path_k = path + (str(k) if k > 0 else '')

        dataset_impl_k = dataset_impl
        if dataset_impl_k is None:
            dataset_impl_k = indexed_dataset.infer_dataset_impl(path_k)

        dataset = indexed_dataset.make_dataset(
            path_k,
            impl=dataset_impl_k or default,
            fix_lua_indexing=True,
            dictionary=dictionary,
        )
        if dataset is None:
            break
        print('| loaded {} examples from: {}'.format(len(dataset), path_k), flush=True)
        datasets.append(dataset)
        if not combine:
            break
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def collect_filtered(function, iterable, filtered):
    """
    Similar to :func:`filter` but collects filtered elements in ``filtered``.

    Args:
        function (callable): function that returns ``False`` for elements that
            should be filtered
        iterable (iterable): iterable to filter
        filtered (list): list to store filtered elements
    """
    for el in iterable:
        if function(el):
            yield el
        else:
            filtered.append(el)


def _filter_by_size_dynamic(indices, size_fn, max_positions, raise_exception=False):
    def check_size(idx):
        if isinstance(max_positions, float) or isinstance(max_positions, int):
            return size_fn(idx) <= max_positions
        elif isinstance(max_positions, dict):
            idx_size = size_fn(idx)
            assert isinstance(idx_size, dict)
            intersect_keys = set(max_positions.keys()) & set(idx_size.keys())
            return all(
                all(a is None or b is None or a <= b
                    for a, b in zip(idx_size[key], max_positions[key]))
                for key in intersect_keys
            )
        else:
            # Hacky as heck, for the specific case of multilingual training with RoundRobin.
            if isinstance(size_fn(idx), dict) and isinstance(max_positions, tuple):
                return all(
                    a is None or b is None or a <= b
                    for a, b in zip(size_fn(idx).values(), max_positions)
                )
            # For MultiCorpusSampledDataset, will generalize it later
            if not isinstance(size_fn(idx), Iterable):
                return all(size_fn(idx) <= b for b in max_positions)
            return all(
                a is None or b is None or a <= b
                for a, b in zip(size_fn(idx), max_positions)
            )
    ignored = []
    itr = collect_filtered(check_size, indices, ignored)
    indices = np.fromiter(itr, dtype=np.int64, count=-1)
    return indices, ignored


def filter_by_size(indices, dataset, max_positions, raise_exception=False):
    """
    Filter indices based on their size.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset (FairseqDataset): fairseq dataset instance
        max_positions (tuple): filter elements larger than this size.
            Comparisons are done component-wise.
        raise_exception (bool, optional): if ``True``, raise an exception if
            any elements are filtered (default: False).
    """
    if isinstance(max_positions, float) or isinstance(max_positions, int):
        if hasattr(dataset, 'sizes') and isinstance(dataset.sizes, np.ndarray):
            ignored = indices[dataset.sizes[indices] > max_positions].tolist()
            indices = indices[dataset.sizes[indices] <= max_positions]
        elif hasattr(dataset, 'sizes') and isinstance(dataset.sizes, list) and len(dataset.sizes) == 1:
            ignored = indices[dataset.sizes[0][indices] > max_positions].tolist()
            indices = indices[dataset.sizes[0][indices] <= max_positions]
        else:
            indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)
    else:
        indices, ignored = _filter_by_size_dynamic(indices, dataset.size, max_positions)

    if len(ignored) > 0 and raise_exception:
        raise Exception((
            'Size of sample #{} is invalid (={}) since max_positions={}, '
            'skip this example with --skip-invalid-size-inputs-valid-test'
        ).format(ignored[0], dataset.size(ignored[0]), max_positions))
    if len(ignored) > 0:
        print((
            '| WARNING: {} samples have invalid sizes and will be skipped, '
            'max_positions={}, first few sample ids={}'
        ).format(len(ignored), max_positions, ignored[:10]))
    return indices


def batch_by_size_dep(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    try:
        from fairseq.data.data_utils_fast import batch_by_size_fast
    except ImportError:
        raise ImportError(
            'Please build Cython components with: `pip install --editable .` '
            'or `python setup.py build_ext --inplace`'
        )

    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    return batch_by_size_fast(indices, num_tokens_fn, max_tokens, max_sentences, bsz_mult)


def process_bpe_symbol(sentence: str, bpe_symbol: str):
    if bpe_symbol == 'sentencepiece':
        sentence = sentence.replace(' ', '').replace('\u2581', ' ').strip()
    elif bpe_symbol == '_EOW':
        sentence = sentence.replace(' ', '').replace('_EOW', ' ').strip()
    elif bpe_symbol is not None:
        sentence = (sentence + ' ').replace(bpe_symbol, '').rstrip()
    return sentence


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if max_sentences > 0 and len(batch) == max_sentences:
        return 1
    if max_tokens > 0 and num_tokens > max_tokens:
        return 1
    return 0

def batch_by_size(
    indices, num_tokens_fn, max_tokens=None, max_sentences=None,
    required_batch_size_multiple=1,
):
    max_tokens = max_tokens if max_tokens is not None else -1
    max_sentences = max_sentences if max_sentences is not None else -1
    bsz_mult = required_batch_size_multiple
    print("| At batch_by_size ... max_tokens=%d max_sentences=%d" % (max_tokens, max_sentences), flush=True)

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)
    print("| At batch_by_size, fromiter finish len(indices)=%d" % len(indices),
        flush=True)

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []

    i = 0
    while i < len(indices):
        batch = []
        for j in range(i, min(len(indices), i + max_sentences)):
            batch.append(indices[j])
        batches.append(batch)
        i += max_sentences
    print("| At batch_by_size, finish ... ", flush=True)
    return batches


    for i in range(len(indices)):
        idx = indices[i]

        if max_tokens == -1: 
            num_tokens = 0
        else:
            num_tokens = num_tokens_fn(idx)

            sample_lens.append(num_tokens)
            sample_len = max(sample_len, num_tokens)

            assert max_tokens <= 0 or sample_len <= max_tokens, (
                "sentence at index {} of size {} exceeds max_tokens "
                "limit of {}!".format(idx, sample_len, max_tokens)
            )
            num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            if max_tokens != -1:
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    print("| At batch_by_size, finish ... ", flush=True)
    return batches