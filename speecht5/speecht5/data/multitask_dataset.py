# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import bisect

import logging
import numpy as np
from torch.utils.data.dataloader import default_collate
from fairseq.data import data_utils

from fairseq.data.fairseq_dataset import FairseqDataset

logger = logging.getLogger(__name__)

class MultitaskDataset(FairseqDataset):
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            curr_len = len(e)
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sample_ratios=1, batch_ratio=None):
        super(MultitaskDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        if isinstance(sample_ratios, int):
            sample_ratios = [sample_ratios] * len(self.datasets)
            if batch_ratio is not None:
                logger.info('batch ratio is ' + str(batch_ratio))
                self.batch_ratio = batch_ratio
            else:
                self.batch_ratio = None
        else:
            logger.info('set sample ratio to ' + str(sample_ratios))
            if batch_ratio is not None:
                logger.info('batch ratio is ' + str(batch_ratio))
                self.batch_ratio = batch_ratio
            else:
                self.batch_ratio = None
        self.sample_ratios = sample_ratios
        self._ordered_indices = None
        self._update_size()

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        sample = self.datasets[dataset_idx][sample_idx]
        if isinstance(sample, dict):
            sample["dataset_idx"] = dataset_idx
        else:
            sample = sample + (dataset_idx,)
        return sample

    def _update_size(self):
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.real_sizes = [len(d) for d in self.datasets]

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    def collater(self, samples, **extra_args):
        # For now only supports datasets with same underlying collater implementations
        if samples is not None and len(samples) > 0:
            if isinstance(samples[0], dict):
                dataset_idx = samples[0]["dataset_idx"]
            else:
                dataset_idx = samples[0][-1]
                samples = [sample[:-1] for sample in samples]
        else:
            dataset_idx = 0

        if hasattr(self.datasets[dataset_idx], "collater"):
            return self.datasets[dataset_idx].collater(samples, **extra_args)
        else:
            return default_collate(samples, **extra_args)

    def size(self, idx: int):
        """
        Return an example's size as a float or tuple.
        """
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        return self.datasets[dataset_idx].size(sample_idx)

    def num_tokens(self, index: int):
        return np.max(self.size(index))

    def attr(self, attr: str, index: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        return getattr(self.datasets[dataset_idx], attr, None)

    @property
    def sizes(self):
        _dataset_sizes = []
        for ds in self.datasets:
            if isinstance(ds.sizes, np.ndarray):
                _dataset_sizes.append(ds.sizes)
            else:
                # Only support underlying dataset with single size array.
                assert isinstance(ds.sizes, list)
                _dataset_sizes.append(ds.sizes[0])
        return np.concatenate(_dataset_sizes)

    @property
    def supports_prefetch(self):
        return all(d.supports_prefetch for d in self.datasets)

    def ordered_indices(self):
        # ordered_indices = []
        # for i, dataset in enumerate(self.datasets):
        #     indice = dataset.ordered_indices()
        #     ordered_indices.append(indice)
        if self._ordered_indices is None:
            # Call the underlying dataset's ordered_indices() here, so that we
            # get the same random ordering as we would have from using the
            # underlying sub-datasets directly.
            self._ordered_indices = [
                dataset.ordered_indices()
                for dataset in self.datasets
            ]
        return np.arange(len(self))

    def prefetch(self, indices):
        frm = 0
        for to, ds in zip(self.cumulative_sizes, self.datasets):
            real_size = len(ds)
            if getattr(ds, "supports_prefetch", False):
                ds.prefetch([(i - frm) % real_size for i in indices if frm <= i < to])
            frm = to

    def batch_by_size(
        self,
        indices,
        max_tokens=None,
        max_sentences=None,
        required_batch_size_multiple=1,
    ):
        if not hasattr(self, "max_tokens"):
            self.max_tokens = max_tokens
        if not hasattr(self, "max_sentences"):
            self.max_sentences = max_sentences
        if not hasattr(self, "required_batch_size_multiple"):
            self.required_batch_size_multiple = required_batch_size_multiple
        batch_samplers = []
        for i, dataset in enumerate(self.datasets):
            batch_sampler = dataset.batch_by_size(
                self._ordered_indices[i],
                max_tokens=max_tokens if self.batch_ratio is None else max_tokens * self.batch_ratio[i],
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            if i > 0:
                for batch in batch_sampler:
                    batch += self.cumulative_sizes[i - 1]
            if self.sample_ratios[i] != 1.0:
                batch_sampler = np.array(batch_sampler)
                batch_sampler = np.random.choice(batch_sampler, int(len(batch_sampler) * self.sample_ratios[i]))
                batch_sampler = list(batch_sampler)
            logger.info('Adjust batch by ratio ' + str(self.sample_ratios[i]) + ' and the number of batch is ' + str(int(len(batch_sampler))) + ' for dataset ' + str(i))
            batch_samplers.extend(batch_sampler)
        return batch_samplers

    def filter_indices_by_size(self, indices, max_positions):
        """
        Filter each sub-dataset independently, then update the round robin to work
        on the filtered sub-datasets.
        """
        if not hasattr(self, "max_positions"):
            self.max_positions = max_positions
        ignored_some = False
        for i in range(len(self.datasets)):
            # ignored = []
            self._ordered_indices[i], ignored = self.datasets[i].filter_indices_by_size(
                self._ordered_indices[i], self.max_positions[i]
            )
            if len(ignored) > 0:
                ignored_some = True
                logger.warning(
                    f"{len(ignored)} samples from {i} have invalid sizes and will be skipped, "
                    f"max_positions={self.max_positions[i]}, first few sample ids={ignored[:10]}"
                )

        logger.info('update dataset size')
        self._update_size()

        # Since we are modifying in place the _ordered_indices,
        # it's not possible anymore to return valid ignored indices.
        # Hopefully the extra debug information print above should be enough to debug.
        # Ideally we would receive ignore_invalid_inputs so that we could have
        # a proper error message.
        return (np.arange(len(self)), [0] if ignored_some else [])

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return all(d.can_reuse_epoch_itr_across_epochs for d in self.datasets)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    def shuffle_batches(self, batches, seed):
        logger.info("shuffle batches")
        new_batches_fromlist = []
        new_batches_notlist = []
        new_batches = []
        with data_utils.numpy_seed(seed):
            np.random.shuffle(batches)
            for batch in batches:
                if isinstance(batch, list):
                    # np.random.shuffle(batch)
                    new_batches_fromlist.append(batch)
                else:
                    new_batches_notlist.append(batch)
            logger.info("Get " + str(len(new_batches_fromlist)) + " chunk from speech sides")
            logger.info("Get " + str(sum([len(batch_list) for batch_list in new_batches_fromlist])) + " batches from speech sides")
            logger.info("Get " + str(len(new_batches_notlist)) + " batches from text sides")
            if len(new_batches_fromlist) == 0:
                return new_batches_notlist
            st_ratio = int(len(new_batches_notlist) / len(new_batches_fromlist))
            logger.info("Get st_ratio " + str(st_ratio))
            last_idx = 0
            for i in range(len(new_batches_fromlist)):
                if i == len(new_batches_fromlist) - 1:
                    new_batches_fromlist[i].extend(new_batches_notlist[last_idx:])
                else:
                    new_batches_fromlist[i].extend(new_batches_notlist[last_idx : last_idx + st_ratio])
                np.random.shuffle(new_batches_fromlist[i])
                new_batches.extend(new_batches_fromlist[i])
                last_idx = last_idx + st_ratio
        logger.info("Finish shuffle")
        return new_batches

    def reset_batch_sampler(self):
        logger.info("reset batch sampler")
        self._ordered_indices = [
            self.datasets[i].ordered_indices()
            for i in range(len(self.datasets))
        ]
        self.filter_indices_by_size(None, None)

        batch_samplers = self.batch_by_size(
            None,
            self.max_tokens,
            self.max_sentences,
            self.required_batch_size_multiple
        )
        return batch_samplers
