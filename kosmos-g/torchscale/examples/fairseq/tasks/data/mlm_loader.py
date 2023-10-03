# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy
import itertools
import os

import numpy as np
from infinibatch import iterators

from .basic_loader import BaseBatchGen
from .utils import NativeCheckpointableIterator, WeightIterator


class MLMLoader(BaseBatchGen):
    def __init__(
        self,
        args,
        dataset,
        dictionary,
        tokenizer,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
    ):
        super().__init__()
        self.args = args
        self.data = dataset.data
        self.data_dir = dataset.data_dir
        self.shuffle = dataset.shuffle
        self.dictionary = dictionary
        self.tokenizer = tokenizer

        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.max_positions = max_positions
        self.tokens_per_sample = args.tokens_per_sample
        self.sample_break_mode = args.sample_break_mode
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead

        self._build_iter()

    def _build_iter(self):
        tokenized_lines = self._multilingual_tokenize()
        self.padded_batches = self._batchify(tokenized_lines)

        prefetch_batches = iterators.PrefetchIterator(
            self.padded_batches,
            buffer_size=10000,
            buffer_in_main_process=True,
            log_empty_buffer_warning=True and self.shard_id == 0,
        )

        prefetch_batches = iterators.MapIterator(prefetch_batches, self._move_to_tensor)

        self._iter = prefetch_batches

    def _multilingual_tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(self._tokenize(data))
            if "weight" in data:
                weights.append(float(data["weight"]))
            else:
                weights.append(int(data["count"]))

        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightIterator(weights)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(
            control_iterator, multilingual_iters
        )

        return tokenized_lines

    def _tokenize(self, data):
        """
        data:
        {
            'source': list[Path],
            'source_lang': str,
            'count': int,
            'weight': float,
            'name': str,
        }
        """
        dataset = list(
            zip(
                data["source"],
                itertools.repeat(data["source_lang"]),
            )
        )

        if self.shuffle:
            chunk_files = iterators.InfinitePermutationSourceIterator(
                dataset,
                seed=self.seed,
                shuffle=self.shuffle,
                num_instances=self.num_shards,
                instance_rank=self.shard_id,
            )
        else:
            chunk_files = iterators.ChunkedSourceIterator(
                dataset,
                num_instances=self.num_shards,
                instance_rank=self.shard_id,
            )

        tokenized_lines = iterators.SelectManyIterator(
            chunk_files, lambda files: self._read_from_files(*files)
        )
        tokenized_lines = iterators.SamplingRandomMapIterator(
            tokenized_lines, self._prepare, self.seed
        )

        return tokenized_lines

    def _batchify(self, lines):

        if self.max_sentences is not None:
            if self.batch_read_ahead > 0:
                lines = iterators.BlockwiseShuffleIterator(
                    lines, self.batch_read_ahead, self.seed
                )
            batches = iterators.FixedBatchIterator(lines, self.max_sentences)
        else:

            def dynamic_batch_size(sample):
                lengths = [len(x) for x in sample]
                batch_size = self.max_tokens // max(lengths)
                batch_size = (
                    batch_size
                    // self.required_batch_size_multiple
                    * self.required_batch_size_multiple
                )
                return max(1, batch_size)

            batches = iterators.BucketedReadaheadBatchIterator(
                lines,
                read_ahead=self.batch_read_ahead,
                key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None,
                batch_size=dynamic_batch_size,
                shuffle=self.shuffle,
                seed=self.seed,
            )

        def collate(batch):
            batch_size = len(batch)

            mlm_source_max_length = max([len(x[0]) for x in batch])
            mlm_target_max_length = max([len(x[1]) for x in batch])
            s2s_source_max_length = max([len(x[2]) for x in batch])
            s2s_target_max_length = max([len(x[3]) for x in batch])

            mlm_source_ids = np.full(
                shape=(batch_size, mlm_source_max_length),
                dtype=np.int32,
                fill_value=self.dictionary.pad(),
            )
            mlm_target_ids = np.full(
                shape=(batch_size, mlm_target_max_length),
                dtype=np.int32,
                fill_value=self.dictionary.pad(),
            )
            s2s_source_ids = np.full(
                shape=(batch_size, s2s_source_max_length),
                dtype=np.int32,
                fill_value=self.dictionary.pad(),
            )
            s2s_target_ids = np.full(
                shape=(batch_size, s2s_target_max_length - 1),
                dtype=np.int32,
                fill_value=self.dictionary.pad(),
            )
            s2s_prev_input_ids = np.full(
                shape=(batch_size, s2s_target_max_length - 1),
                dtype=np.int32,
                fill_value=self.dictionary.pad(),
            )

            for i, (
                mlm_input_ids,
                mlm_label_ids,
                s2s_input_ids,
                s2s_label_ids,
            ) in enumerate(batch):
                mlm_source_ids[i, : len(mlm_input_ids)] = mlm_input_ids
                mlm_target_ids[i, : len(mlm_label_ids)] = mlm_label_ids
                s2s_source_ids[i, : len(s2s_input_ids)] = s2s_input_ids
                s2s_target_ids[i, : len(s2s_label_ids) - 1] = s2s_label_ids[1:]
                s2s_prev_input_ids[i, : len(s2s_label_ids) - 1] = s2s_label_ids[:-1]

            ret_batch = {
                "net_input": {
                    "src_tokens": mlm_source_ids.astype(np.int64),
                },
                "target": mlm_target_ids.astype(np.int64),
                "nsentences": batch_size,
                "ntokens": sum([len(x[0]) for x in batch]),
            }

            return ret_batch

        padded_batches = iterators.MapIterator(batches, collate)

        return padded_batches

    def _prepare(self, _random, doc):
        nonmasked_tokens, masked_tokens = self._mask_lm(_random, doc)
        nonnoise_spans, noise_spans = self._span_corruption(_random, doc)
        return nonmasked_tokens, masked_tokens, nonnoise_spans, noise_spans

    def _mask_lm(self, _random, doc):
        def mask_tokens():
            return "<mask>"

        length = len(doc)
        mask_tokens_num = int(length * self.args.mask_prob)
        mask_tokens_num = min(max(mask_tokens_num, 1), length - 1)
        possible_mask_positions = _random.sample(range(length), k=mask_tokens_num)
        possible_mask_positions = sorted(possible_mask_positions)

        nonmasked_tokens = copy.deepcopy(doc)
        masked_tokens = [self.dictionary.pad() for _ in range(len(doc))]

        for position in possible_mask_positions:
            # masked_tokens.append(nonmasked_tokens[position])
            masked_tokens[position] = nonmasked_tokens[position]
            nonmasked_tokens[position] = self.dictionary.indices[mask_tokens()]

        return nonmasked_tokens, masked_tokens

    def _span_corruption(self, _random, doc):
        def mask_tokens(i):
            return f"<mask_{i}>"

        length = len(doc)
        noise_tokens_num = int(length * self.args.mask_prob)
        noise_tokens_num = min(max(noise_tokens_num, 1), length - 1)
        noise_spans_num = int(noise_tokens_num / self.args.span_length)
        noise_spans_num = max(noise_spans_num, 1)
        nonnoise_tokens_num = length - noise_tokens_num

        if noise_spans_num == 1:
            noise_split_positions = [0, noise_tokens_num]
        else:
            possible_split_positions = list(range(1, noise_tokens_num))
            _random.shuffle(possible_split_positions)
            noise_split_positions = sorted(
                possible_split_positions[: noise_spans_num - 1]
            )
            noise_split_positions = [0] + noise_split_positions + [noise_tokens_num]

        possible_insert_positions = list(range(nonnoise_tokens_num))
        _random.shuffle(possible_insert_positions)
        noise_insert_positions = sorted(possible_insert_positions[:noise_spans_num])

        nonnoise_spans, noise_spans = [], []
        last_end = 0
        for i in range(noise_spans_num):
            start_pos = noise_insert_positions[i] + noise_split_positions[i]
            end_pos = noise_insert_positions[i] + noise_split_positions[i + 1]
            mask_id = self.dictionary.indices[mask_tokens(i)]

            if getattr(self.args, "remove_target_sentinel", False):
                noise_spans.append(doc[start_pos:end_pos])
            else:
                noise_spans.append([mask_id] + doc[start_pos:end_pos])

            if getattr(self.args, "remove_source_sentinel", False):
                nonnoise_spans.extend(doc[last_end:start_pos])
            else:
                nonnoise_spans.extend(doc[last_end:start_pos] + [mask_id])

            last_end = end_pos

        nonnoise_spans.extend(doc[last_end:])
        noise_spans = sum(noise_spans, [])

        return nonnoise_spans, noise_spans

    def _read_from_files(self, source_file, source_lang):
        # data = []
        file_path = os.path.join(self.data_dir, source_file)

        if not os.path.exists(file_path):
            print("| file {} not exists".format(file_path), flush=True)
            return iter([])  # skip bad file

        with open(file_path, "r", encoding="utf8") as f:
            lines = f.read().strip().split("\n")

        doc = [self.dictionary.bos()]
        for line in lines:
            if line == "":
                if self.sample_break_mode == "complete_doc":
                    # data.append(doc)
                    yield doc
                    doc = [self.dictionary.bos()]
                continue

            tokenized_line = self.tokenizer.EncodeAsPieces(line)
            tokenized_id = [
                self.dictionary.index(token) for token in tokenized_line
            ] + [self.dictionary.eos_index]

            if len(tokenized_id) > self.tokens_per_sample:
                continue
            if len(doc) + len(tokenized_id) > self.tokens_per_sample:
                # data.append(doc)
                yield doc
                doc = [self.dictionary.bos()]
            doc.extend(tokenized_id)

        if len(doc) > 1 and len(doc) <= self.tokens_per_sample:
            # data.append(doc)
            yield doc

        # return data
