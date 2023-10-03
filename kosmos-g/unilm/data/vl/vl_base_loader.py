import json
import os
import multiprocessing
import itertools

from infinibatch import iterators
from functools import partial

try:
    from fairseq.data.encoders.gpt2_bpe import GPT2BPE
except:
    print("GPT2BPE not found, please install fairseq first if you want to use GPT2BPE")

import glob
import os
import torch
import numpy as np
import time
import json
import random
import itertools
import hydra
import copy

from infinibatch import iterators
from unilm.data.basic_loader import BaseBatchGen
from unilm.data.utils import NativeCheckpointableIterator, WeightIterator


class VLBaseLoader(BaseBatchGen):

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
            epoch=1,
            num_shards=1,
            shard_id=0,
            no_prefetch=False,
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
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.required_batch_size_multiple = required_batch_size_multiple
        self.seed = str(seed)
        self.epoch = epoch
        self.num_shards = num_shards
        self.shard_id = shard_id

        self.batch_read_ahead = args.batch_read_ahead
        self.no_prefetch = no_prefetch

        # build filter and transform
        self._setup()
        self.filters = self._build_filter()
        self.image_transform = self._build_image_transform()
        self.text_transform = self._build_text_transform()

        self._build_iter()

    def getstate(self):
        state = super().getstate()
        state["epoch"] = self.epoch
        state["iterations_in_epoch"] = None
        return state

    def _setup(self):
        pass

    def _build_filter(self):
        raise NotImplementedError

    def _build_image_transform(self):
        raise NotImplementedError
    
    def _build_text_transform(self):
        raise NotImplementedError

    def _build_iter(self):
        # read data, filter, and transform
        tokenized_lines = self._tokenize()

        # batchify and collate
        self.padded_batches = self._batchify(tokenized_lines)
        
        if self.no_prefetch: # no prefetch is for debug
            prefetch_batches = self.padded_batches
        else:
            prefetch_batches = iterators.PrefetchIterator(
                self.padded_batches, 
                buffer_size=self.args.batch_read_ahead,
                buffer_in_main_process=True, 
                log_empty_buffer_warning=True and self.shard_id == 0,
            )

        prefetch_batches = iterators.MapIterator(
            prefetch_batches, self._move_to_tensor
        )

        self._iter = prefetch_batches

    def _tokenize(self):
        multilingual_iters = []
        weights = []

        for data in self.data:
            multilingual_iters.append(
                self._tokenize_foreach_lang(data)
            )
            if 'weight' in data:
                weights.append(float(data['weight']))
            else:
                weights.append(int(data['count']))
        
        if len(multilingual_iters) == 1:
            return multilingual_iters[0]

        sampling_iterator = WeightIterator(weights, self.seed)
        control_iterator = NativeCheckpointableIterator(sampling_iterator)
        tokenized_lines = iterators.MultiplexIterator(control_iterator, multilingual_iters)
        
        return tokenized_lines

    def _tokenize_foreach_lang(self, data):
        dataset = list(zip(data['source']))
        if self.shuffle:
            chunk_files = iterators.InfinitePermutationSourceIterator(
                dataset,
                seed=self.seed, 
                shuffle=self.shuffle, 
                num_instances=self.num_shards, 
                instance_rank=self.shard_id,)
        else:
            chunk_files = iterators.ChunkedSourceIterator(
                dataset,
                num_instances=self.num_shards, 
                instance_rank=self.shard_id,)
        
        # read file based filter in self._read_from_files function
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        
        # add image/text transform in self._prepare function
        tokenized_lines = iterators.SamplingRandomMapIterator(tokenized_lines, self._prepare, self.seed)
        
        return tokenized_lines

    # def _batchify(self, lines):
        
    #     if self.max_sentences is not None:
    #         if self.batch_read_ahead > 0:
    #             lines = iterators.BlockwiseShuffleIterator(lines, self.batch_read_ahead, self.seed)
    #         batches = iterators.FixedBatchIterator(lines, self.max_sentences)
    #     else:
    #         # -
    #         def dynamic_batch_size(sample):
    #             lengths = [len(x) for x in sample]
    #             batch_size = self.max_tokens // max(lengths) // self.required_batch_size_multiple * self.required_batch_size_multiple
    #             return max(1, batch_size)
            
    #         batches = iterators.BucketedReadaheadBatchIterator(
    #                 lines,
    #                 read_ahead=self.batch_read_ahead, 
    #                 key=(lambda x: max(len(x[0]), len(x[1]))) if self.shuffle else None, 
    #                 batch_size=dynamic_batch_size, 
    #                 shuffle=self.shuffle,
    #                 seed=self.seed,
    #         )

    #     def collate(batch):
    #         batch_size = len(batch)
    #         mlm_batch_size = sum([len(x[2]) for x in batch]) 

    #         gpt_max_length = max([len(x[0]) for x in batch])

    #         mlm_max_length = 0
    #         mlm_ntokens = 0
    #         for x in batch:
    #             for y in x[2]:
    #                 mlm_max_length = max(mlm_max_length, len(y))
    #                 mlm_ntokens += len(y)

    #         gpt_source_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
    #                              fill_value=self.dictionary.pad())
    #         gpt_target_ids = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32,
    #                                  fill_value=self.dictionary.pad())
    #         audio_source_ids = np.full(shape=(batch_size, self.audio_segment_size, self.max_audio_token_length * self.audio_downsample_rate, self.n_bins), dtype=np.float32,
    #                              fill_value=self.dictionary.pad())
    #         gpt_input_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=0)
    #         gpt_loss_mask_all = np.full(shape=(batch_size, gpt_max_length-1), dtype=np.int32, fill_value=1)
    #         audio_mask_all = np.full(shape=(batch_size, self.audio_segment_size, self.max_audio_token_length), dtype=np.int32, fill_value=0)

    #         for i, (full_tokens, gpt_input_mask, audio_tokens, audio_masks, gpt_loss_mask) in enumerate(batch):
    #             gpt_source_ids[i, :len(full_tokens)-1] = full_tokens[:-1]
    #             gpt_target_ids[i, :len(full_tokens)-1] = full_tokens[1:]
    #             gpt_input_mask_all[i, :len(full_tokens)-1] = gpt_input_mask[:-1]
    #             gpt_loss_mask_all[i, :len(full_tokens)-1] = gpt_loss_mask[1:]
    #             audio_source_ids[i] = audio_tokens
    #             audio_mask_all[i] = audio_masks
            
    #         ret_batch = {
    #             'audio':{
    #                 'net_input': {
    #                     'src_tokens': gpt_source_ids.astype(np.int64),
    #                     'aud_src_tokens': audio_source_ids.astype(np.float32),
    #                     'aud_gpt_input_mask': gpt_input_mask_all.astype(np.bool_),
    #                     'gpt_loss_mask': gpt_loss_mask_all.astype(np.bool_),
    #                     'aud_mask': audio_mask_all.astype(np.bool_)
    #                 },
    #                 'target': gpt_target_ids.astype(np.int64),
    #                 'nsentences': batch_size,
    #                 'ntokens': sum([len(x[0]) for x in batch]),
    #             }
    #         }

    #         return ret_batch

    #     padded_batches = iterators.MapIterator(
    #         batches, collate
    #     )

    #     return padded_batches

    # def _prepare(self, _random, doc):
    #     """
    #     speech json line to ids,
    #     <s> <audio> <audio> * N </audio> Text Text text <audio> <audio> * N </audio> Text text </s>
    #     """
    #     full_tokens, gpt_input_mask, audio_tokens, audio_masks, gpt_loss_mask = self._audio_cut(_random, doc)
    #     return full_tokens, gpt_input_mask, audio_tokens, audio_masks, gpt_loss_mask

    # def _audio_cut(self, _random, doc):
    #     """
    #     # start with audio
    #     # full token: <s> <audio> <audio> * N </audio> Text Text text <audio> <audio> * N </audio> Text text </s>
    #     # gpt_input_mask: 0 0           1 * N    0      0    ...          0     1 * N        0      0    0    0
    #     # gpt_loss_mask:  0 0           0 * 1    1      1    ...          0     0 * 1        1      1    1    1
    #     """

    #     def audio_token_segment(audio_len):
    #         #TODO: now we use image tag to represent audio, need to change to audio tag
    #         boi_id = self.dictionary.index("<image>")
    #         eoi_id = self.dictionary.index("</image>")
    #         return [boi_id] + [boi_id] * audio_len + [eoi_id]

    #     # import pudb; pu.db
    #     full_tokens, gpt_input_mask, audio_tokens, audio_masks, gpt_loss_mask = [], [], [], [], []
        
    #     full_tokens.append(self.dictionary.bos())
    #     gpt_input_mask.append(0)
    #     gpt_loss_mask.append(0)

    #     shuffled_ids = [i for i in range(1, len(doc)) if int(doc[i]['size']) < self.max_single_audio_length]
    #     _random.shuffle(shuffled_ids)
    #     audio_part = shuffled_ids[:self.audio_segment_size - 1]

    #     for i, segment in enumerate(doc):
    #         wav_path = os.path.join(self.audio_root_path, segment['file']) 
    #         if i == 0 or i in audio_part:
    #             # fbank = get_fbank(wav_path, output_sample_rate=16000, n_bins=40)
    #             # NOTE: it will be blocked in get_fbank (kaldi feature extraction)
    #             # waveform, sample_rate = get_waveform(wav_path, normalization=False, output_sample_rate=self.output_sample_rate)
    #             # TODO: use tensor as input, if the data loader blocked, we can use other format to speed up

    #             fbank = np.load(wav_path)
    #             fbank_len = fbank.shape[0]
    #             audio_hidden_size = fbank_len // self.audio_downsample_rate
    #             if self.audio_tokens_size != 0:
    #                 audio_tokens_size = self.audio_tokens_size
    #             else:
    #                 audio_tokens_size = audio_hidden_size
    #             full_tokens += audio_token_segment(audio_tokens_size)
    #             gpt_input_mask += [0] + [1] * audio_tokens_size + [0]
    #             gpt_loss_mask +=  [0] + [0] * audio_tokens_size + [1]
    #             audio_token = np.zeros((self.max_audio_token_length * self.audio_downsample_rate, self.n_bins))
    #             audio_token[:fbank_len] = fbank
    #             audio_tokens.append(audio_token)
    #             audio_mask = np.zeros(self.max_audio_token_length)
    #             audio_mask[audio_hidden_size:] = 1
    #             audio_masks.append(audio_mask)
    #         else:
    #             full_tokens += segment['text']
    #             gpt_input_mask += [0] * len(segment['text'])
    #             gpt_loss_mask += [1] * len(segment['text'])

    #     return full_tokens, gpt_input_mask, audio_tokens, audio_masks, gpt_loss_mask

    # @staticmethod
    # def fs_encode_line(fs_dict, words, append_eos=True):
    #     ids = []
    #     for i, word in enumerate(words):
    #         idx = fs_dict.index(word)
    #         ids.append(idx)
    #     if append_eos:
    #         ids.append(fs_dict.eos_index)
    #     return ids

    # @staticmethod
    # def _doc_jsonstr_to_ids(doc_jsonstr, spm_tokenizer=None, fs_dict=None):
    #     """
    #     """
    #     doc = json.loads(doc_jsonstr)
    #     for item in doc:
    #         line = item['text']
    #         if isinstance(spm_tokenizer, GPT2BPE):
    #             tokens = spm_tokenizer.encode(line).split(' ')
    #         else:
    #             tokens = spm_tokenizer.encode(line, out_type=str)
    #         tokenized_ids = SpeechLoader.fs_encode_line(fs_dict, tokens, append_eos=False)
    #         item['text'] = tokenized_ids
    #     return doc
        
    # def _read_from_files(self, source_file):
    #     def judge_doc(doc):
    #         if len(doc) > self.audio_segment_size and int(doc[0]['size']) < self.max_single_audio_length:
    #             for item in doc[1:]:
    #                 if int(item['size']) < self.max_single_audio_length:
    #                     return True
    #         return False
    #     data = []
    #     file_path = os.path.join(self.data_dir, source_file)
        
    #     if not os.path.exists(file_path):
    #         print('| file {} not exists'.format(file_path), flush=True)
    #         return iter([]) # skip bad file

    #     try:
    #         with open(file_path, 'r', encoding='utf8') as f:
    #             lines = f.read().strip().split('\n')
    #     except:
    #         return iter([]) # skip bad file

    #     for doc_jsonstr in lines:
    #         ret = SpeechLoader._doc_jsonstr_to_ids(doc_jsonstr, spm_tokenizer=self.tokenizer, fs_dict=self.dictionary)
    #         if len(ret) <= self.audio_segment_size:
    #             continue

    #         # break the doc based self.max_text_length = 448  # 2028 - 800 - 800
    #         doc = []
    #         doc_text_length = 0
    #         for i in range(0, len(ret)):
    #             if i == 0:
    #                 doc.append(ret[i])
    #                 doc_text_length += len(ret[i]['text'])
    #             else:
    #                 if doc_text_length + len(ret[i]['text']) > self.max_text_length:
    #                     if judge_doc(doc):
    #                         data.append(doc)
    #                     doc = [ret[i]]
    #                     doc_text_length = len(ret[i]['text'])
    #                 else:
    #                     doc.append(ret[i])
    #                     doc_text_length += len(ret[i]['text'])
    #         if judge_doc(doc):
    #             data.append(doc)
    #     return data

