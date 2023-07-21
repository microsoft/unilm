import json
import os
import multiprocessing
import itertools

from infinibatch import iterators
from functools import partial
from .lm_loader import LMLoader
from .utils import NativeCheckpointableIterator, WeightIterator, EOL_SYMBOL
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from tiktoken.core import Encoding


class SpmLmLoader(LMLoader):
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
        
        tokenized_lines = iterators.SelectManyIterator(chunk_files, lambda files: self._read_from_files(*files))
        tokenized_lines = iterators.SamplingRandomMapIterator(tokenized_lines, self._prepare, self.seed)
        
        return tokenized_lines
    
    @staticmethod
    def fs_encode_line(fs_dict, words, append_eos=True):
        ids = []
        for i, word in enumerate(words):
            idx = fs_dict.index(word)
            ids.append(idx)
        if append_eos:
            ids.append(fs_dict.eos_index)
        return ids

    @staticmethod
    def _doc_jsonstr_to_ids(doc_jsonstr, spm_tokenizer=None, fs_dict=None):
        assert EOL_SYMBOL in fs_dict.indices
        eol_index = fs_dict.indices[EOL_SYMBOL]
        tokenized_ids = []
        for line in filter(None, json.loads(doc_jsonstr)["text"].split("\n")):
            if isinstance(spm_tokenizer, GPT2BPE):
                tokens = spm_tokenizer.encode(line).split(' ')
            elif isinstance(spm_tokenizer, Encoding):
                tokens = list(map(str, spm_tokenizer.encode(line + '\n', allowed_special="all")))
            else:
                tokens = spm_tokenizer.encode(line, out_type=str)
            tokenized_tokens = SpmLmLoader.fs_encode_line(fs_dict, tokens, append_eos=False)
            if not isinstance(spm_tokenizer, Encoding):
                tokenized_tokens.append(eol_index)
            tokenized_ids.append(tokenized_tokens)
        if len(tokenized_ids) > 0:
            last_line_ids = tokenized_ids[-1]
            if last_line_ids[-1] == eol_index:
                last_line_ids[-1] = fs_dict.eos_index
            else:
                last_line_ids.append(fs_dict.eos_index)
        else:
            print("[W] At SpmLmLoader._doc_jsonstr_to_ids, A null document with no lines!")
            tokenized_ids = [[fs_dict.eos_index]]
        return tokenized_ids

    def _read_from_files(self, source_file):
        data = []
        file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        # NOTE #### simple spm implementation ###############
        tokenized_ids = []
        for doc_jsonstr in lines:
            ret = SpmLmLoader._doc_jsonstr_to_ids(doc_jsonstr, spm_tokenizer=self.tokenizer, fs_dict=self.dictionary)
            tokenized_ids.extend(ret)
        # ###################################################

        doc = [self.dictionary.bos()]
        for ids in tokenized_ids:
            
            if getattr(self.args, "debug_p100", False):
                if len(ids) > 256:
                    ids = ids[:256]

            if len(ids) >= self.tokens_per_sample: # drop too long sentence
                # data.append(doc[:])
                ids = ids[:self.tokens_per_sample - 1]
                # continue

            if len(doc) + len(ids) > self.tokens_per_sample:
                if len(doc) > 5/2*self.mlm_cut_length + 1:
                    data.append(doc)
                doc = [self.dictionary.bos()]
            doc.extend(ids)

        if len(doc) > 1 and len(doc) <= self.tokens_per_sample:
            if len(doc) > 5/2*self.mlm_cut_length + 1:
                data.append(doc)

        return data