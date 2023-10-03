import json
import os
import multiprocessing
import itertools
import random

from infinibatch import iterators
from functools import partial
from tasks.data.lm_loader import LMLoader
from tasks.data.utils import NativeCheckpointableIterator, WeightIterator, EOL_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, image_code_to_token
from fairseq.data.encoders.gpt2_bpe import GPT2BPE


class LaionLoader(LMLoader):
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

    def _read_from_files(self, source_file):
        """
        <s> <image> image token </image> sentence </s>
        <s> sentence  <image> image token </image> </s>

        """
        file_path = os.path.join(self.data_dir, source_file)
        
        if not os.path.exists(file_path):
            print('| file {} not exists'.format(file_path), flush=True)
            return iter([]) # skip bad file

        try:
            with open(file_path, 'r', encoding='utf8') as f:
                lines = f.read().strip().split('\n')
        except:
            return iter([]) # skip bad file

        for doc_jsonstr in lines:
            try:
                obj = json.loads(doc_jsonstr)
                if int(obj['width']) < 200 or int(obj['height']) < 200:
                    continue
                
                line = obj['caption']
                spm_tokenizer=self.tokenizer
                if isinstance(spm_tokenizer, GPT2BPE):
                    tokens = spm_tokenizer.encode(line).split(' ')
                else:
                    tokens = spm_tokenizer.encode(line, out_type=str)
                tokenized_tokens = LaionLoader.fs_encode_line(self.dictionary, tokens, append_eos=False)
                image_tokens = [image_code_to_token(i) for i in obj['input_ids']]
                image_tokens = LaionLoader.fs_encode_line(self.dictionary, image_tokens, append_eos=False)
                r = random.random()
                doc = [self.dictionary.bos()]
                if r < 0.5:
                    doc.append(self.dictionary.index(BOI_SYMBOL))
                    doc.extend(image_tokens)
                    doc.append(self.dictionary.index(EOI_SYMBOL))
                    doc.extend(tokenized_tokens)
                else:
                    doc.extend(tokenized_tokens)
                    doc.append(self.dictionary.index(BOI_SYMBOL))
                    doc.extend(image_tokens)
                    doc.append(self.dictionary.index(EOI_SYMBOL))
                doc.append(self.dictionary.eos())
                yield doc
            except:
                continue
