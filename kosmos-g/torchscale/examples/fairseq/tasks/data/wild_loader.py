import json
import os
import multiprocessing
import itertools
import random
import re

from infinibatch import iterators
from functools import partial
from tasks.data.lm_loader import LMLoader
from tasks.data.utils import NativeCheckpointableIterator, WeightIterator, EOL_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, image_code_to_token
from fairseq.data.encoders.gpt2_bpe import GPT2BPE
from spacy.lang.en import English


IMAGE_KEY="Images"
TEXT_KEY="Extracted"


class WildLoader(LMLoader):
    def _setup(self):
        self.nlp_sentencizer = English()
        self.nlp_sentencizer.add_pipe("sentencizer")
        self.max_image_num = 5

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

    def text_transform(self, line): 
        spm_tokenizer=self.tokenizer
        if isinstance(spm_tokenizer, GPT2BPE):
            tokens = spm_tokenizer.encode(line).split(' ')
        else:
            tokens = spm_tokenizer.encode(line, out_type=str)
        tokenized_tokens = WildLoader.fs_encode_line(self.dictionary, tokens, append_eos=False)
        return tokenized_tokens
    
    def clean(self, text):
        # python re, remove html tags
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)


    def _read_from_files(self, source_file):
        """
        <s> <image> image token </image> sentence <image> image token </image> sentence </s>
        1, sample a random subsequnece:  3 sentences + the first image ... take up to 5 images + 3 sentences
        2, filter html tags <p>, <br>, <br/>
        3, single image, random sample rate as 0.5
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
                json_obj = json.loads(doc_jsonstr)
                doc = [self.dictionary.bos()]
                start_idx = 0
                image_num = len(json_obj[IMAGE_KEY])
                if image_num == 1:
                    r = random.random()
                    if r > 0.5:
                        continue
                for image_idx, image_item in enumerate(json_obj[IMAGE_KEY]):
                    if image_idx >= self.max_image_num:
                        if len(doc) < self.tokens_per_sample:
                            yield doc
                        break
                    
                    text_snippet = json_obj[TEXT_KEY][start_idx:image_item['Span'][0]-1]
                    text_snippet = self.clean(text_snippet)
                    if len(text_snippet) != 0:
                        if image_idx == 0:
                            # crop 3 sentences before the first image
                            sentences = list(self.nlp_sentencizer(text_snippet).sents)
                            text_snippet = ' '.join([str(sent) for sent in sentences[-3:]])
                        text_token = self.text_transform(text_snippet)
                        doc.extend(text_token)
                        if len(doc) >= self.tokens_per_sample: # drop too long sentence
                            # data.append(doc[:])
                            doc = doc[:self.tokens_per_sample - 2]
                            doc.append(self.dictionary.eos())
                            yield doc
                            break
                        
                    image_tokens = [image_code_to_token(i) for i in image_item['input_ids']]
                    image_tokens = WildLoader.fs_encode_line(self.dictionary, image_tokens, append_eos=False)
                    doc.append(self.dictionary.index(BOI_SYMBOL))
                    doc.extend(image_tokens)
                    doc.append(self.dictionary.index(EOI_SYMBOL))

                    start_idx = image_item['Span'][1] + 1
                    if image_idx == image_num - 1:
                        # crop 3 sentences after the last image
                        text_snippet = json_obj[TEXT_KEY][start_idx:]
                        text_snippet = self.clean(text_snippet)
                        sentences = list(self.nlp_sentencizer(text_snippet).sents)
                        text_snippet = ' '.join([str(sent) for sent in sentences[:3]])
                        text_token = self.text_transform(text_snippet)
                        doc.extend(text_token)
                        doc.append(self.dictionary.eos())
                        if len(doc) < self.tokens_per_sample:
                            yield doc
                            break
            except:
                continue