from __future__ import absolute_import
from __future__ import division

from numpy.core.fromnumeric import argsort
from text_encoder import SubwordTextEncoder
import tokenizer
import tempfile
import argparse
from transformers import BertTokenizer
import random
import math
import numpy as np
def merge_output_file_with_bert_vocab(output_filename, bert_vocab, temp_path):
  writer = open(output_filename, 'w', encoding='utf-8')
  _set = set()
  with open(bert_vocab, 'r', encoding='utf-8') as reader:
      for line in reader:
          writer.write(line)
          _set.add(line.strip())
  print(temp_path)
  with open(temp_path, 'r', encoding='utf-8') as reader:
      for line in reader:
          if line.strip() not in _set:
              writer.write(line)

  writer.close()

def build_target_size_vocab(token_counts, reserved_tokens, target_size):
    min_val = 1
    max_val = len(token_counts) // (target_size ** 0.5)
    encoder = SubwordTextEncoder.build_to_target_size(target_size,token_counts,min_val, max_val, num_iterations=5, 
                                                    reserved_tokens=reserved_tokens, max_subtoken_length=None)
    fd, temp_vocab = tempfile.mkstemp()
    encoder.store_to_file(temp_vocab, add_single_quotes=False)
    return encoder, temp_vocab

def compute_language_model(documents, vocab_file):
    all_tokens = 0
    tokenized_documents = []
    bert_tokenizer = BertTokenizer(vocab_file ,do_lower_case = True)
    words = bert_tokenizer.vocab
    for word in words.keys():
        words[word] = 0
    for doc in documents:
        tokens = bert_tokenizer.tokenize(doc)
        all_tokens += len(tokens)
        for token in tokens:
            words[token] +=1
        tokenized_documents.append(tokens)
    for word in words.keys():
        words[word] /= all_tokens
    probs = []
    for doc in tokenized_documents:
        p = 0.0 
        for token in doc:
            p += math.log(words[token])
        probs.append(p)

    return np.mean(probs)

def vocab_extend(corpus, raw_vocab, output_filename, interval=10000 , threshold = 0.01):
    """
    @description  : The function to get the incremental vocabulary for 
    
    @param  :
    
    @Returns  :
    
    """
    
    
    documents = []
    for line in open(corpus, "r",encoding='utf-8'):
        line = line.replace('\n','')
        if len(line) < 5:
            continue
        documents.append(line)
    print("docunments: "+str(len(documents)))
    token_counts = tokenizer.corpus_token_counts(
        corpus, corpus_max_lines = 4400000,
        split_on_newlines = True, additional_chars="", do_lower_case=True)
    lines = open(raw_vocab, 'r', encoding='utf-8').readlines()
    lines = [s.strip() for s in lines if len(s) > 0]
    reserved_tokens = lines
    random.shuffle(documents)
    origin_size = (len(reserved_tokens) // interval) * interval
    pre_lm = compute_language_model(documents, raw_vocab)
    print("origin_size: " + str(origin_size))
    print("pre_lm: "+ str(pre_lm))
    target_size = origin_size
    while True:
        target_size = target_size + interval
        _, temp_vocab = build_target_size_vocab(token_counts, reserved_tokens, target_size)
        now_lm = compute_language_model(documents, temp_vocab)
        print('now_lm: '+ str(now_lm))
        delta = (pre_lm - now_lm)/pre_lm
        print('delta: ' + str(delta))
        if delta <= threshold:
           merge_output_file_with_bert_vocab(output_filename, raw_vocab, temp_vocab)
           break
        pre_lm = now_lm


#vocab_extend('cs_data.txt', 'vocab.txt', 'cs.vocab')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, type=str, required=True,
                        help="the file of the corpus to train the vocabulary.")
    parser.add_argument("--raw_vocab", default=None, type=str, required=True,
                        help="the path to the file of the origin vocabulary")
    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="the output file of the final vocabulary")
    parser.add_argument('--interval', type=int, default=10000,
                        help="The interval of the vocabulary size.")
    parser.add_argument('--threshold', type=int, default=10000,
                        help="The final threhold of the P(D)'s increase")                                 
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    vocab_extend(args.corpus, args.raw_vocab, args.output_file, args.interval, args.threshold)

if __name__ == '__main__':
    main()
