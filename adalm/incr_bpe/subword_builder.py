#-*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from text_encoder import SubwordTextEncoder
import tokenizer
import os
import tempfile

import tensorflow as tf

tf.flags.DEFINE_string('output_filename', '/tmp/my.subword_text_encoder',
                       'where to store the SubwordTextEncoder')
tf.flags.DEFINE_string('corpus_filepattern', '',
                       'Corpus of one or more text files')
tf.flags.DEFINE_string('vocab_filepattern', '', 'One or more vocabulary files '
                       '(one word per line as "word,count")')
tf.flags.DEFINE_integer('min_count', 5, 'Minimum subtoken count in corpus')
tf.flags.DEFINE_integer('vocab_size', 30000, 'The final vocab size. It will produce a vocab with a near vocab size')
tf.flags.DEFINE_integer('corpus_max_lines', None,
                        'How many lines of corpus to read')
tf.flags.DEFINE_integer('num_iterations', 5, 'Number of iterations')
tf.flags.DEFINE_bool('split_on_newlines', True, 'Break corpus into lines.')
tf.flags.DEFINE_string('additional_chars', "", 'Set special characters to be included in vocab. ex : "~", "/".')
tf.flags.DEFINE_integer('max_subtoken_length', None, 'Max subtoken length')
tf.flags.DEFINE_string('raw_vocab', None, 'Raw bert vovab file')
tf.flags.DEFINE_bool('do_lower_case', False, 'Whether or not to lowercase the input corpus')
FLAGS = tf.flags.FLAGS


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
  # os.remove(temp_path)

def main(unused_argv):
  if FLAGS.corpus_filepattern and FLAGS.vocab_filepattern:
    raise ValueError(
        'Must only provide one of --corpus_filepattern or --vocab_filepattern')

  elif FLAGS.corpus_filepattern:
    token_counts = tokenizer.corpus_token_counts(
        FLAGS.corpus_filepattern,
        FLAGS.corpus_max_lines,
        split_on_newlines=FLAGS.split_on_newlines, additional_chars=FLAGS.additional_chars, do_lower_case=FLAGS.do_lower_case)

  elif FLAGS.vocab_filepattern:
    token_counts = tokenizer.vocab_token_counts(FLAGS.vocab_filepattern,
                                                FLAGS.corpus_max_lines, FLAGS.do_lower_case)

  else:
    raise ValueError(
        'Must provide one of --corpus_filepattern or --vocab_filepattern')
  reserved_tokens = None
  if FLAGS.raw_vocab:
      lines = open(FLAGS.raw_vocab, 'r', encoding='utf-8').readlines()
      lines = [s.strip() for s in lines if len(s) > 0]
      reserved_tokens = lines

  print(len(token_counts))
  print(len(reserved_tokens))
  target_size = FLAGS.vocab_size
  if target_size <= len(reserved_tokens):
    raise ValueError("The vocab_size must be larger than the origin vocab's size ")
  if target_size >= len(token_counts):
    raise ValueError("The vocab_size is too large. Please set it smaller or prepare more corpus.")
  min_val = 1
  max_val = len(token_counts) // (target_size ** 0.5)  
  fd, temp_path = tempfile.mkstemp()
  encoder = SubwordTextEncoder.build_to_target_size(target_size,token_counts,min_val, max_val, num_iterations=FLAGS.num_iterations, 
                                                    reserved_tokens=reserved_tokens, max_subtoken_length=FLAGS.max_subtoken_length)
  # encoder = SubwordTextEncoder()
  # encoder.build_from_token_counts(token_counts, FLAGS.min_count,
  #                                 FLAGS.num_iterations, reserved_tokens=reserved_tokens, max_subtoken_length=FLAGS.max_subtoken_length)
  encoder.store_to_file(temp_path, add_single_quotes=False)
  merge_output_file_with_bert_vocab(FLAGS.output_filename, FLAGS.raw_vocab, temp_path)

if __name__ == '__main__':
  tf.app.run()
