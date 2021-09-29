"""Loading examples and features for WiLI-2018 dataset"""

import logging
import os
import torch

from transformers.data.processors.utils import (DataProcessor,
  InputExample, InputFeatures)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from src.data import convert_examples_to_features
from src.io import lines_gen


logger = logging.getLogger(__name__)


_alias2lang = {}
_lang2id = {}
_langs = []

def get_alias2lang(data_dir):
  if len(_alias2lang) > 0: return _alias2lang, _lang2id, _langs
  for line, in lines_gen(os.path.join(data_dir, "labels-new")):
    value = None
    for alias in line.split(";"):
      alias = alias.strip()
      if alias == "": continue
      if value is None: value = alias
      _alias2lang[alias] = value
    _langs.append(value)
  for i, lang in enumerate(_langs): _lang2id[lang] = i
  return _alias2lang, _lang2id, _langs


def load_and_cache_examples(args, data_dir, split, run_lang2id, tokenizer, key=""):
  cache_filename = os.path.join(
    data_dir, "cached_%s_%s" % (split, key))
  
  if os.path.exists(cache_filename) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s" % cache_filename)
    features = torch.load(cache_filename)
  else:
    processor = WiliProcessor()
    logger.info("Creating features from dataset file at %s" % data_dir)
    label_list = processor.get_labels(data_dir)
    examples = processor.get_examples(data_dir, split)
    logger.info("%d Examples loaded" % len(examples))
    features = convert_examples_to_features(
      processor, examples, tokenizer, max_length=args.max_seq_length,
      label_list=label_list, pad_token_segment_id=0,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
    logger.info("Saving features to cache file %s" % cache_filename)
    torch.save(features, cache_filename)
  
  # Cut dataset to test langs
  alias2lang, lang2id, _ = get_alias2lang(data_dir)
  test_lang_ids = {lang2id[alias2lang[lang]] for lang in run_lang2id.keys()}
  wili_id2run_langid = {
    lang2id[alias2lang[lang]]:val for lang, val in run_lang2id.items()}
  
  all_input_ids, all_attention_mask = [], [] 
  all_token_type_ids, all_labels = [], []
  for f in features:
    if f.label not in test_lang_ids: continue
    all_input_ids.append(f.input_ids)
    all_attention_mask.append(f.attention_mask)
    all_token_type_ids.append(f.token_type_ids)
    all_labels.append(wili_id2run_langid[f.label])
  
  all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
  all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
  all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
  all_labels = torch.tensor(all_labels,  dtype=torch.long)
  
  dataset = TensorDataset(
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

  return dataset


class WiliProcessor(DataProcessor):

  def get_examples(self, data_dir, split):
    examples = []
    filename_x = os.path.join(data_dir, "x_%s.txt" % split)
    filename_y = os.path.join(data_dir, "y_%s.txt" % split)
    for i, (line_x, line_y) in enumerate(lines_gen(filename_x, filename_y)):
      guid = "%s-%s" % (split, i)
      examples.append(
        InputExample(guid=guid, text_a=line_x, text_b=None, label=line_y))
    return examples
  
  def get_labels(self, data_dir):
    _, _, langs = get_alias2lang(data_dir)
    return langs
