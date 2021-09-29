"""Load examples from BUCC"""


import logging
import os
import torch


from transformers.data.processors.utils import (
  DataProcessor, InputExample, InputFeatures)
from torch.utils.data import (
  DataLoader, RandomSampler, SequentialSampler, TensorDataset)


logger = logging.getLogger(__name__)


def load_and_cache_examples(args, langpair, lang, tokenizer, key="", prefix="tatoeba"):

  cache_dir = os.path.join(args.data_dir, "pequod_cache")
  os.makedirs(cache_dir, exist_ok=True)
  cache_filename = os.path.join(
    cache_dir, "cached_%s_%s_%s" % (langpair, lang, key))
  
  if os.path.exists(cache_filename) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s" % cache_filename)
    features = torch.load(cache_filename)
  else:
    processer = TatoebaProcesser()
    logger.info("Creating features from dataset file at %s" % args.data_dir)
    examples = processer.get_examples(args.data_dir, langpair, lang, prefix)
    features = TatoebaProcesser.convert_examples_to_features(
      examples, tokenizer, args.max_seq_length, 0,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],)
    #logger.info("Saving features to cache file %s" % cache_filename)
    #torch.save(features, cache_filename)
  
  all_input_ids = torch.tensor(
    [f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor(
    [f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor(
    [f.token_type_ids for f in features], dtype=torch.long)

  dataset = TensorDataset(
    all_input_ids, all_attention_mask, all_token_type_ids)

  return dataset

class TatoebaProcesser(DataProcessor):

  @classmethod
  def convert_examples_to_features(cls, examples, tokenizer, max_length, pad_token_segment_id, pad_token, mask_padding_with_zero=True):

    features = []
    for ex_index, example in enumerate(examples):
      inputs = tokenizer.encode_plus(
        example.text_a,
        None,
        add_special_tokens=True,
        max_length=max_length,
      )
      input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

      attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

      padding_length = max_length - len(input_ids)
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
    
      assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
      assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
      assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

      if ex_index < 3:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      
      features.append(InputFeatures(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        label=None,
      ))

    return features

  def get_examples(self, data_dir, langpair, lang, prefix="tatoeba"):
    examples = []
    if prefix == "bucc":
      fn = os.path.join(data_dir, "%s.%s.txt" % (langpair, lang))
    else:
      fn = os.path.join(data_dir, "%s.%s" % (langpair, lang))
    #fn = os.path.join(data_dir, "%s.%s.%s" % (prefix, langpair, lang))
    with open(fn, encoding='utf-8') as fp:
      for i, line in enumerate(fp):
        line = line.strip()
        examples.append(InputExample(
          guid="%s-%s-%d" % (langpair, lang, i),
          text_a=line,
        ))
    return examples
