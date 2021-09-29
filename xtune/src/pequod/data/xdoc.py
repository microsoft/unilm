"""Loading examples and features for CLS and MLDoc"""

import logging
import os
import torch

from transformers.data.processors.utils import (DataProcessor,
  InputExample, InputFeatures)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

logger = logging.getLogger(__name__)


def get_processor_class(dataset_name):
  if dataset_name == "MLDoc": return MLDocProcessor
  elif dataset_name == "CLS": return CLSProcessor
  elif dataset_name == "XNLI": return XNLIProcesser
  elif dataset_name == "TriXNLI": return TriXNLIProcesser
  else: raise ValueError


def xdoc_convert_examples_to_features(
  processor, examples, tokenizer, max_length, label_list,
  pad_token=0, pad_token_segment_id=0, mask_padding_with_zero=True):
  
  if label_list is None: label_list = processor.get_labels()

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for ex_index, example in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % ex_index)
    inputs = tokenizer.encode_plus(
      example.text_a,
      example.text_b,
      add_special_tokens=True,
      max_length=max_length)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
  
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

    label = label_map[example.label]
    if ex_index < 3:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label))
    
    features.append(InputFeatures(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      label=label))
    
  return features


def load_and_cache_examples(args, processor, split, lang, tokenizer, key=""):
  cache_filename = os.path.join(
    args.data_dir, "cached_%s_%s_%s" % (split, lang, key))

  if os.path.exists(cache_filename) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s" % cache_filename)
    features = torch.load(cache_filename)
  else:
    logger.info("Creating features from dataset file at %s" % args.data_dir)
    label_list = processor.get_labels()
    examples = processor.get_examples(args.data_dir, split, lang)
    logger.info("%d Examples loaded" % len(examples))
    features = xdoc_convert_examples_to_features(
      processor, examples, tokenizer, max_length=args.max_seq_length,
      label_list=label_list, pad_token_segment_id=0,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0])
    logger.info("Saving features to cache file %s" % cache_filename)
    torch.save(features, cache_filename)
  
  all_input_ids = torch.tensor(
    [f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor(
    [f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor(
    [f.token_type_ids for f in features], dtype=torch.long)
  all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

  dataset = TensorDataset(
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

  return dataset


class XDocProcessor(DataProcessor):
  """Processor for the MLDoc dataset."""

  def get_example_from_tensor_dict(self, tensor_dict):
    return InputExample(
      tensor_dict["idx"].numpy(),
      tensor_dict["sentence"].numpy().decode("utf-8"),
      str(tensor_dict["label"].numpy()))

  def get_examples(self, data_dir, split, lang):
    filename = "%s-%s.tsv" % (split, lang)
    logger.info("LOOKING AT %s" % os.path.join(data_dir, filename))
    return self._create_examples(
      self._read_tsv(os.path.join(data_dir, filename)), filename)

  def _create_examples(self, lines, set_type):
    examples = []
    for i, line in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      try:
        label, text_a = line[0], line[1]
      except IndexError:
        logger.warn("IndexError while decomposing line %s" % str(line))
        logger.warn("Line ignored... Loop continued...")
        continue
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class MLDocProcessor(XDocProcessor):
  def get_labels(self): return ["ECAT", "CCAT", "GCAT", "MCAT"]


class CLSProcessor(XDocProcessor):
  def get_labels(self): return ["0", "1"]


class XNLIProcesser(XDocProcessor):
  """data format: a pair: (label, text)"""
  def get_labels(self): return ["neutral", "entailment", "contradiction"]


class TriXNLIProcesser(XNLIProcesser):
  """data format: a 3-tuple: (label, text-a, text-b)"""
  def _create_examples(self, lines, set_type):
    examples = []
    for i, line in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      label, text_a, text_b = line[0], line[1], line[2]
      examples.append(
        InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples