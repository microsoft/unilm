import os
import logging
import torch

from torch.utils.data import TensorDataset
from src.pequod.data.utils_squad import (read_squad_examples,
  convert_examples_to_features)


logger = logging.getLogger(__name__)


def load_and_cache_examples(args, split, lang, tokenizer, key="", evaluate=False):
  cache_filename = os.path.join(
    args.data_dir, "cached_%s_%s_%s" % (split, lang, key))
  
  input_file = os.path.join(args.data_dir, "%s-%s.json" % (split, lang))
  if os.path.exists(cache_filename):
    logger.info("Loading features from cached file %s", cache_filename)
    features = torch.load(cache_filename)
    if evaluate:
      examples = read_squad_examples(input_file=input_file,
        is_training=not evaluate,
        version_2_with_negative=args.version_2_with_negative)
    else: examples = None
  else:
    logger.info("Creating features from dataset file at %s", input_file)
    examples = read_squad_examples(input_file=input_file,
      is_training=not evaluate,
      version_2_with_negative=args.version_2_with_negative)
    features = convert_examples_to_features(examples=examples,
      tokenizer=tokenizer, max_seq_length=args.max_seq_length,
      doc_stride=args.doc_stride, max_query_length=args.max_query_length,
      is_training=not evaluate, cls_token=tokenizer.cls_token,
      sep_token=tokenizer.sep_token)
    logger.info("Saving features into cached file %s", cache_filename)
    torch.save(features, cache_filename)
  
  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor(
    [f.input_ids for f in features], dtype=torch.long)
  all_input_mask = torch.tensor(
    [f.input_mask for f in features], dtype=torch.long)
  all_segment_ids = torch.tensor(
    [f.segment_ids for f in features], dtype=torch.long)
  all_cls_index = torch.tensor(
    [f.cls_index for f in features], dtype=torch.long)
  all_p_mask = torch.tensor(
    [f.p_mask for f in features], dtype=torch.float)
  if evaluate:
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
      all_example_index, all_cls_index, all_p_mask)
  else:
    all_start_positions = torch.tensor(
      [f.start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor(
      [f.end_position for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
      all_start_positions, all_end_positions, all_cls_index, all_p_mask)

  return dataset, examples, features
