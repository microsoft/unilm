import argparse
import os

import torch
from fairseq.data import (FairseqDataset, PrependTokenDataset,
                          TokenBlockDataset, TruncateDataset, data_utils, StripTokenDataset, ConcatDataset, PrependTokenDataset, AppendTokenDataset)
from fairseq.data.indexed_dataset import make_builder
from tqdm import tqdm
from transformers import AutoTokenizer

from infoxlm.data.tlm_dataset import TLMDataset


class IndexDataset(FairseqDataset):
  
  def __init__(self, indices):
    self.indices = indices
    self._sizes = [len(i) for i in indices]

  @property
  def sizes(self):
    return self._sizes
  
  def size(self, index):
    item = self.__getitem__(index)
    return len(item)
  
  def __getitem__(self, index):
    item = self.indices[index]
    item = torch.LongTensor(item)
    return item
  
  def __len__(self):
    return len(self.indices)
  
  def collater(self, samples):
    raise NotImplementedError


def build_tokenizer(args):
  tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  return tokenizer


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, default="CZWin32768/xlm-align")
  parser.add_argument("--input_src", type=str, default="")
  parser.add_argument("--input_trg", type=str, default="")
  parser.add_argument("--output", type=str, default="")
  parser.add_argument("--max_pos", type=int, default=256)
  args = parser.parse_args()
  return args


def save_items(items, prefix, vocab_size):
  bin_fn = "%s.bin" % prefix
  idx_fn = "%s.idx" % prefix
  builder = make_builder(bin_fn, "mmap", vocab_size=vocab_size)
  print("builder: " + str(builder))
  for item in items: builder.add_item(item)
  builder.finalize(idx_fn)


def get_indices(input_fn, tokenizer):
  indices = []
  with open(input_fn) as fp:
    for lid, line in tqdm(enumerate(fp)):
      # DEBUG 
      # if lid > 500: break
      line = line.strip()
      indices.append(tokenizer.encode(line))
  print("tokenize finished.")
  return indices



def main(args):
  tokenizer = build_tokenizer(args)
  src_indices = get_indices(args.input_src, tokenizer)
  trg_indices = get_indices(args.input_trg, tokenizer)

  src_dataset = IndexDataset(src_indices)
  trg_dataset = IndexDataset(trg_indices)

  eos = tokenizer.sep_token_id
  bos = tokenizer.cls_token_id
  max_pos = args.max_pos

  datasets = []

  src_dataset = TruncateDataset(
    StripTokenDataset(src_dataset, eos), max_pos - 2,)
  trg_dataset = TruncateDataset(
    StripTokenDataset(trg_dataset, eos), max_pos - 2,)

  src_dataset = PrependTokenDataset(src_dataset, bos)
  trg_dataset = PrependTokenDataset(trg_dataset, bos)

  src_dataset = AppendTokenDataset(src_dataset, eos)
  trg_dataset = AppendTokenDataset(trg_dataset, eos)

  print("| get all items ...")
  # items = [i for i in tqdm(dataset)]
  items = []
  for t1, t2 in tqdm(zip(src_dataset, trg_dataset)):
    items.append(t1)
    items.append(t2)

  print("| writing binary file ...")
  prefix = os.path.join(args.output, "train.0")
  save_items(items, prefix, len(tokenizer))


if __name__ == "__main__":
  args = get_args()
  main(args)
