import argparse
import os

import torch
from fairseq.data import (FairseqDataset, PrependTokenDataset,
                          TokenBlockDataset, TruncateDataset, data_utils)
from fairseq.data.indexed_dataset import make_builder
from tqdm import tqdm
from transformers import AutoTokenizer


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
  parser.add_argument("--input", type=str, default="")
  parser.add_argument("--output", type=str, default="")
  parser.add_argument('--sample-break-mode', default='complete',
                        choices=['none', 'complete', 'complete_doc', 'eos'],
                        help='If omitted or "none", fills each sample with tokens-per-sample '
                        'tokens. If set to "complete", splits samples only at the end '
                        'of sentence, but may include multiple sentences per sample. '
                        '"complete_doc" is similar but respects doc boundaries. '
                        'If set to "eos", includes only one sentence per sample.')
  parser.add_argument('--tokens-per-sample', default=510, type=int,
                      help='max number of total tokens over all segments per sample')
  parser.add_argument('--dataset_impl', default="mmap", type=str)
  args = parser.parse_args()
  return args


def save_items(items, prefix, vocab_size):
  bin_fn = "%s.bin" % prefix
  idx_fn = "%s.idx" % prefix
  builder = make_builder(bin_fn, "mmap", vocab_size=vocab_size)
  print("builder: " + str(builder))
  for item in items: builder.add_item(item)
  builder.finalize(idx_fn)


def main(args):
  tokenizer = build_tokenizer(args)

  indices = []
  with open(args.input) as fp:
    for line in tqdm(fp):
      line = line.strip()
      indices.append(tokenizer.encode(line))
  print("tokenize finished.")
  for i in range(5):
    print("example[%d]:" % i)
    input_ids = indices[i]
    print(input_ids)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    print(tokens)

  dataset = IndexDataset(indices)
  dataset = TruncateDataset(dataset, args.tokens_per_sample - 1)
  dataset = TokenBlockDataset(
    dataset,
    dataset.sizes,
    args.tokens_per_sample - 1,  # one less for <s>
    pad=tokenizer.pad_token_id,
    eos=tokenizer.sep_token_id,
    break_mode=args.sample_break_mode,
  )
  print('| loaded {} blocks from: {}'.format(len(dataset), args.input), flush=True)

  dataset = PrependTokenDataset(dataset, tokenizer.cls_token_id)
  print("| get all items ...")
  items = [i for i in tqdm(dataset)]
  print("| writing binary file ...")
  prefix = os.path.join(args.output, "train.0")
  save_items(items, prefix, len(tokenizer))



if __name__ == "__main__":
  args = get_args()
  main(args)
