import numpy as np
import torch

from fairseq.data import data_utils, FairseqDataset, MaskTokensDataset, TruncateDataset, BaseWrapperDataset
from infoxlm.data.dict_dataset import DictDataset


def get_xlco_dataset(args, dataset_path, vocab, mask_idx, combine=False):
  dataset = data_utils.load_indexed_dataset(
    dataset_path, vocab, args.dataset_impl, combine=combine)
  
  dataset, _ = MaskTokensDataset.apply_mask(
    dataset,
    vocab=vocab,
    pad_idx=vocab.pad(),
    mask_idx=mask_idx,
    seed=args.seed,
    mask_prob=args.mask_prob,
    mask_whole_words=None,
  )
  dataset = XlcoDataset(dataset, vocab)
  return dataset


class XlcoDataset(FairseqDataset):
  
  def __init__(self, dataset, vocab, remove_bos_of_item2=True, seed=1):
    # dataset: pair -> (line i, line i + 1) where i % 2 == 0
    self.dataset = dataset
    self.vocab = vocab
    self.remove_bos_of_item2 = remove_bos_of_item2
    self.seed = seed
    self.epoch = 0
  
  def set_epoch(self, epoch):
    self.epoch = epoch
    if hasattr(self.dataset, 'set_epoch'):
        self.dataset.set_epoch(epoch)
  
  def __len__(self):
    return len(self.dataset) // 4
  
  # NOTE mix-up contrast
  def __getitem__(self, index):
    src_item1 = self.dataset[index*4]
    tgt_item1 = self.dataset[index*4+1]
    src_item2 = self.dataset[index*4+2]
    tgt_item2 = self.dataset[index*4+3]

    with data_utils.numpy_seed(self.seed, self.epoch, index):
      mode = np.random.randint(8)
    if mode & 1: src_item1, src_item2 = src_item2, src_item1
    if mode & 2: tgt_item1, tgt_item2 = tgt_item2, tgt_item1

    bos = self.vocab.bos()
    if self.remove_bos_of_item2 and src_item2[0] == bos:
      src_item2 = src_item2[1:]
    if self.remove_bos_of_item2 and tgt_item2[0] == bos:
      tgt_item2 = tgt_item2[1:]
    
    src_item = torch.cat([src_item1, src_item2])
    tgt_item = torch.cat([tgt_item1, tgt_item2])
    if mode & 4: src_item, tgt_item = tgt_item, src_item

    return {
      'id': index,
      'source': src_item,
      'target': tgt_item,
    }
  
  def collater(self, samples):
    if len(samples) == 0:
      return {}

    pad_idx = self.vocab.pad()
    eos_idx = self.vocab.eos()

    def merge(key, left_pad, move_eos_to_beginning=False):
      return data_utils.collate_tokens(
        [s[key] for s in samples],
        pad_idx, eos_idx, left_pad, move_eos_to_beginning,
      )

    id = torch.LongTensor([s['id'] for s in samples])

    src_tokens = merge('source', left_pad=False)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    tgt_tokens = merge('target', left_pad=False)
    tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples])

    n_src_tokens = sum(len(s['source']) for s in samples)
    n_tgt_tokens = sum(len(s['target']) for s in samples)

    batch = {
      'id': id,
      'nsentences': len(samples),
      'ntokens': n_src_tokens + n_tgt_tokens,
      'src_net_input': {
        'src_tokens': src_tokens,
        'src_lengths': src_lengths,
      },
      # NOTE the Roberta forward function takes src_tokens as input
      'tgt_net_input': {
        'src_tokens': tgt_tokens,
        'src_lengths': tgt_lengths,
      },
    }

    return batch