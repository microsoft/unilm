import torch

from fairseq.data import BaseWrapperDataset
from fairseq.data import (data_utils,
  TokenBlockDataset, PrependTokenDataset, PadDataset, TruncateDataset,
  NumelDataset, NumSamplesDataset, NestedDictionaryDataset, 
  MaskTokensDataset, AppendTokenDataset, )

from infoxlm.data.mlm_utils import get_mlm_dataset, get_prepended_token_block_dataset


def get_mlm_dataset_with_offset(args, dataset_path, vocab, mask_idx,mask_whole_words=None, combine=False):
  ptb_dataset = get_prepended_token_block_dataset(
    args, dataset_path, vocab, combine=combine)
  src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
    ptb_dataset,
    vocab=vocab,
    pad_idx=vocab.pad(),
    mask_idx=mask_idx,
    seed=args.seed,
    mask_prob=args.mask_prob,
    mask_whole_words=mask_whole_words,
  )
  dataset = NestedDictionaryDataset(
    {
      'net_input': {
        'src_tokens': PadDataset(
          src_dataset,
          pad_idx=vocab.pad(),
          left_pad=False,
        ),
        'src_lengths': NumelDataset(src_dataset, reduce=False),
      },
      'target': PadDataset(
        tgt_dataset,
        pad_idx=vocab.pad(),
        left_pad=False,
      ),
      'nsentences': NumSamplesDataset(),
      'ntokens': NumelDataset(src_dataset, reduce=True),
      'offsets': OffsetDataset(ptb_dataset, vocab),
    },
    sizes=[src_dataset.sizes],
  )
  return dataset


class OffsetDataset(BaseWrapperDataset):

  def __init__(self, ptb_dataset, vocab):
    super().__init__(ptb_dataset)
    self.vocab = vocab
  
  def get_check_ptb_offsets(self, ptb_item):
    # parse ptb_item
    eos_idx = self.vocab.eos()
    bos_idx = self.vocab.bos()
    _nonzero = (ptb_item == eos_idx).nonzero()
    if len(_nonzero) != 2:
      # raise ValueError
      # NOTE WALKAROUND
      _nonzero_0 = _nonzero[0].item()
      _nonzero_1 = len(ptb_item)
    else:
      _nonzero_0 = _nonzero[0].item()
      _nonzero_1 = _nonzero[1].item()
    
    assert ptb_item[0].item() == bos_idx, (ptb_item[0].item(), bos_idx)
    src_fr = 1
    src_to = _nonzero[0].item()
    trg_fr = src_to + 1
    trg_to = _nonzero[1].item()
    # print("ptb_item:")
    # print(ptb_item)
    # print("offsets:")
    # print("%d %d %d %d" % (src_fr, src_to, trg_fr, trg_to))
    # print("4 items: %d %d %d %d" % tuple(ptb_item[i].item() for i in [src_fr, src_to, trg_fr, trg_to]))

    if src_to - src_fr <= 0 or trg_to - trg_fr <= 0:
      print("[W] ptb_item=%s offsets=%d,%d,%d,%d" % (
        str(ptb_item), src_fr, src_to, trg_fr, trg_to,
      ))
      # raise ValueError

    return src_fr, src_to, trg_fr, trg_to
  
  def __getitem__(self, index):
    ptb_item = self.dataset[index]
    return self.get_check_ptb_offsets(ptb_item)

  def collater(self, samples):
    src_fr = [s[0] for s in samples]
    src_to = [s[1] for s in samples]
    trg_fr = [s[2] for s in samples]
    trg_to = [s[3] for s in samples]
    return src_fr, src_to, trg_fr, trg_to