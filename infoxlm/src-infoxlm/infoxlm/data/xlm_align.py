import torch

from fairseq.data import (data_utils,
  TokenBlockDataset, PrependTokenDataset, PadDataset, TruncateDataset,
  NumelDataset, NumSamplesDataset, NestedDictionaryDataset, 
  MaskTokensDataset, AppendTokenDataset, )
from fairseq.data.encoders.utils import get_whole_word_mask

from infoxlm.data.mlm_utils import get_prepended_token_block_dataset
from infoxlm.data.offset_dataset import OffsetDataset


def get_xlm_align_dataset_with_mask(args, dataset_path, vocab, mask_idx, combine=False):
  ptb_dataset = get_prepended_token_block_dataset(
    args, dataset_path, vocab, combine=combine)
  src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
    ptb_dataset,
    vocab=vocab,
    pad_idx=vocab.pad(),
    mask_idx=mask_idx,
    seed=args.seed,
    mask_prob=args.mask_prob,
  )
  dataset = NestedDictionaryDataset({
    'net_input': {
      'src_tokens': PadDataset(
        ptb_dataset,
        pad_idx=vocab.pad(),
        left_pad=False,
      ),
      'src_lengths': NumelDataset(ptb_dataset, reduce=False),
    },
    'nsentences': NumSamplesDataset(),
    'ntokens': NumelDataset(ptb_dataset, reduce=True),
    'offsets': OffsetDataset(ptb_dataset, vocab),
    'net_input_tlm': {
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
  }, sizes=[ptb_dataset.sizes])
  return dataset