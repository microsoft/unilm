import torch

from fairseq.data import (data_utils,
  TokenBlockDataset, PrependTokenDataset, PadDataset, TruncateDataset,
  NumelDataset, NumSamplesDataset, NestedDictionaryDataset, 
  MaskTokensDataset, AppendTokenDataset, )
from fairseq.data.encoders.utils import get_whole_word_mask


def get_mlm_dataset(args, dataset_path, vocab, mask_idx, mask_whole_words=None, combine=False):
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
      # 'lang_id': RawLabelDataset([lang_id] * src_dataset.sizes.shape[0]),
    },
    sizes=[src_dataset.sizes],
  )
  return dataset

def add_mlm_args(parser):
  parser.add_argument('--mask-whole-words', default=False, action='store_true',
                      help='mask whole words; you may also want to set --bpe')
  parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
  parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                      help='probability that a masked token is unmasked')
  parser.add_argument('--random-token-prob', default=0.1, type=float,
                        help='probability of replacing a token with a random token')
  parser.add_argument('--sample-break-mode', default='complete',
                        choices=['none', 'complete', 'complete_doc', 'eos'],
                        help='If omitted or "none", fills each sample with tokens-per-sample '
                        'tokens. If set to "complete", splits samples only at the end '
                        'of sentence, but may include multiple sentences per sample. '
                        '"complete_doc" is similar but respects doc boundaries. '
                        'If set to "eos", includes only one sentence per sample.')


def get_preprocessed_ptb_dataset(args, dataset_path, vocab, combine=False):
  dataset = data_utils.load_indexed_dataset(
    dataset_path, vocab, args.dataset_impl, combine=combine, )
  if dataset is None:
    raise FileNotFoundError('Dataset not found: ({})'.format(dataset_path))
  return dataset


def get_prepended_token_block_dataset(args, dataset_path, vocab, combine=False):
  dataset = data_utils.load_indexed_dataset(
    dataset_path, vocab, args.dataset_impl, combine=combine, )

  if dataset is None:
    raise FileNotFoundError('Dataset not found: ({})'.format(dataset_path))

  if not args.apply_ptb: 
    print("| [I] ptb not applied.", flush=True)
    return dataset

  dataset = TruncateDataset(dataset, args.tokens_per_sample - 1)
  dataset = TokenBlockDataset(
    dataset,
    dataset.sizes,
    args.tokens_per_sample - 1,  # one less for <s>
    pad=vocab.pad(),
    eos=vocab.eos(),
    break_mode=args.sample_break_mode,
  )
  print('| loaded {} blocks from: {}'.format(len(dataset), dataset_path), flush=True)

  dataset = PrependTokenDataset(dataset, vocab.bos())
  return dataset
