import os
from functools import lru_cache

import numpy as np
import torch

from fairseq import utils
from fairseq.data.data_utils import process_bpe_symbol
from fairseq.data.dictionary import Dictionary
from fairseq.tasks import FairseqTask, register_task

from infoxlm.data import mlm_utils
from infoxlm.data.dict_dataset import DictDataset
from infoxlm.data.xlm_align import get_xlm_align_dataset_with_mask


def extract_wa_from_pi_xi(pi, xi):
  m, n = pi.size()
  forward = torch.eye(n)[pi.argmax(dim=1)]
  backward = torch.eye(m)[xi.argmax(dim=0)]
  inter = forward * backward.transpose(0, 1)
  ret = []
  for i in range(m):
    for j in range(n):
      if inter[i, j].item() > 0:
        ret.append((i, j))
  return ret


def _sinkhorn_iter(S, num_iter=2):
  assert S.dim() == 2
  S[S <= 0] = 1e-6
  pi = S
  xi = pi
  for i in range(num_iter):
    pi_sum_over_i = pi.sum(dim=0, keepdim=True)
    xi = pi / pi_sum_over_i
    xi_sum_over_j = xi.sum(dim=1, keepdim=True)
    pi = xi / xi_sum_over_j
  return pi, xi


@register_task('xlm_align')
class XlmAlignTask(FairseqTask):

  @staticmethod
  def add_args(parser):
    # MLM args
    mlm_utils.add_mlm_args(parser)
    parser.add_argument('data', help='colon separated path to data directories list, '
                        'will be iterated upon during epochs in round-robin manner')
    parser.add_argument('--tokens-per-sample', default=512, type=int,
                        help='max number of total tokens over all segments per sample')
    # apply prepend bos + tokenblock
    parser.add_argument('--apply_ptb', default=False, action='store_true')

    # TLM args
    parser.add_argument('--tlm_data', type=str, default="")

    # Word Alignment Self-Labeling
    parser.add_argument('--wa_layer', type=int, default=8, help="the layer to obtain word alignment")
    parser.add_argument('--wa_max_count', type=int, default=2, help="max_count for itermax")
    parser.add_argument('--align_enable_step', default=-1, type=int)
    parser.add_argument('--feed_inner_states', default=False, action='store_true')
    parser.add_argument('--sinkhorn_iter', type=int, default=2, help="num of sinkhorn iterations")
  
  @classmethod
  def setup_task(cls, args, **kwargs):
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    print('| Dictionary: {} types'.format(len(dictionary)), flush=True)
    return cls(args, dictionary)
  
  def __init__(self, args, dictionary):
    super().__init__(args)
    self.dictionary = dictionary
    self.mask_idx = self.dictionary.add_symbol('<mask>')
    self.seed = args.seed
    self.mww = self._get_whole_word_mask()

    self.sa_model = None
    self._enable_align = False
  
  def prepare_train(self, model, criterion):
    print("| Prepare train ...", flush=True)
    self.model = model
    model.train()
  
  def _get_whole_word_mask(self):
    # create masked input and targets
    if self.args.mask_whole_words:
      print("| Get whole word mask ...")
      return mlm_utils.get_whole_word_mask(self.args, self.dictionary)
    return None
  
  @property
  def source_dictionary(self):
    return self.dictionary

  @property
  def target_dictionary(self):
    return self.dictionary
  
  def load_dataset(self, split, epoch=0, combine=False, **kwargs):
    print("| Loading dataset at epoch %d" % epoch, flush=True)
    args = self.args
    sid = 0

    dataset_path = os.path.join(args.data, "train.%d" % sid)
    mlm_dataset = mlm_utils.get_mlm_dataset(
      args, dataset_path, self.dictionary, self.mask_idx, self.mww, combine=False)

    dataset_path = os.path.join(args.tlm_data, "train.%d" % sid)
    sa_dataset = get_xlm_align_dataset_with_mask(args, dataset_path, self.dictionary, self.mask_idx, combine=False)

    dataset = DictDataset({
      "mlm": mlm_dataset,
      "sa": sa_dataset
    })

    # NOTE Set dataset epoch as sid for different random state 
    # of each shard, because when local indices are the same, the
    # random states are the same.
    dataset.set_epoch(sid)
    self.datasets[split] = dataset
  
  def iter_max(self, sim_matrix):
    sim_matrix = sim_matrix.cpu().detach().numpy()
    max_count = self.args.wa_max_count
    alpha_ratio = 0.9
    m, n = sim_matrix.shape
    forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
    backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
    inter = forward * backward.transpose()

    # if min(m, n) <= 2:
    #   return inter

    if  min(m, n) > 2:
      new_inter = np.zeros((m, n))
      count = 1
      while count < max_count:
        mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
        mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
        mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
        mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
        if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
          mask *= 0.0
          mask_zeros *= 0.0

        new_sim = sim_matrix * mask
        fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
        bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
        new_inter = fwd * bac

        if np.array_equal(inter + new_inter, inter):
          break
        inter = inter + new_inter
        count += 1
    
    ret = []
    for i in range(m):
      for j in range(n):
        if inter[i, j] > 0:
          ret.append((i, j))
    return inter, ret
  
  def get_gold_or_silver_wa(self, sample, batch_sim, src_fr, src_to, trg_fr, trg_to):
    gold_wa = []
    for i, sim in enumerate(batch_sim):
      sim_wo_offset = sim[src_fr[i]: src_to[i], trg_fr[i]: trg_to[i]]
      if src_to[i] - src_fr[i] <= 0 or trg_to[i] - trg_fr[i] <= 0:
        print("[W] src or trg len=0")
        gold_wa.append([])
        continue
      pi, xi = _sinkhorn_iter(sim_wo_offset, self.args.sinkhorn_iter)
      gold_wa_i_wo_offset = self._extract_wa_from_pi_xi(pi, xi)
      gold_wa_i = []
      for src_idx, trg_idx in gold_wa_i_wo_offset:
        gold_wa_i.append((src_idx + src_fr[i], trg_idx + trg_fr[i]))
      gold_wa.append(gold_wa_i)
    
    return gold_wa
  
  def get_aligned_tokens(self, sample, model, use_csls=False, return_inner_states=False):
    _, inner_states = model(**sample['net_input'], 
      features_only=True, return_all_hiddens=True)
    # rep: batch, hidden, length
    rep = inner_states["inner_states"][self.args.wa_layer]

    src_fr, src_to, trg_fr, trg_to = sample["offsets"]
    # rep: batch, length, hidden
    rep = rep.transpose(0, 1)

    if use_csls: raise NotImplementedError
    batch_sim = torch.bmm(rep, rep.transpose(1,2))
    wa = self.get_gold_or_silver_wa(sample, batch_sim, src_fr, src_to, trg_fr, trg_to)
    if return_inner_states: return wa, inner_states
    else: return wa

  def _extract_wa_from_pi_xi(self, pi, xi):
    # return extract_wa_from_pi_xi(pi, xi)
    _, wa = self.iter_max(pi)
    return wa
  
  def _set_enable_align(self, num_updates):
    if num_updates < self.args.align_enable_step: self._enable_align = False
    else: self._enable_align = True
  
  def update_step(self, num_updates):
    self._set_enable_align(num_updates)
  
  def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):

    if self.sa_model is None:
      self.sa_model = model
    
    agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
    
    if self._enable_align:
      self.sa_model.eval()
      if self.args.feed_inner_states:
        with torch.no_grad():
          aligned_tokens, inner_states = self.get_aligned_tokens(sample["sa"], self.sa_model, return_inner_states=True)
        model.train()
        loss, sample_size, logging_output = criterion(
          model, sample["sa"], reduce=True, aligned_tokens=aligned_tokens, inner_states=inner_states)
      else:
        with torch.no_grad():
          aligned_tokens = self.get_aligned_tokens(sample["sa"], self.sa_model)
        model.train()
        loss, sample_size, logging_output = criterion(
          model, sample["sa"], reduce=True, aligned_tokens=aligned_tokens)
      if ignore_grad: loss *= 0
      optimizer.backward(loss)
    else:
      model.train()
      loss, sample_size, logging_output = criterion(model, sample["sa"], tlm=True)
      if ignore_grad: loss *= 0
      optimizer.backward(loss)
        

    agg_loss += loss.detach().item()
    agg_sample_size += sample_size
    agg_logging_output.update(logging_output)

    loss, sample_size, logging_output = criterion(model, sample["mlm"], mlm=True)
    if ignore_grad: loss *= 0
    optimizer.backward(loss)

    agg_loss += loss.detach().item()
    agg_sample_size += sample_size
    agg_logging_output.update(logging_output)

    return agg_loss, agg_sample_size, agg_logging_output

