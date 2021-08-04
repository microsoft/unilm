import collections
import logging
import math
import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch import distributed

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import process_bpe_symbol

from infoxlm.utils import _get_logging_loss, construct_idx_tensor_from_list


@register_criterion('dwa_mlm_tlm')
class DwaMlmTlm(FairseqCriterion):

  IGNORE_INDEX = 1000000

  def __init__(self, args, task):
    super().__init__(args, task)
    self.padding_idx = self.task.dictionary.pad_index
  
  @staticmethod
  def add_args(parser):
    parser.add_argument('--no_tlm_loss', default=False, action='store_true')
  
  def forward_mlm(self, model, sample, reduce=True, dep_rep_size=3):
    masked_tokens = sample['target'].ne(self.padding_idx)
    sample_size = masked_tokens.int().sum().item()

    # (Rare case) When all tokens are masked, the model results in empty
    # tensor and gives CUDA error.
    if sample_size == 0:
      masked_tokens = None
    # logger.warning(str(sample["net_input"]["src_tokens"]))
    # logger.warning("index - " + str(sample["net_input"]["src_tokens"].max()))
    # logger.warning("len - " + str(sample["net_input"]["src_lengths"].max()))
    features, _ = model(**sample['net_input'], use_model_fast=True, features_only=True)
    logits = model.output_layer(features, masked_tokens=masked_tokens, use_model_fast=True)
    targets = model.get_targets(sample, [logits])

    if sample_size != 0:
      targets = targets[masked_tokens]

    # loss could be FloatTensor caused by deprecated functional method
    loss = F.nll_loss(
      F.log_softmax(
        logits.view(-1, logits.size(-1)),
        dim=-1,
        dtype=torch.float32,
      ),
      targets.view(-1),
      reduction='sum',
      ignore_index=self.padding_idx,
    ).half()
    logging_loss = utils.item(loss.data) if reduce else loss.data
    logging_output = {
      'mlm_loss': logging_loss,
      'mlm_ntokens': sample['ntokens'],
      'mlm_nsentences': sample['nsentences'],
      'mlm_sample_size': sample_size,
    }


    # NOTE WALKAROUND We have to use all parameters for ddp.
    hidden_sz = features.size(-1)
    if hasattr(model, "qa_layer"):
      dep_rep = features.new(hidden_sz * dep_rep_size).fill_(0)
      dep_rep = model.qa_layer(dep_rep)
      loss += dep_rep.mean() * 0.0
    if hasattr(model, "q_linear"):
      dep_rep = features.new(hidden_sz).fill_(0)
      dep_rep1 = model.q_linear(dep_rep).mean()
      dep_rep2 = model.k_linear(dep_rep).mean()
      loss += dep_rep1 * 0.0 + dep_rep2 * 0.0
    if hasattr(model, "predictor"):
      dep_rep = features.new(hidden_sz).fill_(0)
      dep_rep = model.predictor(dep_rep)
      loss += dep_rep.mean() * 0.0

    return loss, sample_size, logging_output
  
  def forward_tlm(self, model, sample, reduce=True, dep_rep_size=3, net_input_key="net_input_tlm"):
    masked_tokens = sample['target'].ne(self.padding_idx)
    sample_size = masked_tokens.int().sum().item()

    # (Rare case) When all tokens are masked, the model results in empty
    # tensor and gives CUDA error.
    if sample_size == 0:
      masked_tokens = None
    # logger.warning(str(sample["net_input"]["src_tokens"]))
    # logger.warning("index - " + str(sample["net_input"]["src_tokens"].max()))
    # logger.warning("len - " + str(sample["net_input"]["src_lengths"].max()))
    features, _ = model(**sample[net_input_key], use_model_fast=True, features_only=True)
    logits = model.output_layer(features, masked_tokens=masked_tokens, use_model_fast=True)
    targets = model.get_targets(sample, [logits])

    if sample_size != 0:
      targets = targets[masked_tokens]

    # loss could be FloatTensor caused by deprecated functional method
    loss = F.nll_loss(
      F.log_softmax(
        logits.view(-1, logits.size(-1)),
        dim=-1,
        dtype=torch.float32,
      ),
      targets.view(-1),
      reduction='sum',
      ignore_index=self.padding_idx,
    ).half()
    logging_loss = utils.item(loss.data) if reduce else loss.data
    logging_output = {
      'tlm_loss': logging_loss,
      'tlm_ntokens': sample['ntokens'],
      'tlm_nsentences': sample['nsentences'],
      'tlm_sample_size': sample_size,
    }


    # NOTE WALKAROUND We have to use all parameters for ddp.
    hidden_sz = features.size(-1)
    if hasattr(model, "qa_layer"):
      dep_rep = features.new(hidden_sz * dep_rep_size).fill_(0)
      dep_rep = model.qa_layer(dep_rep)
      loss += dep_rep.mean() * 0.0
    if hasattr(model, "q_linear"):
      dep_rep = features.new(hidden_sz).fill_(0)
      dep_rep1 = model.q_linear(dep_rep).mean()
      dep_rep2 = model.k_linear(dep_rep).mean()
      loss += dep_rep1 * 0.0 + dep_rep2 * 0.0
    if hasattr(model, "predictor"):
      dep_rep = features.new(hidden_sz).fill_(0)
      dep_rep = model.predictor(dep_rep)
      loss += dep_rep.mean() * 0.0

    return loss, sample_size, logging_output
  
  def forward(self, model, sample, reduce=True, aligned_tokens=None, mlm=False, tlm=False):
    if mlm:
      return self.forward_mlm(model, sample, reduce, dep_rep_size=2)
    elif tlm:
      return self.forward_tlm(model, sample, reduce, dep_rep_size=2, net_input_key="net_input_tlm")
    else:
      return self.forward_denoise_word_alignment(model, sample, reduce, aligned_tokens, use_tlm_loss=(not self.args.no_tlm_loss))
  
  
  def forward_masked_lm(self, features, tlm_targets, model):
    masked_tokens = tlm_targets.ne(self.padding_idx)
    sample_size = masked_tokens.int().sum().item()

    if sample_size == 0: masked_tokens = None
    logits = model.output_layer(features, masked_tokens=masked_tokens)
    targets = tlm_targets
    if sample_size != 0: targets = targets[masked_tokens]

    loss = F.nll_loss(
      F.log_softmax(
        logits.view(-1, logits.size(-1)),
        dim=-1,
        dtype=torch.float32,
      ),
      targets.view(-1),
      reduction='sum',
      ignore_index=self.padding_idx,
    ).half()

    logging_output = {
      'tlm_loss': _get_logging_loss(loss),
      'tlm_sample_size': sample_size,
    }

    return loss, sample_size, logging_output
  
  def _positions2masked_features(self, positions, features, hidden_sz):
    # bsz, max_num_spans
    # NOTE paddings are filled with -1, but we need to replace -1 to 0 to gather
    positions4gather = positions.clone().detach()
    positions4gather[positions==DwaMlmTlm.IGNORE_INDEX] = 0
    # bsz, max_num_spans -> bsz, max_num_spans, hidden
    positions4gather = positions4gather.unsqueeze(-1).expand(-1, -1, hidden_sz)
    masked_features = features.gather(dim=1, index=positions4gather)
    return masked_features
  
  
  @staticmethod
  def aggregate_logging_outputs(logging_outputs):
    """Aggregate logging outputs from data parallel training."""
    # loss_sum = sum(log.get('loss', 0) for log in logging_outputs)

    reduced_log = collections.defaultdict(float)
    # TODO sa EM & F1
    reduced_keys = ["sa_loss", 'sa_EM', 'sa_EM_tot', 'sa_nsentences', 'sa_ntokens', 'sa_sample_size', "tlm_loss", "tlm_sample_size", "mlm_ntokens", "mlm_nsentences", "mlm_sample_size", "mlm_loss"]

    for log in logging_outputs:
      for key in reduced_keys:
        reduced_log[key] += log.get(key, 0)

    eps = 1e-7
    sa_sample_size = reduced_log["sa_sample_size"]
    sa_loss = reduced_log["sa_loss"] / (sa_sample_size + eps) / math.log(2)
    tlm_sample_size = reduced_log["tlm_sample_size"]
    tlm_loss = reduced_log["tlm_loss"] / (tlm_sample_size + eps) / math.log(2)
    mlm_sample_size = reduced_log["mlm_sample_size"]
    mlm_loss = reduced_log["mlm_loss"] / (mlm_sample_size + eps) / math.log(2)
    sample_size = sa_sample_size + tlm_sample_size + mlm_sample_size
    loss = (reduced_log["sa_loss"] + reduced_log["tlm_loss"] + reduced_log["mlm_loss"]) / (sample_size + eps) / math.log(2)

    # WALKAROUND
    if reduced_log["sa_EM_tot"] < 1: reduced_log["sa_EM_tot"] = 1

    agg_output = {
      'loss': loss,
      'ntokens': reduced_log["sa_ntokens"] + reduced_log["mlm_ntokens"],
      'nsentences': reduced_log["sa_nsentences"] + reduced_log["mlm_nsentences"],
      'dwa_loss': sa_loss,
      'dwa_sample_size': sa_sample_size,
      'dwa_EM': 0 if reduced_log["sa_EM_tot"] == 0 else 100 * reduced_log["sa_EM"] / reduced_log["sa_EM_tot"],
      'mlm_loss': mlm_loss,
      'mlm_sample_size': mlm_sample_size,
      'tlm_loss': tlm_loss,
      'tlm_sample_size': tlm_sample_size,
      'sample_size': sample_size,
    }

    # DEBUG
    # for k, v in agg_output.items():
    #   print("%s: %.2f" % (k, v), end=" | ")
    # print("")

    return agg_output
  
  def construct_tensor_from_list(self, idx_list2d, lens, pad_idx, device=None):
    max_len = max(lens)
    padded_list = [list_i + [pad_idx] * (max_len - lens[i]) for i, list_i in enumerate(idx_list2d)]
    tensor = torch.LongTensor(padded_list)
    if device is not None:
      tensor = tensor.to(device=device)
    return tensor
  
  def prepare_positions(self, sample, aligned_tokens, device=None):
    masked_tokens = sample['target'].ne(self.padding_idx)
    bsz = masked_tokens.size(0)
    src_fr, src_to, trg_fr, trg_to = sample["offsets"]

    # NOTE aligned_tokens should be extracted from the jointly encoded representations
    align_dicts = []
    for tokens_i in aligned_tokens:
      dict_i = {}
      for src, trg in tokens_i:
        dict_i[src] = trg
        dict_i[trg] = src
      align_dicts.append(dict_i)
    
    positions_fwd = [[] for i in range(bsz)]
    positions_bwd = [[] for i in range(bsz)]
    masked_positions_fwd = [[] for i in range(bsz)]
    masked_positions_bwd = [[] for i in range(bsz)]

    pos_cnt_fwd = [0] * bsz
    pos_cnt_bwd = [0] * bsz
    for ij in masked_tokens.nonzero():
      i = ij[0].item()
      masked_j = ij[1].item()
      if masked_j not in align_dicts[i]: continue
      aligned_j = align_dicts[i][masked_j]
      if src_fr[i] <= masked_j < src_to[i] and trg_fr[i] <= aligned_j < trg_to[i]:
        masked_positions_fwd[i].append(masked_j)
        positions_fwd[i].append(aligned_j)
        pos_cnt_fwd[i] += 1
      elif src_fr[i] <= aligned_j < src_to[i] and trg_fr[i] <= masked_j < trg_to[i]:
        masked_positions_bwd[i].append(masked_j)
        positions_bwd[i].append(aligned_j)
        pos_cnt_bwd[i] += 1
      else:
        print("[W] Value Error of alignments!!!")
        continue
    
    positions_fwd = self.construct_tensor_from_list(positions_fwd, pos_cnt_fwd, DwaMlmTlm.IGNORE_INDEX, device=device)
    positions_bwd = self.construct_tensor_from_list(positions_bwd, pos_cnt_bwd, DwaMlmTlm.IGNORE_INDEX, device=device)
    masked_positions_fwd = self.construct_tensor_from_list(masked_positions_fwd, pos_cnt_fwd, DwaMlmTlm.IGNORE_INDEX, device=device)
    masked_positions_bwd = self.construct_tensor_from_list(masked_positions_bwd, pos_cnt_bwd, DwaMlmTlm.IGNORE_INDEX, device=device)

    return positions_fwd, positions_bwd, masked_positions_fwd, masked_positions_bwd
  
  def forward_denoise_word_alignment(self, model, sample, reduce=True, aligned_tokens=None, use_tlm_loss=True):

    src_fr, src_to, trg_fr, trg_to = sample["offsets"]
    features, _ = model(**sample["net_input_tlm"], features_only=True)

    device = features.device
    positions_fwd, positions_bwd, masked_positions_fwd, masked_positions_bwd = \
      self.prepare_positions(sample, aligned_tokens, device=device)

    if use_tlm_loss:
      tlm_loss, tlm_sample_size, tlm_logging_output = self.forward_masked_lm(
        features, sample["target"], model)

    fwd_loss, fwd_em_cnt, fwd_tot = self.get_token_align_loss(model, features, positions_fwd, masked_positions_fwd, trg_fr, trg_to)
    bwd_loss, bwd_em_cnt, bwd_tot = self.get_token_align_loss(model, features, positions_bwd, masked_positions_bwd, src_fr, src_to)
    loss = fwd_loss + bwd_loss
    em_cnt = fwd_em_cnt + bwd_em_cnt
    tot = fwd_tot + bwd_tot
    em = 0 if tot == 0 else 100.0 * em_cnt / tot

    sample_size = tot
    logging_output = {
      'sa_loss': _get_logging_loss(loss),
      'sa_EM': em_cnt,
      'sa_EM_tot': tot,
      'sa_nsentences': sample["nsentences"],
      'sa_ntokens': sample["ntokens"],
      'sa_sample_size': sample_size,
    }

    if use_tlm_loss:
      loss += tlm_loss
      sample_size += tlm_sample_size
      logging_output.update(tlm_logging_output)
    else:
      hidden_sz = features.size(-1)
      dep_rep = features.new(hidden_sz).fill_(0)
      dep_rep = model.output_layer(dep_rep, features_only=True)
      loss += dep_rep.mean() * 0.0
    
    if hasattr(model, "forward_proj"):
      hidden_sz = features.size(-1)
      dep_rep = features.new(hidden_sz).fill_(0)
      dep_rep = model.forward_proj(dep_rep[None, :], "en", use_model_fast=True)
      loss += dep_rep.mean() * 0.0

    return loss, sample_size, logging_output

  def get_token_align_loss(self, model, features, positions, masked_positions, fr, to):

    if len(positions.view(-1)) <= 0:
      dep_rep = features[0, 0, :]
      loss = dep_rep.mean() * 0.0
      em_cnt = tot = 0
      return loss, em_cnt, tot

    bsz, seq_len, hidden_sz = features.size()
    # _, max_num_spans = positions.size()
    device = features.device

    #  get attention mask
    fr_tensor = torch.LongTensor(fr).to(device=device)
    to_tensor = torch.LongTensor(to).to(device=device)
    # bsz, seq_len
    attention_mask = (torch.arange(seq_len)[None, :].to(device=device) >= fr_tensor[:, None]) & (torch.arange(seq_len)[None, :].to(device=device) < to_tensor[:, None])
    # bsz, 1, seq_len
    attention_mask = attention_mask[:, None, :]
    attention_mask = (1.0-attention_mask.half()) * -1e4
    # print(attention_mask)

    # masked_features: bsz, max_num_spans, hidden
    masked_features = self._positions2masked_features(masked_positions, features, hidden_sz)
    q_features = model.q_linear(masked_features)
    # bsz, len, hidden
    k_features = model.k_linear(features)
    # bsz, max_num_spans, len
    attention_scores = torch.matmul(q_features, k_features.transpose(-1, -2))
    attention_scores = attention_scores / math.sqrt(hidden_sz)
    attention_scores = attention_scores + attention_mask
    logits = attention_scores

    loss_fct = nn.CrossEntropyLoss(ignore_index=DwaMlmTlm.IGNORE_INDEX, reduction='sum')
    loss = loss_fct(logits.view(-1, logits.size(-1)), positions.view(-1))

    # calc EM & F1
    def _get_em_mask(logits, targets):
      logits = logits.view(-1, logits.size(-1))
      targets = targets.view(-1)
      prediction = logits.argmax(dim=-1)
      return targets == prediction, (targets != DwaMlmTlm.IGNORE_INDEX).sum().item()
     
    em_mask, tot = _get_em_mask(logits, positions)
    em_cnt = em_mask.sum().item()
    return loss, em_cnt, tot
