import collections
import logging
import math
import torch

from torch import nn
from torch.nn import functional as F
from torch import distributed

from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion


logger = logging.getLogger(__name__)


@register_criterion('xlco')
class XlCoCriterion(FairseqCriterion):

  @staticmethod
  def add_args(parser):
    parser.add_argument('--xlco_queue_size', default=256, type=int)
    parser.add_argument('--xlco_softmax_tau', default=0.25, type=float)
    parser.add_argument('--xlco_layer', default=8, type=int)
    parser.add_argument('--xlco_lambda', default=1.0, type=float)

  def __init__(self, args, task):
    super().__init__(args, task)
    self.criterion = nn.CrossEntropyLoss(reduction='sum')
    self.xlco_queue_size = args.xlco_queue_size
  
  def contrastive_loss(self, q, k, queue):
    queue = queue.clone().detach()
    N, C = q.size()
    assert k.size() == (N,C), (N, C, k.size())

    logits_pos = torch.bmm(q.view(N, 1, C), k.view(N, C, 1)).view(N, 1)
    logits_neg = torch.mm(q, queue.transpose(0, 1))
    logits = torch.cat([logits_pos, logits_neg], dim=1) / self.args.xlco_softmax_tau

    labels = torch.zeros(N).cuda().long()
    loss = self.criterion(logits, labels)
    cxlm_ncorrect = utils.item((logits.argmax(dim=1) == labels).sum())
    return loss, cxlm_ncorrect
  
  def _get_logging_loss(self, loss, reduce=True):
    if loss is None: return 0
    return utils.item(loss.data) if reduce else loss.data
  
  def forward_xlco(self, model, sample, reduce=True):
    cxlm_head_key = "share_lang"

    with torch.no_grad():
      _, inner_states = model(**sample['tgt_net_input'], use_model_fast=False, features_only=True, return_all_hiddens=True)
      slow_features = inner_states["inner_states"][self.args.xlco_layer]
      slow_features = slow_features[0, :, :].clone().detach()
      if self.args.use_proj:
        slow_rep = model.forward_proj(
          slow_features, cxlm_head_key, use_model_fast=False)
      else: slow_rep = slow_features

    if model.is_queue_ready():
      fast_features, inner_states = model(**sample['src_net_input'],
        use_model_fast=True, features_only=True, return_all_hiddens=True)
      fast_features = inner_states["inner_states"][-1][0, :, :]
      fast_features8 = inner_states["inner_states"][self.args.xlco_layer][0, :, :]

      if self.args.use_proj:
        fast_rep = model.forward_proj(
          fast_features8, cxlm_head_key, use_model_fast=True)
      else: fast_rep = fast_features8

      cxlm_loss, cxlm_ncorrect = self.contrastive_loss(fast_rep, slow_rep, model.queue)

      cxlm_loss *= self.task.xlco_lambda
      loss = cxlm_loss

      # NOTE WALKAROUND We have to use all parameters for ddp.
      dep_logits = model.output_layer(fast_features, features_only=True)
      loss += dep_logits.mean() * 0.0

      if hasattr(model, "q_linear"):
        hidden_sz = fast_features.size(-1)
        dep_rep = fast_features.new(hidden_sz).fill_(0)
        dep_rep1 = model.q_linear(dep_rep).mean()
        dep_rep2 = model.k_linear(dep_rep).mean()
        loss += dep_rep1 * 0.0 + dep_rep2 * 0.0

      cxlm_logging_loss = self._get_logging_loss(cxlm_loss, reduce)
    else:
      loss = None
      cxlm_logging_loss = 0
      cxlm_ncorrect = 0
    
    if model.training:
      rank = self.args.distributed_rank
      model.update_queue(slow_rep)

    sample_size = sample["nsentences"]
    logging_output = {
      'cxlm_loss': cxlm_logging_loss,
      'cxlm_nsentences': sample["nsentences"],
      'cxlm_ntokens': sample["ntokens"],
      'cxlm_sample_size': sample_size,
      'cxlm_ncorrect': cxlm_ncorrect,
    }
    return loss, sample_size, logging_output
  
  def forward_mlm(self, model, sample, reduce=True):
    masked_tokens = sample['target'].ne(self.padding_idx)
    sample_size = masked_tokens.int().sum().item()

    # (Rare case) When all tokens are masked, the model results in empty
    # tensor and gives CUDA error.
    if sample_size == 0:
      masked_tokens = None
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
    if self.args.use_proj:
      dep_rep = model.forward_proj(features[:, 0, :], "en", use_model_fast=True)
      loss += dep_rep.mean() * 0.0
    if hasattr(model, "q_linear"):
      hidden_sz = features.size(-1)
      dep_rep = features.new(hidden_sz).fill_(0)
      dep_rep1 = model.q_linear(dep_rep).mean()
      dep_rep2 = model.k_linear(dep_rep).mean()
      loss += dep_rep1 * 0.0 + dep_rep2 * 0.0
    return loss, sample_size, logging_output


  def forward(self, model, sample, reduce=True, mlm=False):
    if mlm:
      return self.forward_mlm(model, sample, reduce=reduce)
    else:
      return self.forward_xlco(model, sample, reduce=reduce)

  @staticmethod
  def aggregate_logging_outputs(logging_outputs):
    """Aggregate logging outputs from data parallel training."""
    # loss_sum = sum(log.get('loss', 0) for log in logging_outputs)

    reduced_log = collections.defaultdict(float)
    reduced_keys = ["cxlm_loss", "mlm_loss", "cxlm_ntokens",
      "cxlm_nsentences", "mlm_ntokens", "mlm_nsentences", "cxlm_sample_size",
      "mlm_sample_size", "cxlm_ncorrect", "momentum"]

    for log in logging_outputs:
      for key in reduced_keys:
        reduced_log[key] += log.get(key, 0)

    loss_sum_cxlm = reduced_log["cxlm_loss"]
    loss_sum_mlm = reduced_log["mlm_loss"]
    loss_sum = loss_sum_cxlm + loss_sum_mlm
    cxlm_ntokens = reduced_log["cxlm_ntokens"]
    cxlm_nsentences = reduced_log["cxlm_nsentences"]
    mlm_ntokens = reduced_log["mlm_ntokens"]
    mlm_nsentences = reduced_log["mlm_nsentences"]
    cxlm_sample_size = reduced_log["cxlm_sample_size"]
    mlm_sample_size = reduced_log["mlm_sample_size"]
    sample_size = cxlm_sample_size + mlm_sample_size
    ncorrect = reduced_log["cxlm_ncorrect"]

    eps = 1e-7

    agg_output = {
      'loss': loss_sum / (sample_size + eps) / math.log(2),
      'ntokens': cxlm_ntokens + mlm_ntokens,
      'nsentences': cxlm_nsentences + mlm_nsentences,
      'xlco_loss': loss_sum_cxlm / (cxlm_sample_size + eps) / math.log(2),
      'mlm_loss': loss_sum_mlm / (mlm_sample_size + eps) / math.log(2),
      'xlco_accuracy': 100.0 * ncorrect / (cxlm_nsentences + eps),
      'momentum': reduced_log["momentum"] / len(logging_outputs),
      'xlco_ntokens': cxlm_ntokens,
      'xlco_nsentences': cxlm_nsentences,
      'mlm_ntokens': mlm_ntokens,
      'mlm_nsentences': mlm_nsentences,
      'xlco_sample_size': cxlm_sample_size,
      'mlm_sample_size': mlm_sample_size,
      'sample_size': sample_size,
    }

    # DEBUG
    # for k, v in agg_output.items():
    #   print("%s: %f" % (k, v), end=" | ")
    # print("")

    return agg_output