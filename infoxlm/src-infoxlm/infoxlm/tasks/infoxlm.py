import os
import torch

from functools import lru_cache
from fairseq.tasks import register_task, FairseqTask
from fairseq.data.dictionary import Dictionary
from fairseq.data import FairseqDataset
from fairseq import utils

from infoxlm.data import mlm_utils
from infoxlm.data.dict_dataset import DictDataset
from infoxlm.data.xlco_dataset import get_xlco_dataset
from infoxlm.tasks.mlm import Mlm


def _prepare_sample(sample, cuda=True, fp16=True):
  if sample is None or len(sample) == 0:
    return None

  if cuda:
    sample = utils.move_to_cuda(sample)

  def apply_half(t):
    if t.dtype is torch.float32:
      return t.half()
    return t

  if fp16:
    sample = utils.apply_to_sample(apply_half, sample)

  return sample


@register_task("infoxlm")
class InfoXLM(Mlm):

  @staticmethod
  def add_args(parser):
    Mlm.add_args(parser)
    parser.add_argument('--tlm_data', type=str, default="")
    parser.add_argument('--xlco_data', type=str, default="")
    # e.g. constant,0.999
    # e.g. linear,0,700000,0.999,1.0
    parser.add_argument('--xlco_momentum', default="constant,0.999", type=str)
    parser.add_argument('--xlco_enable_step', default=-1, type=int)
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # NOTE walkaround for model building
    # Actually, self.langs represents the keys of proj heads
    self.model_langs = ["share_lang"]
    self.xlco_lambda = self.args.xlco_lambda

    # parse xlco_momentum
    cxlm_args = self.args.xlco_momentum.split(",")
    # self.constant_xlco_momentum = True
    self.cxlm_scheduler = "constant"
    self.constant_momentum_refresh_interval = -1
    if cxlm_args[0] == "constant":
      self._xlco_momentum = float(cxlm_args[1])
      print("Momentum args: consant momentum: %.4f" % (self._xlco_momentum), flush=True)
    elif cxlm_args[0] == "linear":
      # self.constant_xlco_momentum = False
      self.cxlm_scheduler = "linear"
      self._mom_schedule_begin, self._mom_schedule_end, self._xlco_momentum_min, self._xlco_momentum_max = map(float, cxlm_args[1:])
      print("Momentum args: linear self._mom_schedule_begin: %.4f, self._mom_schedule_end: %.4f, self._xlco_momentum_min: %.4f, self._xlco_momentum_max: %.4f " % (self._mom_schedule_begin, self._mom_schedule_end, self._xlco_momentum_min, self._xlco_momentum_max), flush=True)
      assert self._mom_schedule_end >= self._mom_schedule_begin
    elif cxlm_args[0] == "constant_with_refresh":
      self._xlco_momentum = float(cxlm_args[1])
      self.constant_momentum_refresh_interval = int(cxlm_args[2])
      print("Momentum args: consant momentum: %.4f, refresh interval: %d" % (self._xlco_momentum, self.constant_momentum_refresh_interval), flush=True)
    elif cxlm_args[0] == "exponential":
      # example exponential,0.51,0.0,0.9995
      self.cxlm_scheduler = "exponential"
      self._xlco_momentum_alpha, self._xlco_momentum_min, self._xlco_momentum_max = map(float, cxlm_args[1:])
      print("Momentum args: exponential self._xlco_momentum_alpha: %.4f, self._xlco_momentum_min: %.4f, self._xlco_momentum_max: %.4f " % (self._xlco_momentum_alpha, self._xlco_momentum_min, self._xlco_momentum_max), flush=True)
    else:
      raise NotImplementedError
    self._cur_momentum = self.get_xlco_momentum(0)

    print("Test get_xlco_momentum ...")
    for i in range(10):
      num_updates = i * 100000
      print("num_updates: %d get_xlco_momentum:%f" % (i, self.get_xlco_momentum(num_updates)))
  
  def get_xlco_momentum(self, num_updates):
    if self.cxlm_scheduler == "constant":
      if self.constant_momentum_refresh_interval == -1:
        return self._xlco_momentum
      else:
        if num_updates % self.constant_momentum_refresh_interval == 0:
          return 0.0
        else:
          return self._xlco_momentum
    elif self.cxlm_scheduler == "linear":
      if num_updates <= self._mom_schedule_begin:
        return self._xlco_momentum_min
      elif num_updates >= self._mom_schedule_end:
        return self._xlco_momentum_max
      else:
        return (num_updates - self._mom_schedule_begin) * (self._xlco_momentum_max - self._xlco_momentum_min) / (self._mom_schedule_end - self._mom_schedule_begin) + self._xlco_momentum_min
    elif self.cxlm_scheduler == "exponential":
      if num_updates <= 0: return self._xlco_momentum_min
      mom = 1.0 - num_updates ** (-self._xlco_momentum_alpha)
      mom = max(mom, self._xlco_momentum_min)
      mom = min(mom, self._xlco_momentum_max)
      return mom
    else:
      raise ValueError

  def prepare_train(self, model, criterion):
    print("| Prepare train ...", flush=True)

    # DEBUG
    # print("Test get_xlco_momentum ...")
    # for i in range(10):
    #   num_updates = i * 100000
    #   print("num_updates: %d get_xlco_momentum:%f" % (i, self.get_xlco_momentum(num_updates)))

    self.model = model
    model.train()
    if not model.is_queue_ready():
      self.fill_queue(criterion)
      assert model.is_queue_ready()
  
  def fill_queue(self, criterion):
    print("| Filling language queue ... ")
    fill_opt_cnt = 0
    dummy_batch = None
    epoch_itr = self.get_batch_iterator(
      dataset=self.load_xlco_dataset(self.args.train_subset),
      max_tokens=self.args.max_tokens,
      max_sentences=self.args.max_sentences,
      max_positions=utils.resolve_max_positions(
        self.max_positions(), self.model.max_positions()
      ),
      ignore_invalid_inputs=True,
      required_batch_size_multiple=self.args.required_batch_size_multiple,
      seed=self.args.seed,
      num_shards=self.args.distributed_world_size,
      shard_id=self.args.distributed_rank,
      num_workers=0,
      epoch=0,)
    itr = epoch_itr.next_epoch_itr(
      fix_batches_to_gpus=self.args.fix_batches_to_gpus,
      shuffle=False,)
    # DEBUG
    # NOTE add a ref to prevent deletion
    # self._fill_queue_itr = itr

    ddp_size = 1 if not hasattr(self.args, "distributed_world_size") else self.args.distributed_world_size
    tot_fill_opt = criterion.xlco_queue_size // self.args.max_sentences // ddp_size + 100
    # print("| %d filling opt in total." % tot_fill_opt, flush=True)
    for _ in range(tot_fill_opt):
      sample = next(itr)
      if dummy_batch is None: dummy_batch = sample
      sample = _prepare_sample(sample)
      if sample is None:
        sample = _prepare_sample(dummy_batch)
        print("| [W] a dummy batch used", flush=True)
      with torch.no_grad():
        criterion(self.model, sample)
      
      if fill_opt_cnt % 100 == 0:
        print("| Filling queue, fill_opt_cnt: %d" % fill_opt_cnt, flush=True)
      fill_opt_cnt += 1
    
    print("| %d filling opt in total." % fill_opt_cnt, flush=True)
    
    assert self.model.is_queue_ready()
    print("| queue.mean(): %f, queue.var(): %f" % (self.model.queue.mean().item(), self.model.queue.var().item()))

    del itr
    del epoch_itr
  
  def update_step(self, num_updates):

    if num_updates < self.args.xlco_enable_step:
      self.xlco_lambda = 0.0
      self._cur_momentum = 0.0
      if num_updates + 5 >= self.args.xlco_enable_step:
        self.model.update_slow_weight(0.0)
    else:
      self.xlco_lambda = self.args.xlco_lambda
      self._cur_momentum = self.get_xlco_momentum(num_updates)
      self.model.update_slow_weight(self._cur_momentum)
    # pass

  def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
    model.train()
    agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
    
    # cxlm_step
    loss, sample_size, logging_output = criterion(model, sample["xlco"])
    if loss is None:
      raise ValueError
    if ignore_grad: loss *= 0
    cxlm_loss = loss
    optimizer.backward(cxlm_loss)
    if loss is not None:
      agg_loss += cxlm_loss.detach().item()
    agg_sample_size += sample_size
    agg_logging_output.update(logging_output)

    # tlm step
    loss, sample_size, logging_output = criterion(model, sample["tlm"], mlm=True)
    if ignore_grad: loss *= 0
    tlm_loss = loss
    optimizer.backward(tlm_loss)
    agg_loss += tlm_loss.detach().item()
    agg_sample_size += sample_size
    agg_logging_output.update(logging_output)

    # mlm_step
    loss, sample_size, logging_output = criterion(model, sample["mlm"], mlm=True)
    if ignore_grad: loss *= 0
    optimizer.backward(loss)
    agg_loss += loss.detach().item()
    agg_sample_size += sample_size
    # agg_logging_output.update(logging_output)
    for key, value in logging_output.items():
      agg_logging_output[key] += value
    
    # print("DEBUG2: %s" % str(agg_logging_output))
    agg_logging_output["momentum"] = self._cur_momentum
    return agg_loss, agg_sample_size, agg_logging_output

  def load_dataset(self, split, epoch=0, combine=False, **kwargs):
    print("| Loading dataset at epoch %d" % epoch, flush=True)
    
    args = self.args
    sid = 0
    dataset_path = os.path.join(args.data, "train.%d" % sid)
    mlm_dataset = mlm_utils.get_mlm_dataset(
      args, dataset_path, self.dictionary, self.mask_idx, self.mww, combine=False)

    dataset_path = os.path.join(args.tlm_data, "train.%d" % sid)
    tlm_dataset = mlm_utils.get_mlm_dataset(
      args, dataset_path, self.dictionary, self.mask_idx, self.mww, combine=False)

    dataset_path = os.path.join(args.xlco_data, "train.%d" % sid)
    xlco_dataset = get_xlco_dataset(
      args, dataset_path, self.dictionary, self.mask_idx, combine=False)

    dataset = DictDataset({
      "tlm": tlm_dataset,
      "mlm": mlm_dataset,
      "xlco": xlco_dataset
    })

    # NOTE Set dataset epoch as sid for different random state 
    # of each shard, because when local indices are the same, the
    # random states are the same.
    dataset.set_epoch(sid)
    self.datasets[split] = dataset
  
  def load_xlco_dataset(self, split, epoch=0, combine=False, **kwargs):
    args = self.args
    dataset_path = os.path.join(args.xlco_data, "train.0")
    xlco_dataset = get_xlco_dataset(
      args, dataset_path, self.dictionary, self.mask_idx)
    return xlco_dataset
