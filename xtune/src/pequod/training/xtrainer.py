import logging
import numpy as np
import os
import torch
import random

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

try:
  from apex import amp
except ImportError:
  pass

from src.pequod.trainer import (Trainer,
  XClassificationTrainer, XQATrainer, SelfTrainer)
from transformers import AdamW, ConstantLRSchedule, WarmupLinearSchedule

logger = logging.getLogger(__name__)


class BaseTrainer(Trainer):

  def __init__(self, args, model, tokenizer):
    super().__init__(args, model, tokenizer)
    self.optimizer = None
    self.scheduler = None
    self.global_steps = 0
    self.all_shard_fn = {}
  
  def init_optimizer(self, model, lr, t_total, fixed=None):
    args = self.args
    no_decay = ['bias', 'LayerNorm.weight']
    if fixed is None: fixed = []
    optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay) and not any(f in n for f in fixed)
        ], "weight_decay": args.weight_decay},
      {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay) and not any(f in n for f in fixed)
        ], "weight_decay": 0.0}]
    # TODO calculate t_total
    optimizer = AdamW(
      optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)

    if args.scheduler == "linear":
      warmup_steps = t_total * args.warmup_ratio if args.warmup_steps == -1 else args.warmup_steps
      logger.info("Setting scheduler, warmups=%d, lr=%.7f, total_updates=%d" % (
        warmup_steps, lr, t_total))
      scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    elif args.scheduler == "constant":
      logger.info("Setting scheduler, ConstantLRSchedule")
      scheduler = ConstantLRSchedule(optimizer)
    else:
      raise ValueError
    return optimizer_grouped_parameters, optimizer, scheduler
  
  def optim_step(self, **kwargs):
    args = self.args
    # self.model.zero_grad()
    if args.fp16:
      # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
      #   scaled_loss.backward()
      torch.nn.utils.clip_grad_norm_(
        amp.master_params(self.optimizer), args.max_grad_norm)
    else:
      # loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
    
    self.optimizer.step()
    self.scheduler.step()
    self.model.zero_grad()
  
  def backward_step(self, loss, **kwargs):
    args = self.args
    if args.accumulate_steps > 1:
        loss = loss / args.accumulate_steps
    if args.fp16:
      with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        scaled_loss.backward()
    else:
      loss.backward()

  def step(self, *args, **kwargs):
    algo = kwargs.pop("algo", self.args.algo)
    if algo is None: algo = self.args.algo
    step_func_names = ["%s_step" % s for s in algo.split(",")]
    return getattr(self, random.choice(step_func_names))(*args, **kwargs)
  
  def base_step(self, batches, is_qa=False, **kwargs):
    tot_loss = 0.0
    for step_batches in batches:
      batch = step_batches[0]
      batch_dict = self._parse_batch(batch)
      loss = self.model(**batch_dict)[0]
      self.backward_step(loss)
      tot_loss += loss.item()
    self.optim_step()
    return tot_loss / len(batches)

  def train_full_epoch(self, train_ds_keys, epoch_id, is_qa=False, algo=None):
    if train_ds_keys == "": return
    logger.info("***** Training epoch %d - train_ds_keys: %s *****" % (
      epoch_id, str(train_ds_keys)))
    args = self.args

    n_instances = 0
    data_loaders = []
    if isinstance(train_ds_keys, str):
      train_ds_keys = train_ds_keys.split(";")

    for ds_key_str in train_ds_keys:
      data_loaders.append(self.get_dataloader_from_str(ds_key_str, epoch_id))
    
    if self.optimizer is None:
      _, self.optimizer, self.scheduler = self.init_optimizer(
        self.model, args.learning_rate, len(data_loaders[0]) * args.num_train_epochs // args.accumulate_steps)
      if args.fp16:
        self.model, self.optimizer = amp.initialize(
          self.model, self.optimizer, opt_level=args.fp16_opt_level)
    
    model = self.model
    model.train()
    losses = []

    step = 0
    step_func_dict = {"batches": [], "is_qa": is_qa, "epoch_id": epoch_id}
    # for step, batches in enumerate(zip(*data_loaders)):
    for batches in zip(*data_loaders):
      # step_func_dict = {"batches": batches, "is_qa": is_qa, "epoch_id": epoch_id}
      step_func_dict["batches"].append(batches)
      if len(step_func_dict["batches"]) == args.accumulate_steps:
        loss = self.step(**step_func_dict, algo=algo)
        losses.append(loss)
        step_func_dict["batches"] = []
      else:
        continue

      n_instances += args.train_batch_size * args.accumulate_steps
      self.global_steps += 1
      step += 1

      if step % args.logging_steps == 0:
        cur_lr = self.scheduler.get_lr()[0]
        logger.info(
          "Epoch %d - step %7d - global step %d - lr %.8f - n instances %7d - loss: %.4f " % (
          epoch_id, step, self.global_steps, cur_lr, n_instances, sum(losses) / len(losses)))
        losses = []
  
  def _parse_ds_key(self, ds_key_str):
    assert isinstance(ds_key_str, str)
    args, kwargs = [], {}
    for s in ds_key_str.split(","):
      if ":" in s:
        k, v = s.split(":")
        kwargs[k] = v
      else: args.append(s)
    return args, kwargs
  
  def get_mixed_dataloader(self, *dataloaders):
    iters = [iter(d) for d in dataloaders]
    len_dl = len(iters)
    finish = [False] * len_dl
    cnt = 0
    while cnt < len_dl:
      idx = random.randint(0, len_dl - 1)
      if finish[idx]: continue
      try:
        yield next(iters[idx])
      except StopIteration:
        finish[idx] = True
        cnt += 1
  
  def get_all_shard_fn(self, *args, cache_filename=None):
    if args in self.all_shard_fn: return self.all_shard_fn[args]
    all_shard_fn = []
    shard_id = 0
    while True:
      fn = cache_filename + "." + str(shard_id)
      if not os.path.exists(fn): break
      all_shard_fn.append(fn)
      shard_id += 1
    logger.info("%d shards found." % len(all_shard_fn))
    np.random.shuffle(all_shard_fn)
    self.all_shard_fn[args] = all_shard_fn
    return all_shard_fn
  
  def get_sharded_dataloader(self, *args, **kwargs):
    logger.info("Getting dataloader - args: %s" % str(args))
    split, lang, epoch_id = args
    cache_key = self.get_cache_key()
    cache_filename = os.path.join(
      self.args.data_dir, "cached_%s_%s_%s" % (split, lang, cache_key))

    all_shard_fn = self.get_all_shard_fn(
      split, lang, cache_filename=cache_filename)
    fn = all_shard_fn[epoch_id % len(all_shard_fn)]

    logger.info("Loading dataset from %s" % str(fn))
    tensor_dict = torch.load(fn)
    tensors = []
    for _, t in tensor_dict.items():
      tensors.append(t.long())
    dataset = TensorDataset(*tensors)
    
    sampler = self.get_sampler(dataset, *args, **kwargs)
    dataloader = DataLoader(dataset, sampler=sampler,
      batch_size=self.args.train_batch_size)
    return dataloader
  
  def get_dataloader_from_str(self, ds_key_str, epoch_id):
    if ds_key_str.startswith("mix("):
      # example: mix(train,en,cut:200|train,zh,cut:20)
      assert ds_key_str[-1] == ")"
      ds_key_str = ds_key_str[4:-1]

      dataloaders = []
      for dks in ds_key_str.split("|"):
        dataloaders.append(self.get_dataloader_from_str(dks, epoch_id))
      return self.get_mixed_dataloader(*dataloaders)
    
    ds_key_args, ds_key_kwargs = self._parse_ds_key(ds_key_str)
    sharded_dataloader = ds_key_kwargs.pop("sharded_dataloader", "")
    if sharded_dataloader == "True":
      return self.get_sharded_dataloader(*ds_key_args, epoch_id, **ds_key_kwargs)
    return self.get_dataloader(*ds_key_args, **ds_key_kwargs)


def get_model_class(proto_train_class=None, is_qa=False):

  class ProtoXClassificationTrainer(XClassificationTrainer, proto_train_class):

    def __init__(self, args, model, tokenizer):
      proto_train_class.__init__(self, args, model, tokenizer)
      # _, self.optimizer, self.scheduler = self.init_optimizer(
      #   model, args.learning_rate)
    
    def train_full_epoch(self, train_ds_keys, epoch_id, algo=None):
      proto_train_class.train_full_epoch(
        self, train_ds_keys, epoch_id, is_qa=False, algo=algo)
    
    def before_loop(self):
      # args = self.args
      # if args.labeling_unlabeled_data:
      #   assert args.semi_split != ""
      #   for lang in args.test_langs.split(","):
      #     logger.info("Labeling lang: %s" % lang)
      #     self.labeling_dataset(self.model, (args.semi_split, lang))
      pass
    
    def init_optimizer(self, *args, **kwargs):
      return proto_train_class.init_optimizer(self, *args, **kwargs)

  class ProtoXQATrainer(XQATrainer, proto_train_class):

    def __init__(self, args, model, tokenizer):
      proto_train_class.__init__(self, args, model, tokenizer)
      # _, self.optimizer, self.scheduler = self.init_optimizer(
      #   model, args.learning_rate)
      self.example_feature_cache = {}
    
    def train_full_epoch(self, train_ds_keys, epoch_id, algo=None):
      proto_train_class.train_full_epoch(
        self, train_ds_keys, epoch_id, is_qa=True, algo=algo)
    
    def init_optimizer(self, *args, **kwargs):
      return proto_train_class.init_optimizer(self, *args, **kwargs)
    
  return ProtoXQATrainer if is_qa else ProtoXClassificationTrainer
