import os
import json
import logging
import random
import torch
import numpy as np

try:
  from apex import amp
except ImportError:
  pass

from torch.utils.data import (DataLoader, 
  RandomSampler, SequentialSampler, TensorDataset, SubsetRandomSampler,
  Subset, ConcatDataset)
#from transformers import AdamW, ConstantLRSchedule, WarmupLinearSchedule
from transformers import AdamW, get_constant_schedule, get_linear_schedule_with_warmup

from src.pequod.training import to_cuda, set_seed
from src.pequod.eval import (eval_classification, eval_qa,
  score_dict_to_string, score_dicts_to_latex)
from src.pequod.data import xdoc, xqa
from src.pequod.data.sampler import SubSampler


logger = logging.getLogger(__name__)


class Trainer(object):

  def __init__(self, args, model, tokenizer):
    self.args = args
    self.datasets = {}
    self.dataloaders = {}
    self.iter_cache = {}
    self.best_scores = {}
    self.model = model
    self.tokenizer = tokenizer
  
  def run(self):
    raise NotImplementedError
  
  def _parse_batch(self, batch, **kwargs):
    _batch = to_cuda(batch)
    # _batch = batch
    ret = {"input_ids": _batch[0],
      "attention_mask": _batch[1],
      "token_type_ids": _batch[2] if self.args.model_type == "bert" else None,
      "labels": _batch[3]}
    ret.update(**kwargs)
    return ret

  def train_epoch(self, split, lang, epoch_id):
    raise NotImplementedError

  def before_loop(self):
    return
  
  def train_full_epoch(self, split, lang, epoch_id):
    raise NotImplementedError

  def eval_epoch(self, split, lang, epoch_id):
    raise NotImplementedError

  def load_and_cache_examples(self, *args, **kwargs):
    raise NotImplementedError
  
  def save(self, name, epoch=0):
    path = os.path.join(self.args.dump_path, "%s.pth" % name)
    logger.info("Saving %s to %s ..." % (name, path))
    data = {
      "epoch":epoch,
      "model":self.model.state_dict(),
      "params": {k: v for k, v in self.args.__dict__.items()}}
    torch.save(data, path)

  def get_dataset_deprecated(self, *args, **kwargs):
    logger.warning("cut_args is deprecated, please use train_ds_keys.")
    if args in self.datasets: return self.datasets[args]
    dataset = self.load_and_cache_examples(*args, **kwargs)
    cut_split, cut_num = self.args.cut_args.split(",")
    cut_num = int(cut_num)
    if cut_num != -1 and cut_num < len(dataset) and cut_split == args[0]:
      # cut_indices = random.sample(range(len(dataset)), cut_num)
      cut_indices = [i for i in range(cut_num)]
      # dataset = Subset(dataset, cut_indices)
      dataset = TensorDataset(
        *tuple(tensor[cut_indices] for tensor in dataset.tensors))
    self.datasets[args] = dataset
    return dataset
  
  def get_cache_key(self):
    cache_key = "%s-%s" % (self.args.model_key, self.args.model_type)
    return cache_key
  
  def get_dataset(self, *args, **kwargs):
    if args in self.datasets: return self.datasets[args]
    dataset = self.load_and_cache_examples(*args, **kwargs)

    cut_num = int(kwargs.pop("cut", "-1"))
    if cut_num != -1 and cut_num < len(dataset):
      cut_indices = [i for i in range(cut_num)]
      dataset = TensorDataset(
        *tuple(tensor[cut_indices] for tensor in dataset.tensors))
    self.datasets[args] = dataset
    return dataset
  
  def get_sampler(self, data_source, *args, **kwargs):
    shuffle = kwargs.get("shuffle", args[0] == "train")
    num_samples = kwargs.get("num_samples", None)

    if num_samples is not None:
      num_samples = int(num_samples)
      sampler = SubSampler(data_source, num_samples)
    else:
      sampler = RandomSampler(data_source) if shuffle else SequentialSampler(data_source)

    return sampler

  def get_dataloader(self, *args, **kwargs):
    logger.info("Getting dataloader - args: %s" % str(args))
    if args in self.dataloaders: return self.dataloaders[args]
    dataset = kwargs["dataset"] if "dataset" in kwargs \
      else self.get_dataset(*args, **kwargs)
    
    sampler = self.get_sampler(dataset, *args, **kwargs)

    dataloader = DataLoader(dataset, sampler=sampler,
      batch_size=self.args.train_batch_size)

    self.dataloaders[args] = dataloader
    return dataloader
  
  def next_batch(self, *args, **kwargs):
    if args not in self.iter_cache:
      self.iter_cache[args] = iter(self.get_dataloader(*args, **kwargs))
    try:
      ret = next(self.iter_cache[args])
    except StopIteration:
      self.iter_cache[args] = iter(self.get_dataloader(*args, **kwargs))
      ret = next(self.iter_cache[args])
    return ret
  
  def set_dataset(self, dataset, args):
    self.datasets[args] = dataset
    if args in self.dataloaders: self.dataloaders.pop(args)
    if args in self.iter_cache: self.iter_cache.pop(args)
  
  def copy_label(self, trg_key, src_key):
    src_ds = self.get_dataset(*src_key)
    trg_ds = self.get_dataset(*trg_key)
    new_trg_ds = TensorDataset(*(trg_ds.tensors[:-1]) + (src_ds.tensors[-1],))
    self.set_dataset(new_trg_ds, trg_key)


class SelfTrainer(Trainer):

  def __init__(self, args, model=None, tokenizer=None):
    super().__init__(args, model, tokenizer)
  
  def labeling_dataset(self, model, ds_key):
    logger.info("Labeling dataset %s" % str(ds_key))
    model.eval()
    dataset:TensorDataset = self.get_dataset(*ds_key)

    # NOTE all_labels must be the last
    preds = None
    for batch in self.get_dataloader(*ds_key, shuffle=False):
      with torch.no_grad():
        batch_dict = self._parse_batch(batch, labels=None)
        outputs = model(**batch_dict)
        logits = outputs[0]
      pred = logits.detach().cpu().numpy()
      preds = pred if preds is None else np.append(preds, pred, axis=0)

    new_labels = np.argmax(preds, axis=1)
    new_labels = torch.tensor(new_labels, dtype=torch.long)
    self.set_dataset(
      TensorDataset(*(dataset.tensors[:-1] + (new_labels, ))), ds_key)
    return preds
  
  def update_concat_dataset_cache(self, ds_keys, preds_list, key_prefix="concat"):
    """
    if preds_list[i] is None, then the ith dataset won't be cut by confidence.
    """
    assert len(ds_keys) == len(preds_list)
    assert all(ds_key[1:] == ds_keys[0][1:] for ds_key in ds_keys)
    new_split = "-".join((key_prefix,) + tuple(ds_key[0] for ds_key in ds_keys))
    logger.info("Concating %d dataset %s ..." % (len(ds_keys), new_split))
    new_ds_key = (new_split, ) + ds_keys[0][1:]

    datasets = []
    for ds_key, preds in zip(ds_keys, preds_list):
      dataset = self.get_dataset(*ds_key)
      if preds is None:
        datasets.append(dataset)
        continue
      new_labels = dataset.tensors[-1]
      confident_indices = []
      for i in range(len(new_labels)):
        if preds[i,new_labels[i]] >= self.args.confidence_threshold:
          confident_indices.append(i)
      logger.info(
        "Labeled %d confident examples out of %d examples for dataset %s" % (
        len(confident_indices), len(new_labels), str(ds_key)))
      if len(confident_indices) > 0:
        datasets.append(Subset(dataset, confident_indices))

    self.set_dataset(ConcatDataset(datasets), new_ds_key)
    # self.datasets[new_ds_key] = ConcatDataset(datasets)
    logger.info("Construct new dataset %s with %d examples" % (
      str(new_ds_key), len(self.datasets[new_ds_key])))
    return new_ds_key


class DistillTrainer(Trainer):

  def __init__(self, args, model=None, tokenizer=None):
    super().__init__(args, model, tokenizer)
  
  def labeling_dataset(self, model, ds_key):
    logger.info("Labeling dataset %s" % str(ds_key))
    model.eval()
    dataset:TensorDataset = self.get_dataset(*ds_key)

    preds = None
    for batch in self.get_dataloader(*ds_key, shuffle=False):
      with torch.no_grad():
        batch_dict = self._parse_batch(batch, labels=None)
        outputs = model(**batch_dict)
        logits = outputs[0]
      pred = logits.detach().cpu().numpy()
      preds = pred if preds is None else np.append(preds, pred, axis=0)
    
    preds = torch.from_numpy(preds)
    self.set_dataset(
      TensorDataset(*(dataset.tensors[:-1] + (preds, ))), ds_key)

  def update_concat_dataset_cache(self, ds_keys, key_prefix="concat"):
    assert all(ds_key[1:] == ds_keys[0][1:] for ds_key in ds_keys)
    new_split = "-".join((key_prefix,) + tuple(ds_key[0] for ds_key in ds_keys))
    logger.info("Concating %d dataset %s ..." % (len(ds_keys), new_split))
    new_ds_key = (new_split, ) + ds_keys[0][1:]
    new_ds = ConcatDataset([self.get_dataset(*ds_key) for ds_key in ds_keys])
    self.set_dataset(new_ds, new_ds_key)
    logger.info("Construct new dataset %s with %d examples" % (
      str(new_ds_key), len(new_ds)))
    return new_ds_key


class XClassificationTrainer(Trainer):

  def __init__(self, args, model, tokenizer):
    super().__init__(args, model, tokenizer)
    _, self.optimizer, self.scheduler = self.init_optimizer(
      model, args.learning_rate)
    self.example_feature_cache = {}
    self.no_improve_cnt = 0
  
  def init_optimizer(self, model, lr):
    args = self.args
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
      {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], "weight_decay": 0.0}]
    # TODO calculate t_total
    optimizer = AdamW(
      optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #   optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_constant_schedule(optimizer)
    return optimizer_grouped_parameters, optimizer, scheduler

  def run(self):
    args = self.args
    set_seed(args)

    if self.optimizer is not None and args.fp16:
      self.model, self.optimizer = amp.initialize(
        self.model, self.optimizer, opt_level=args.fp16_opt_level)
    
    test_langs = args.test_langs.split(",")
    assert args.dev_mode in ["train_lang", "test_lang", "avg"]
    if args.dev_mode == "train_lang":
      assert args.train_lang in test_langs
      train_lang_index = test_langs.index(args.train_lang)

    logger.info("***** Running Trainer *****")

    logger.info("***** Before Trainer Loop *****")
    self.before_loop()

    def _eval(update_no_improve_cnt=False):
      score_tups = []
      should_save = False

      for lang in test_langs:
        dev_score_dict = self.eval_epoch(
          split="dev", lang=lang, epoch_id=epoch_id)
        test_score_dict = self.eval_epoch(
          split="test", lang=lang, epoch_id=epoch_id)
        score_tup = (dev_score_dict, test_score_dict)
        score_tups.append(score_tup)
        logger.info("Eval epoch %d - lang - %s score - dev: %s - test: %s" % (
          epoch_id, lang, score_dict_to_string(dev_score_dict),
          score_dict_to_string(test_score_dict)))

      dev_scores, test_scores = [], []
      if args.dev_mode == "test_lang":
        # select best results w.r.t. the res on test-lang dev sets.
        for lang, score_tup in zip(test_langs, score_tups):
          if lang not in self.best_scores:
            if lang == test_langs[-1]: should_save = True
            self.best_scores[lang] = score_tup
          elif self.best_scores[lang][0][args.dev_criterion] < \
            score_tup[0][args.dev_criterion]:
            if lang == test_langs[-1]: should_save = True
            self.best_scores[lang] = score_tup
          dev_scores.append(self.best_scores[lang][0])
          test_scores.append(self.best_scores[lang][1])
      elif args.dev_mode == "train_lang":
        # select best results w.r.t. the res on train-lang dev sets.
        if (args.train_lang not in self.best_scores) or self.best_scores[
          args.train_lang][0][args.dev_criterion] < \
          score_tups[train_lang_index][0][args.dev_criterion]:
          should_save = True
          for lang, score_tup in zip(test_langs, score_tups):
            self.best_scores[lang] = score_tup
            dev_scores.append(self.best_scores[lang][0])
            test_scores.append(self.best_scores[lang][1])
          if update_no_improve_cnt:
            self.no_improve_cnt = 0
            logger.info("New best results!")
        else:
          for lang in test_langs:
            dev_scores.append(self.best_scores[lang][0])
            test_scores.append(self.best_scores[lang][1])
          if update_no_improve_cnt:
            self.no_improve_cnt += 1
            logger.info("Results not improved, no_improve_cnt:%d" % self.no_improve_cnt)
      elif args.dev_mode == "avg":
         # select best results by the best sum scores
        avg_key = "_avg"
        sum_dev_scores = sum_test_scores = 0
        for score_tup in score_tups:
          sum_dev_scores += score_tup[0][args.dev_criterion]
          sum_test_scores += score_tup[1][args.dev_criterion]
        if (avg_key not in self.best_scores) or self.best_scores[avg_key] < sum_dev_scores:
          should_save = True
          self.best_scores[avg_key] = sum_dev_scores
          for lang, score_tup in zip(test_langs, score_tups):
            self.best_scores[lang] = score_tup
            dev_scores.append(self.best_scores[lang][0])
            test_scores.append(self.best_scores[lang][1])
          logger.info("New best results! Dev avg: %.2f Test avg: %.2f" % (
            sum_dev_scores/len(test_langs), sum_test_scores/len(test_langs),
          ))
          if update_no_improve_cnt:
            self.no_improve_cnt = 0
        else:
          for lang in test_langs:
            dev_scores.append(self.best_scores[lang][0])
            test_scores.append(self.best_scores[lang][1])
          if update_no_improve_cnt:
            self.no_improve_cnt += 1
            logger.info("Results not improved, no_improve_cnt:%d" % self.no_improve_cnt)
      
      logger.info("Eval epoch %d - langs %s - dev scores - %s" % (
        epoch_id, " & ".join(test_langs), score_dicts_to_latex(dev_scores)))
      logger.info("Eval epoch %d - langs %s - test scores - %s" % (
        epoch_id, " & ".join(test_langs), score_dicts_to_latex(test_scores)))
      
      with open(os.path.join(args.exp_results_dir, args.exp_name), "w") as fp:
        json.dump(self.best_scores, fp)
        fp.flush()
      if should_save and args.save:
        save_to = os.path.join(args.dump_path, "best-%s-%s" % (
          args.dev_criterion, args.model_type))
        logger.info("Epoch %d, saving best model to %s" % (
          epoch_id, save_to))
        torch.save(self.model.state_dict(), save_to)


    logger.info("***** Start Trainer Loop *****")
    for epoch_id in range(args.num_train_epochs):
      self.train_full_epoch(args.train_ds_keys, epoch_id=epoch_id, algo=args.algo)
      _eval(update_no_improve_cnt=args.stopping_threshold>0)
      if args.stopping_threshold > 0 and self.no_improve_cnt >= args.stopping_threshold:
        logger.info("***** Early stop *****")
        break

    if args.add_train_ds_keys != "":
      logger.info("***** Additional Trainer Loop *****")
      state_dict_path = os.path.join(args.dump_path, "best-%s-%s" % (
        args.dev_criterion, args.model_type))
      logger.info("Reloading model parameters from %s ..." % state_dict_path)
      state_dict = torch.load(state_dict_path, map_location="cpu")
      self.model.load_state_dict(state_dict)
      self.model.cuda()
      num_additional_train_epochs = getattr(
        args, "num_additional_train_epochs", args.num_train_epochs)
      for epoch_id in range(num_additional_train_epochs):
        self.train_full_epoch(
          args.add_train_ds_keys, epoch_id=epoch_id, algo=args.add_algo)
        _eval()
  
  def train_full_epoch(self, train_ds_keys, epoch_id, algo=None):
    raise NotImplementedError

  def train_full_epoch_deprecated(self, split, lang, epoch_id):
    logger.info("***** Training epoch %d - lang: %s *****" % (epoch_id, lang))
    args = self.args
    model = self.model
    model.train()
    losses = []

    n_instances = 0
    for step, batch in enumerate(self.get_dataloader(split, lang)):
      feed_dict = self._parse_batch(batch)
      outputs = model(**feed_dict)
      loss = outputs[0]
      model.zero_grad()

      if args.fp16:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
          scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(
          amp.master_params(self.optimizer), args.max_grad_norm)
      else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
      
      self.optimizer.step()
      self.scheduler.step()

      losses.append(loss)
      n_instances += args.train_batch_size
      if step % args.logging_steps == 0:
        logger.info(
          "Epoch %d - n instances %7d - loss: %.4f " % (
          epoch_id, n_instances, sum(losses) / len(losses)))
        losses = []
  
  def eval_epoch(self, split, lang, epoch_id):
    logger.info("***** Evaluating epoch %d - split: %s - lang: %s*****" % (
      epoch_id, split, lang))
    
    def _get_batch_iter():
      for batch in self.get_dataloader(split, lang, shuffle=False):
        yield self._parse_batch(batch)
    
    return eval_classification(self.model, _get_batch_iter())
  
  def load_and_cache_examples(self, split, lang, **kwargs):
    processor = xdoc.get_processor_class(self.args.dataset_name)()
    cache_key = self.get_cache_key()
    return xdoc.load_and_cache_examples(
      self.args, processor, split, lang, self.tokenizer, cache_key)


class XQATrainer(XClassificationTrainer):

  def __init__(self, args, model, tokenizer):
    super().__init__(args, model, tokenizer)
  
  def init_optimizer(self, model, lr):
    args = self.args
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {"params": [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
      {"params": [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(
      optimizer_grouped_parameters, lr=lr, eps=args.adam_epsilon)
    dataloader = self.get_dataloader("train", args.train_lang)
    t_total = len(dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
      optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer_grouped_parameters, optimizer, scheduler

  def _parse_batch(self, batch, training=True, **kwargs):
    _batch = to_cuda(batch)
    # _batch = batch
    if training:
      ret = {"input_ids": _batch[0],
        "attention_mask": _batch[1],
        "token_type_ids": _batch[2] if self.args.model_type == "bert" else None,
        'start_positions': _batch[3],
        'end_positions':   _batch[4]}
    else:
      ret = {"input_ids": _batch[0],
        "attention_mask": _batch[1],
        "token_type_ids": _batch[2] if self.args.model_type == "bert" else None}
        
    ret.update(**kwargs)
    return ret
  
  def eval_epoch(self, split, lang, epoch_id):
    args = self.args
    logger.info("***** Evaluating epoch %d - split: %s - lang: %s*****" % (
      epoch_id, split, lang))
    dataset, examples, features = self.get_eval_data(split, lang)

    def _get_batch_iter():
      for batch in self.get_dataloader(
        split, lang, shuffle=False, dataset=dataset):
        example_indices = batch[3]
        yield self._parse_batch(batch, training=False), example_indices

    return eval_qa(self.model, _get_batch_iter(), **{
      "all_examples": examples,
      "all_features": features,
      "predict_file": os.path.join(args.data_dir, "%s-%s.json" % (split, lang)),
      "output_dir": args.dump_path,
      "n_best_size": args.n_best_size,
      "max_answer_length": args.max_answer_length,
      "do_lower_case": args.do_lower_case,
      "verbose_logging": args.verbose_logging,
      "version_2_with_negative": args.version_2_with_negative,
      "null_score_diff_threshold": args.null_score_diff_threshold})
  
  def load_and_cache_examples(self, split, lang, **kwargs):
    evaluate = kwargs.pop("evaluate", False)
    cache_key = "%s-%s" % (self.args.model_key, self.args.model_type)
    dataset, _, _ =  xqa.load_and_cache_examples(
      self.args, split, lang, self.tokenizer, cache_key, evaluate=evaluate)
    return dataset
  
  def get_eval_data(self, split, lang):
    ds_key = (split, lang)
    if ds_key in self.example_feature_cache:
      return self.example_feature_cache[ds_key]
    cache_key = "%s-%s" % (self.args.model_key, self.args.model_type)
    dataset, examples, features =  xqa.load_and_cache_examples(
      self.args, split, lang, self.tokenizer, cache_key, evaluate=True)
    self.example_feature_cache[ds_key] = (dataset, examples, features)
    return dataset, examples, features
