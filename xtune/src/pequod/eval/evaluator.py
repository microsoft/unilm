import logging
import torch

from torch.utils.data import DataLoader
from src.pequod.training.trainer import to_cuda


logger = logging.getLogger(__name__)


class Evaluator(object):

  def __init__(self, args, model, tokenizer, **kwargs):
    self.args = args
    self.datasets = {}
    self.model = model
    self.tokenizer = tokenizer
  
  def _parse_batch(self, batch, has_label=True, **kwargs):
    _batch = to_cuda(batch)
    # _batch = batch
    ret = {"input_ids": _batch[0],
      "attention_mask": _batch[1],
      "token_type_ids": _batch[2] if self.args.model_type == "bert" else None,}
    if has_label: ret["labels"] = _batch[3]
    ret.update(**kwargs)
    return ret
  
  def run(self):
    raise NotImplementedError

  def get_dataset(self, *args, **kwargs):
    if args in self.datasets: return self.datasets[args]
    dataset = self.load_and_cache_examples(*args, **kwargs)
    self.datasets[args] = dataset
    return dataset
  
  def load_and_cache_examples(self, *args, **kwargs):
    raise NotImplementedError

  def get_dataloader(self, *args, **kwargs):
    logger.info("Getting dataloader - args: %s" % str(args))
    dataset = kwargs.pop("dataset", self.get_dataset(*args, **kwargs))
    dataloader = DataLoader(dataset, batch_size=self.args.eval_batch_size)
    return dataloader
