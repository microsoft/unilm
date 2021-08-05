import numpy as np
import os
import torch

from threading import Thread
from fairseq.data import data_utils, FairseqDataset, FairseqIterableDataset

class DictIterDataset(FairseqIterableDataset):

  def __init__(self, defn, sizes=None):
    self.defn = defn
    for v in self.defn.values():
      if not isinstance(v, (FairseqIterableDataset, )):
        raise ValueError('Expected Dataset but found: {}'.format(v.__class__))
  
  def set_epoch(self, epoch):
    for ds in self.defn.values():
      ds.set_epoch(epoch)
  
  def __iter__(self):
    iters = {key:iter(self.defn[key]) for key in self.defn}

    while True:
      try:
        yield {key:next(iters[key]) for key in iters}
      except StopIteration:
        break
  
  def __len__(self):
    return min(len(v) for v in self.defn.values())

  def collater(self, samples):
    if len(samples) == 0:
      return {}
    sample = {}
    for k, ds in self.defn.items():
      sample[k] = ds.collater([s[k] for s in samples])
    return sample


class DictDataset(FairseqDataset):

  def __init__(self, defn, sizes=None):
    self.defn = defn
    for v in self.defn.values():
      if not isinstance(v, (FairseqDataset, )):
        raise ValueError('Expected Dataset but found: {}'.format(v.__class__))
  
  def set_epoch(self, epoch):
    for ds in self.defn.values():
      ds.set_epoch(epoch)
  
  def __getitem__(self, index):
    ret = {key:self.defn[key][index] for key in self.defn}
    return ret
  
  def __len__(self):
    return min(len(v) for v in self.defn.values())

  def collater(self, samples):
    if len(samples) == 0:
      return {}
    sample = {}
    for k, ds in self.defn.items():
      sample[k] = ds.collater([s[k] for s in samples])

    # DEBUG
    # print(sample)
    return sample

