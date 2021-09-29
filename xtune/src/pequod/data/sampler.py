import torch

from torch.utils.data.sampler import Sampler


class SubSampler(Sampler):

  def __init__(self, data_source, num_samples):
    self.data_source = data_source
    self.num_samples = num_samples

  def __len__(self):
    return self.num_samples
  
  def __iter__(self):
    n = len(self.data_source)
    if self.num_samples <= n:
      return iter(torch.randperm(n).tolist()[:self.num_samples])
    return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())