import torch

from fairseq.data import FairseqDataset


class TLMDataset(FairseqDataset):

  def __init__(self, src_dataset, tgt_dataset, bos, eos):
    assert len(src_dataset) == len(tgt_dataset)
    self.src_dataset = src_dataset
    self.tgt_dataset = tgt_dataset
    self.bos = bos
    self.eos = eos
    self._sizes = src_dataset.sizes + tgt_dataset.sizes

  def __len__(self):
    return len(self.src_dataset)
  
  @property
  def sizes(self):
    return self._sizes

  def __getitem__(self, index):
    src_item = self.src_dataset[index]
    tgt_item = self.tgt_dataset[index]
    return torch.cat([
      src_item.new([self.bos]), src_item, src_item.new([self.eos]),
      tgt_item, tgt_item.new([self.eos]),
    ])
  