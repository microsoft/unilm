import os

from fairseq.tasks import register_task, FairseqTask
from fairseq.data.dictionary import Dictionary
from infoxlm.data import mlm_utils
from infoxlm.data.dict_dataset import DictDataset
from infoxlm.tasks.mlm import Mlm


@register_task("tlm")
class Tlm(Mlm):

  @staticmethod
  def add_args(parser):
    Mlm.add_args(parser)
    parser.add_argument('--tlm_data', type=str, default="")
  
  def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
    model.train()
    agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

    # tlm step
    loss, sample_size, logging_output = criterion(model, sample["tlm"])
    if ignore_grad: loss *= 0
    tlm_loss = loss
    optimizer.backward(tlm_loss)
    agg_loss += tlm_loss.detach().item()
    agg_sample_size += sample_size
    agg_logging_output.update(logging_output)

    # mlm_step
    loss, sample_size, logging_output = criterion(model, sample["mlm"])
    if ignore_grad: loss *= 0
    optimizer.backward(loss)
    agg_loss += loss.detach().item()
    agg_sample_size += sample_size
    for key, value in logging_output.items():
      agg_logging_output[key] += value

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
    dataset = DictDataset({
      "tlm": tlm_dataset,
      "mlm": mlm_dataset,
    })
    self.datasets[split] = dataset
    
