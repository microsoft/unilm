import os

from fairseq.tasks import register_task, FairseqTask
from fairseq.data.dictionary import Dictionary
from infoxlm.data import mlm_utils


@register_task("mlm")
class Mlm(FairseqTask):

  @staticmethod
  def add_args(parser):
    mlm_utils.add_mlm_args(parser)
    parser.add_argument('data', help='colon separated path to data directories list, '
                        'will be iterated upon during epochs in round-robin manner')
    parser.add_argument('--tokens-per-sample', default=512, type=int,
                        help='max number of total tokens over all segments per sample')
    # apply prepend bos + tokenblock
    parser.add_argument('--apply_ptb', default=False, action='store_true')
  
  @classmethod
  def setup_task(cls, args, **kwargs):
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    print('| Dictionary: {} types'.format(len(dictionary)), flush=True)
    return cls(args, dictionary)
  
  def __init__(self, args, dictionary):
    super().__init__(args)
    self.dictionary = dictionary
    self.mask_idx = self.dictionary.add_symbol('<mask>')
    self.seed = args.seed
    self.mww = self._get_whole_word_mask()

  def _get_whole_word_mask(self):
    # create masked input and targets
    if self.args.mask_whole_words:
      print("| Get whole work mask ...")
      return mlm_utils.get_whole_word_mask(self.args, self.dictionary)
    return None
  
  def load_dataset(self, split, epoch=0, combine=False, **kwargs):
    print("| Loading dataset at epoch %d" % epoch, flush=True)
    args = self.args
    sid = 0
    dataset_path = os.path.join(args.data, "train.%d" % sid)
    dataset = mlm_utils.get_mlm_dataset(
      args, dataset_path, self.dictionary, self.mask_idx, self.mww, combine=False)
    self.datasets[split] = dataset
  
  @property
  def source_dictionary(self):
    return self.dictionary

  @property
  def target_dictionary(self):
    return self.dictionary
  