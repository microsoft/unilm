import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils
from fairseq import utils

from fairseq.models import (
  BaseFairseqModel,
  register_model,
  register_model_architecture,
)
from fairseq.models.roberta import (
  RobertaModel,
  RobertaEncoder,
  roberta_base_architecture,
  roberta_large_architecture,
)


@register_model("reload_roberta")
class ReloadRoberta(RobertaModel):

  @staticmethod
  def add_args(parser):
    RobertaModel.add_args(parser)
    parser.add_argument('--roberta-model-path', type=str, default="")
  
  @classmethod
  def build_model(cls, args, task):
    reload_roberta_base(args)
    if not hasattr(args, 'max_positions'):
      args.max_positions = args.tokens_per_sample

    encoder = RobertaEncoder(args, task.source_dictionary)
    model = cls(args, encoder)

    if args.roberta_model_path != "":
      state = checkpoint_utils.load_checkpoint_to_cpu(args.roberta_model_path)
      model.load_state_dict(state["model"], strict=True, args=args)
    
    print(model.__class__)
    return model
  
  @classmethod
  def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='sentencepiece', **kwargs):
    raise NotImplementedError

  # NOTE WALKAROUND  `size` method of dataset classes
  # examples are filtered during preprocessing
  # so we do not need to filter once again
  def max_positions(self):
    """Maximum length supported by the model."""
    return None


@register_model_architecture("reload_roberta", "reload_roberta_base")
def reload_roberta_base(args):
  roberta_base_architecture(args)


@register_model_architecture("reload_roberta", "reload_roberta_large")
def reload_roberta_large(args):
  roberta_large_architecture(args)
