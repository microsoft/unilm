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
  roberta_base_architecture,
  roberta_large_architecture
)
from fairseq.modules import LayerNorm

from infoxlm.models.roberta import ReloadRoberta, reload_roberta_base, RobertaEncoder


@register_model("xlm_align")
class XlmAlignModel(ReloadRoberta):

  @staticmethod
  def add_args(parser):
    ReloadRoberta.add_args(parser)
    parser.add_argument('--no_linear_proj', default=False, action='store_true')

  def __init__(self, args, encoder):
    super().__init__(args, encoder)
    if args.no_linear_proj:
      self.q_linear = self.k_linear = lambda x: x
    else:
      self.q_linear = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim,)
      self.k_linear = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim,)
  
  @classmethod
  def build_model(cls, args, task):
    reload_roberta_base(args)
    if not hasattr(args, 'max_positions'):
      args.max_positions = args.tokens_per_sample

    encoder = RobertaEncoder(args, task.source_dictionary)
    model = cls(args, encoder)

    if args.roberta_model_path != "":
      state = checkpoint_utils.load_checkpoint_to_cpu(args.roberta_model_path)
      model.load_state_dict(state["model"], strict=False, args=args)
    
    print(model.__class__)
    return model


@register_model_architecture("xlm_align", "xlm_align_base")
def xlm_align_base(args):
  roberta_base_architecture(args)


@register_model_architecture("xlm_align", "xlm_align_large")
def xlm_align_large(args):
  roberta_large_architecture(args)
