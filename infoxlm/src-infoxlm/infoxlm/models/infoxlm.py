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
  roberta_base_architecture,
  roberta_large_architecture
)

from infoxlm.utils import concat_all_gather


def build_projection_dict(langs, dim, activation_fn, fp16=False):
  proj_dict = {}
  cnt = 0
  for lang in langs:
    proj_dict[lang] = cnt
    cnt += 1
  proj_matrix_slow = torch.randn(cnt, dim, dim)
  proj_matrix_slow.normal_(mean=0.0, std=0.02)
  proj_matrix_slow = nn.Parameter(proj_matrix_slow, requires_grad=False)
  proj_matrix_fast = nn.Parameter(proj_matrix_slow.data.clone(), requires_grad=True)
  return proj_dict, proj_matrix_fast, proj_matrix_slow


@register_model("infoxlm")
class InfoXlmModel(BaseFairseqModel):

  def __init__(self, model_fast, model_slow, queue, proj=None):
    super().__init__()
    self.model_slow:nn.Module = model_slow
    self.model_fast:nn.Module = model_fast
    self.use_proj = False
    self.share_proj = True
    self.queue_size = queue.size(0)
    self.register_buffer("queue", queue)
    self.register_buffer("enqueue_cnt", torch.zeros(1, dtype=torch.long))
    self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    if proj is not None:
      self.use_proj = True
      self.proj_dict, proj_matrix_fast, proj_matrix_slow = proj
      # if "share_lang" in self.proj_dict: self.share_proj = True
      assert "share_lang" in self.proj_dict
      self.register_parameter("proj_matrix_fast", proj_matrix_fast)
      self.register_parameter("proj_matrix_slow", proj_matrix_slow)

    for param in self.model_slow.parameters():
      param.requires_grad = False
    
  @staticmethod
  def add_args(parser):
    parser.add_argument('--roberta-model-path', type=str, default="")
    parser.add_argument('--dropout', type=float, metavar='D',
                        help='dropout probability')
    parser.add_argument('--attention-dropout', type=float, metavar='D',
                        help='dropout probability for attention weights')
    parser.add_argument('--max-positions', type=int,
                        help='number of positional embeddings to learn')
    parser.add_argument('--activation-dropout', type=float, metavar='D',
                        help='dropout probability after activation in FFN')
    parser.add_argument('--use_proj', default=False, action='store_true')
  
  def is_queue_ready(self):
    return int(self.enqueue_cnt) >= self.queue_size
  
  @torch.no_grad()
  def update_queue(self, k):
    k = concat_all_gather(k)
    batch_size = k.size(0)
    ptr = int(self.queue_ptr)
    # assert self.queue_size % batch_size == 0
    if ptr + batch_size <= self.queue_size:
      self.queue[ptr:ptr+batch_size, :] = k
      ptr = (ptr + batch_size) % self.queue_size
    else:
      left_len = self.queue_size - ptr
      self.queue[ptr:, :] = k[:left_len, :]
      ptr = batch_size-left_len
      self.queue[:ptr, :] = k[left_len:, :]
    self.queue_ptr[0] = ptr
    self.enqueue_cnt += batch_size
  
  @classmethod
  def build_model(cls, args, task):

    model_fast = RobertaModel.build_model(args, task)
    model_slow = RobertaModel.build_model(args, task)

    if args.roberta_model_path != "":
      state = checkpoint_utils.load_checkpoint_to_cpu(args.roberta_model_path)
      model_fast.load_state_dict(state["model"], strict=True, args=args)
      model_slow.load_state_dict(state["model"], strict=True, args=args)
    else:
      model_slow.load_state_dict(model_fast.state_dict(), strict=True, args=args)

    proj = None
    if args.use_proj:
      # NOTE alway be share_proj
      langs = ["share_lang"]
      proj = build_projection_dict(langs, args.encoder_embed_dim, args.activation_fn, args.fp16)

    if "xlco_queue_size" in args:
      xlco_queue_size = args.xlco_queue_size
    else: xlco_queue_size = 1
    print("xlco_queue_size is set as %d" % xlco_queue_size, flush=True)
    queue = torch.randn(xlco_queue_size, args.encoder_embed_dim)

    return cls(model_fast, model_slow, queue, proj=proj)
  
  @classmethod
  def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt',
    data_name_or_path='.', bpe='sentencepiece', **kwargs):
    raise NotImplementedError
  
  def forward(self, src_tokens, use_model_fast=True, **kwargs):    
    forward_model = self.model_fast if use_model_fast else self.model_slow
    return forward_model(src_tokens, **kwargs)
  
  def forward_proj(self, rep, lang, use_model_fast=True, **kwargs):
    proj_matrix = self.proj_matrix_fast if use_model_fast else self.proj_matrix_slow
    
    if self.share_proj: lang = "share_lang"
    if isinstance(lang, str):
      return torch.mm(rep, proj_matrix[self.proj_dict[lang],:,:])
    else:
      proj_indices = [self.proj_dict[l] for l in lang]
      batch_rep = rep.unsqueeze(1)
      return torch.bmm(batch_rep, proj_matrix[proj_indices,:,:])[:,0,:]
  
  def output_layer(self, features, use_model_fast=True, **kwargs):
    forward_model = self.model_fast if use_model_fast else self.model_slow
    return forward_model.decoder.output_layer(features, **kwargs)
  
  @torch.no_grad()
  def update_slow_weight(self, momentum):
    for p1, p2 in zip(self.model_fast.parameters(), self.model_slow.parameters()):
      assert p2.requires_grad == False
      new_p2_data = p2.data * momentum + p1.data * (1. - momentum)
      p2.data.copy_(new_p2_data)

    if self.use_proj:
      p1 = self.proj_matrix_fast.data
      p2 = self.proj_matrix_slow.data
      assert p2.requires_grad == False
      new_p2_data = p2.data * momentum + p1.data * (1. - momentum)
      p2.data.copy_(new_p2_data)


@register_model_architecture("infoxlm", "infoxlm_base")
def infoxlm_base(args):
  roberta_base_architecture(args)

@register_model_architecture("infoxlm", "infoxlm_large")
def infoxlm_large(args):
  roberta_large_architecture(args)
