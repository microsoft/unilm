import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from fairseq import distributed_utils, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from omegaconf import II

from fairseq.model_parallel.megatron.mpu import (
    initialize_model_parallel,
    model_parallel_is_initialized
)
from .decoder.yoco import YOCO, YOCOArgs

DEFAULT_MAX_TARGET_POSITIONS = 4096
logger = logging.getLogger(__name__)


@dataclass
class LanguageConfig(FairseqDataclass):
    yoco_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to load params from"},
    )
    load_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "path to load checkpoint from"},
    )
    dim: int = field(
        default=1024,
    )
    hidden_dim: int = field(
        default=3072,
    )
    n_layers: int = field(
        default=24,
    )
    n_self_heads: int = field(
        default=4,
    )
    n_attn_heads: int = field(
        default=8,
    )
    n_attn_kv_heads: Optional[int] = field(
        default=None,
    )
    batch_size: int = field(
        default=1,
    )
    share_input_output_embed: bool = field(
        default=False, metadata={"help": "share decoder input and output embeddings"}
    )
    sliding_window: Optional[bool] = field(
        default=None,
    )
    rope_theta: Optional[float] = field(
        default=10000.0,
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    tokens_per_sample: int = II("task.tokens_per_sample")
    model_parallel_size: int = II("common.model_parallel_size")
    

@register_model("yoco", dataclass=LanguageConfig)
class LanguageModel(FairseqLanguageModel):
    def __init__(self, args, decoder, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        if not model_parallel_is_initialized():
            initialize_model_parallel(args.model_parallel_size)
            
        if args.yoco_model is None:
            params = {
                "dim": args.dim,
                "n_layers": args.n_layers,
                "n_self_heads": args.n_self_heads,
                "n_attn_heads": args.n_attn_heads,
                "n_attn_kv_heads": args.n_attn_kv_heads,
                "hidden_dim": args.hidden_dim,
                "vocab_size": task.tokenizer.n_words,
                "max_batch_size": args.batch_size,
                "max_seq_len": args.tokens_per_sample,
                "model_parallel_size": args.model_parallel_size,
                "load_checkpoint": args.load_ckpt is not None,
                "rope_theta": args.rope_theta,
            }
            model_args: YOCOArgs = YOCOArgs(
                **params,
            )
        else:
            with open(os.path.join(args.yoco_model, "params.json"), "r") as f:
                params = json.load(f)
                model_args = YOCOArgs(**params)
            model_args.max_batch_size = args.batch_size
            model_args.max_seq_len = args.tokens_per_sample
            model_args.model_parallel_size = args.model_parallel_size
            model_args.load_checkpoint = args.load_ckpt is not None
            
        model = YOCO(
            model_args,
            checkpoint_activations=args.checkpoint_activations,
        )
        if args.load_ckpt is not None:
            loaded = torch.load(args.load_ckpt, mmap=True)
            model.load_state_dict(loaded, assign=True)
        model = YOCOModel(model)
        return cls(args, model, task.tokenizer)

class YOCOModel(FairseqIncrementalDecoder):
    def __init__(self, model):
        super().__init__(None)
        self.model = model
        
    def forward(self, src_tokens, **kwargs):
        return self.model.forward(src_tokens, **kwargs)

    def max_positions(self):
        return self.model.args.max_seq_len
                
def default(args):
    args.n_attn_kv_heads = getattr(args, "n_attn_kv_heads", args.n_attn_heads)
    args.sliding_window = getattr(args, "sliding_window", False)
    args.rope_theta = getattr(args, "rope_theta", 10000.0)
    args.share_input_output_embed = getattr(
        args, "share_input_output_embed", False
    )
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)


@register_model_architecture("yoco", "yoco_3b")
def yoco_3b(args):
    args.dim = getattr(args, "dim", 3072)
    args.hidden_dim = getattr(args, "hidden_dim", 8192)
    args.n_layers = getattr(args, "n_layers", 26)
    args.n_self_heads = getattr(args, "n_self_heads", 24)
    args.n_attn_heads = getattr(args, "n_attn_heads", 24)
    args.n_attn_kv_heads = getattr(args, "n_attn_kv_heads", 8)
    default(args)


    

