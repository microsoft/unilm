import json
import os
from dataclasses import dataclass, field
from typing import Optional

import torch

from fairseq.model_parallel.megatron.mpu import (
    initialize_model_parallel,
    model_parallel_is_initialized,
    get_model_parallel_rank
)

from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)

from omegaconf import II

from .decoder.transformer import ModelArgs, Transformer

DEFAULT_MAX_TARGET_POSITIONS = 4096

@dataclass
class LanguageConfig(FairseqDataclass):
    llama_model: Optional[str] = field(
        default=None,
        metadata={"help": "path to load tokenizer and config"},
    )
    load_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "path to load checkpoint from"},
    )
    init_from_config: bool = field(
        default=False,
    )
    dim: int = field(
        default=1024,
    )
    n_layers: int = field(
        default=8,
    )
    n_heads: int = field(
        default=8,
    )
    n_kv_heads: int = field(
        default=2,
    )
    batch_size: int = field(
        default=1,
    )
    rope_theta: Optional[float] = field(
        default=10000.0,
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    tokens_per_sample: int = II("task.tokens_per_sample")
    model_parallel_size: int = II("common.model_parallel_size")

@register_model("llama", dataclass=LanguageConfig)
class LanguageModel(FairseqLanguageModel):
    def __init__(self, args, decoder, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        if not model_parallel_is_initialized():
            initialize_model_parallel(args.model_parallel_size)
            
        if not args.init_from_config:
            params = {
                "dim": args.dim,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "head_dim": args.dim // args.n_heads,
                "n_kv_heads": args.n_kv_heads,
                "hidden_dim": int(args.dim * 8 / 3),
                "vocab_size": task.tokenizer.n_words,
                "max_batch_size": args.batch_size,
                "max_seq_len": args.tokens_per_sample,
                "model_parallel_size": args.model_parallel_size,
                "load_checkpoint": args.load_ckpt is not None,
                "rope_theta": args.rope_theta,
            }
            model_args: ModelArgs = ModelArgs(
                **params,
            )
        else:
            with open(os.path.join(args.llama_model, "params.json"), "r") as f:
                params = json.load(f)
                model_args = ModelArgs(**params)
            model_args.max_batch_size = args.batch_size
            model_args.max_seq_len = args.tokens_per_sample
            model_args.model_parallel_size = args.model_parallel_size
            model_args.load_checkpoint = args.load_ckpt is not None
        model = Transformer(
            model_args,
            mp_rank=get_model_parallel_rank(),
            checkpoint_activations=args.checkpoint_activations,
        )
        if args.load_ckpt is not None:
            loaded = torch.load(args.load_ckpt, mmap=True)
            model.load_state_dict(loaded, assign=True)
            
        model = LLaMA(model)
        return cls(args, model, task.tokenizer)

class LLaMA(FairseqIncrementalDecoder):
    def __init__(self, model):
        super().__init__(None)
        self.model = model
        
    def forward(self, src_tokens, start_pos = 0, **kwargs):
        padding = src_tokens < 0
        src_tokens = torch.where(padding, torch.zeros_like(src_tokens), src_tokens)
        return self.model.forward(src_tokens, start_pos, **kwargs)

    def max_positions(self):
        return self.model.args.max_seq_len
                
@register_model_architecture("llama", "llama_from_scratch")
def llama_from_scratch(args):
    args.init_from_config = getattr(args, "init_from_config", False)
    args.dim = getattr(args, "dim", 1024)
    args.n_layers = getattr(args, "n_layers", 8)
    args.n_heads = getattr(args, "n_heads", 8)
    args.n_kv_heads = getattr(args, "n_kv_heads", 2)

@register_model_architecture("llama", "llama_from_ckpt")
def llama_from_ckpt(args):
    args.init_from_config = getattr(args, "init_from_config", True)

    
    