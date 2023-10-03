# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm as LayerNorm
from fairseq import utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.models.squad import SQuADHead
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
from fairseq.modules import PositionalEmbedding
from omegaconf import II

from torchscale.architecture.config import EncoderConfig

from .machine_translation import MTEncoder as Encoder

DEFAULT_MAX_SOURCE_POSITIONS = 1024

logger = logging.getLogger(__name__)


@dataclass
class BertConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    encoder_embed_dim: int = field(
        default=512, metadata={"help": "encoder embedding dimension"}
    )
    encoder_output_dim: int = field(
        default=512, metadata={"help": "encoder output dimension"}
    )
    encoder_input_dim: int = field(
        default=512, metadata={"help": "encoder input dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(default=6, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    no_encoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last encoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_encoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share encoder input and output embeddings"}
    )
    encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the encoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for encoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        },
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max source positions"}
    )
    pooler_activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use for pooler layer"}
    )
    pooler_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability in the masked_lm pooler layers"},
    )
    # options from other parts of the config
    # add_bos_token: bool = II("task.add_bos_token")
    # tokens_per_sample: int = II("task.tokens_per_sample")
    tpu: bool = II("common.tpu")
    rel_pos_buckets: int = field(default=0, metadata={"help": ""})
    max_rel_pos: int = field(default=0, metadata={"help": ""})
    moe_freq: int = field(
        default=0,
        metadata={"help": "Frequency at which we insert MoE Transformer layers"},
    )
    moe_expert_count: int = field(
        default=0, metadata={"help": "Number of experts in each MoE Layer"}
    )
    moe_gating_use_fp32: bool = field(
        default=False,
        metadata={"help": "Use FP32 computations in MoE top2 gating function"},
    )
    moe_second_expert_policy: str = field(
        default="sampling",
        metadata={"help": "policy for second expert, options: all/sampling/random"},
    )
    moe_normalize_gate_prob_before_dropping: bool = field(
        default=False,
        metadata={
            "help": "whether to normalize gate probs before or after dropping experts for capacity and randomization"
        },
    )
    moe_expert_ffn_dim: Optional[int] = field(
        default=None, metadata={"help": "MoE expert FFN dimension"}
    )
    moe_top1_expert: Optional[bool] = field(
        default=False, metadata={"help": "Use top1 gate instead of top2"}
    )
    moe_eval_capacity_token_fraction: Optional[float] = field(
        default=0.25,
        metadata={
            "help": (
                "Default: 0.25, Fraction of tokens as capacity during validation, "
                "if set to negative, use same as training. range: (0.0, 1.0]."
            )
        },
    )
    moe_normalize_expert_grad: Optional[str] = field(
        default="world_size",
        metadata={
            "help": "Divide expert gradients by (1) 'world_size' (2) 'sqrt_world_size'"
        },
    )
    record_a2a_perf_stats: Optional[bool] = field(
        default=False,
        metadata={"help": "records all to all perf stats during distributed training"},
    )
    dummy_a2a: Optional[bool] = field(
        default=False,
        metadata={
            "help": "By passes all to all during distributed training by returning the input buffer as output"
        },
    )
    moe_batch_prioritized_routing: Optional[bool] = field(
        default=False,
        metadata={
            "help": "if true orders token by the gate prob before capacity dropping."
        },
    )
    ddp_rank: int = II("distributed_training.distributed_rank")
    deepnorm: Optional[bool] = field(
        default=False,
    )
    subln: Optional[bool] = field(
        default=False,
    )


@register_model("mlm", dataclass=BertConfig)
class BertModel(BaseFairseqModel):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.padding_idx = self.encoder.embed_tokens.padding_idx
        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        args.max_source_positions = getattr(
            args, "max_source_positions", DEFAULT_MAX_SOURCE_POSITIONS
        )

        embed_tokens = cls.build_embedding(
            args, task.dictionary, args.encoder_embed_dim
        )

        embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                args.encoder_embed_dim,
                task.dictionary.pad(),
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        lm_head = cls.build_lm_head(
            args,
            args.encoder_embed_dim,
            len(task.dictionary),
            args.activation_fn,
            weight=embed_tokens.weight,
        )

        config = EncoderConfig()
        config.override(args)

        encoder = Encoder(
            config,
            embed_tokens=embed_tokens,
            embed_positions=embed_positions,
            output_projection=lm_head,
            is_encoder_decoder=False,
            dictionary=task.dictionary,
        )

        return cls(args, encoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens

    @classmethod
    def build_lm_head(cls, args, embed_dim, output_dim, activation_fn, weight):
        return LMHead(embed_dim, output_dim, activation_fn, weight)

    def output_layer(self, features, masked_tokens=None):
        return self.encoder.output_projection(features, masked_tokens=masked_tokens)

    def register_classification_head(
        self, name, num_classes=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    def register_question_answering_head(self, name, num_classes=None):
        self.classification_heads[name] = SQuADHead(
            self.args.encoder_embed_dim,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]  # noqa: E203
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
            ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
            ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes
                    != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim
                    != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

    def forward(
        self,
        src_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        masked_tokens=None,
        **kwargs
    ):
        encoder_out = self.encoder(
            src_tokens, features_only=True, return_all_hiddens=return_all_hiddens
        )
        x, extra = encoder_out["encoder_out"], encoder_out
        x = x.transpose(0, 1)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        elif not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)

        return x, extra


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


@register_model_architecture("mlm", "mlm_base")
def base_unilm_architecture(args):
    if hasattr(args, "encoder_final_norm"):
        args.no_encoder_final_norm = not args.encoder_final_norm

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")

    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)

    # args.add_bos_token = getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", True
    )
    args.encoder_output_dim = getattr(
        args, "encoder_output_dim", args.encoder_embed_dim
    )
    args.encoder_input_dim = getattr(args, "encoder_input_dim", args.encoder_embed_dim)

    # Model training is not stable without this
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.no_encoder_final_norm = getattr(args, "no_encoder_final_norm", False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
