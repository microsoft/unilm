import copy
import os
from typing import Union, Optional, Callable, List, Tuple

import torch
import torch.utils.checkpoint
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.layers import ColumnParallelLinear as ColumnParallelLinearMP, RowParallelLinear as RowParallelLinearMP, \
    VocabParallelEmbedding
from megatron.core.tensor_parallel.cross_entropy import VocabParallelCrossEntropy
from megatron.core.tensor_parallel.utils import VocabUtility, divide
from megatron.core.models.gpt import GPTModel
from torch import nn
from torch.nn import init
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2FlashAttention2,
    Qwen2SdpaAttention,
    Qwen2MLP,
    Qwen2DecoderLayer,
    Qwen2RMSNorm,
    Qwen2Model,
    is_flash_attn_greater_or_equal_2_10,
    Qwen2ForCausalLM as HfQwen2ForCausalLM,
    Qwen2PreTrainedModel,
    CausalLMOutputWithPast,
)

from general_util.logger import get_child_logger
from models.dpo_utils import llama_last_token_forward_value, llama_dpo_batch_forward, sft_loss_on_logits, llama_last_token_cls_batch_forward
from models.mixin import return_reference_model
from models.megatron_tp_mixin import PretrainedModelParallelPreSplitMixin
from models.utils import DPOModelOutput, RewardModelOutput
from megatron.core.model_parallel_config import ModelParallelConfig

logger = get_child_logger(__name__)

_MEGATRON_MP_CONFIG: ModelParallelConfig


class ColumnParallelLinear(ColumnParallelLinearMP):
    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        return super().forward(input_, weight)[0]


class RowParallelLinear(RowParallelLinearMP):
    def forward(self, input_):
        return super().forward(input_)[0]


def init_megatron_mp_config(*args, **kwargs):
    config = ModelParallelConfig(
        *args,
        tensor_model_parallel_size=mpu.get_tensor_model_parallel_world_size(),
        pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
        **kwargs,
    )
    global _MEGATRON_MP_CONFIG
    _MEGATRON_MP_CONFIG = config


def attention_tp_init(self: Qwen2Attention, config: Qwen2Config):
    self.q_proj = ColumnParallelLinear(
        self.hidden_size,
        self.num_heads * self.head_dim,
        config=_MEGATRON_MP_CONFIG,
        bias=config.attention_bias,
        gather_output=False,
        init_method=lambda x: x
    )
    self.k_proj = ColumnParallelLinear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        config=_MEGATRON_MP_CONFIG,
        bias=False,
        gather_output=False,
        init_method=lambda x: x
    )
    self.v_proj = ColumnParallelLinear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        config=_MEGATRON_MP_CONFIG,
        bias=False,
        gather_output=False,
        init_method=lambda x: x
    )
    self.o_proj = RowParallelLinear(
        self.hidden_size,
        self.hidden_size,
        config=_MEGATRON_MP_CONFIG,
        bias=False,
        input_is_parallel=True,
        init_method=lambda x: x,
        skip_bias_add=False,
    )
    if hasattr(self, "_init_rope"):
        self._init_rope()

    # self.output_size_per_partition = self.q_proj.output_size_per_partition
    self.num_heads = self.num_heads // mpu.get_tensor_model_parallel_world_size()
    self.num_key_value_heads = self.num_key_value_heads // mpu.get_tensor_model_parallel_world_size()
    self.hidden_size = self.hidden_size // mpu.get_tensor_model_parallel_world_size()


class Qwen2AttentionParallel(Qwen2Attention):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        attention_tp_init(self, config)


class Qwen2FlashAttention2Parallel(Qwen2FlashAttention2):
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        attention_tp_init(self, config)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


class Qwen2SdpaAttentionParallel(Qwen2SdpaAttention):
    """
    qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from qwen2Attention.forward
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        attention_tp_init(self, config)


class Qwen2MLPParallel(Qwen2MLP):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            config=_MEGATRON_MP_CONFIG,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            config=_MEGATRON_MP_CONFIG,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            config=_MEGATRON_MP_CONFIG,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
            skip_bias_add=False,
        )


modeling_qwen2.Qwen2Attention = Qwen2AttentionParallel
modeling_qwen2.Qwen2MLP = Qwen2MLPParallel
modeling_qwen2.QWEN2_ATTENTION_CLASSES["eager"] = Qwen2AttentionParallel
modeling_qwen2.QWEN2_ATTENTION_CLASSES["flash_attention_2"] = Qwen2FlashAttention2Parallel
modeling_qwen2.QWEN2_ATTENTION_CLASSES["sdpa"] = Qwen2SdpaAttentionParallel


class Qwen2ModelParallel(Qwen2Model):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            config.vocab_size, mpu.get_tensor_model_parallel_rank(), mpu.get_tensor_model_parallel_world_size()
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            # padding_idx=self.padding_idx if config.pad_token_id != config.eos_token_id else None, # TODO: Not sure if this is correct.
            # This should be consistent with the non-parallel version.
            # padding_idx=self.padding_idx - self.vocab_start_index if self.vocab_start_index <= self.padding_idx < self.vocab_end_index else None,
            config=_MEGATRON_MP_CONFIG,
            init_method=init.xavier_normal_,
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # # register a causal mask to separate causal and padding mask creation. Merging happends in the attention class
        # causal_mask = torch.full((config.max_position_embeddings, config.max_position_embeddings), fill_value=1)
        # self.register_buffer("causal_mask", torch.triu(causal_mask, diagonal=1), persistent=False)
        # Initialize weights and apply final processing
        self.post_init()


class Qwen2ForCausalLM(PretrainedModelParallelPreSplitMixin, HfQwen2ForCausalLM):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)

        self.model = Qwen2ModelParallel(config)
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, config=_MEGATRON_MP_CONFIG, init_method=lambda x: x,
                                            gather_output=True)

        self.post_init()

    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                cache_position: Optional[torch.LongTensor] = None,
                **kwargs,
                ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                shift_labels[shift_labels.eq(self.config.pad_token_id)] = -100  # Take care of here.
                loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Qwen2ModelForSequenceClassification(PretrainedModelParallelPreSplitMixin, Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2ModelParallel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            values: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        rewards, sequence_lengths = llama_last_token_cls_batch_forward(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(rewards, values)

        return DPOModelOutput(
            loss=loss,
            logits=rewards,
        )


class Qwen2ModelForSequenceClassificationForRL(PretrainedModelParallelPreSplitMixin, Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config, reduce_func: Callable):
        super().__init__(config)
        self.model = Qwen2ModelParallel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        self.reduce_func = reduce_func

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Union[Tuple, RewardModelOutput]:
        values, rewards, sequence_lengths = llama_last_token_forward_value(self.model, self.score, input_ids, attention_mask, self.config.pad_token_id)
        values = self.reduce_func(values)
        rewards = self.reduce_func(rewards)

        value_mask = input_ids.eq(self.config.pad_token_id)
        values = values.masked_fill(value_mask, 0)

        return RewardModelOutput(
            values=values,
            chosen_end_scores=rewards,
            sequence_lengths=sequence_lengths,
        )


class Qwen2ForCausalLMDPO(Qwen2ForCausalLM):
    def __init__(self, config, beta: float = 0.1, label_smoothing: float = 0.0, use_ipo: bool = False, loss_type: str = "sigmoid",
                 sft_loss: bool = False, sft_loss_weight: float = 1.0):
        super().__init__(config)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.use_ipo = use_ipo
        self.loss_type = loss_type
        self.sft_loss = sft_loss
        self.sft_loss_weight = sft_loss_weight
        logger.warning(f"Using loss type: {self.loss_type}")

        # Initialize weights and apply final processing
        self.post_init()

    @torch.amp.autocast("cuda", enabled=True, dtype=torch.float32)
    def dpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        if self.use_ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "sigmoid":
            log_sigmoid = nn.LogSigmoid()
            losses = -log_sigmoid(self.beta * logits) * (1 - self.label_smoothing) - log_sigmoid(-self.beta * logits) * self.label_smoothing
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses.mean(), chosen_rewards, rejected_rewards

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, DPOModelOutput]:
        half = input_ids.size(0) // 2

        policy_logits, policy_logprobs, policy_loss_mask = llama_dpo_batch_forward(self, input_ids, attention_mask, labels, self.config.pad_token_id)
        with torch.no_grad():
            ref_logits, ref_logprobs, ref_loss_mask = llama_dpo_batch_forward(return_reference_model(), input_ids, attention_mask, labels,
                                                                              self.config.pad_token_id)

        policy_chosen_logits, policy_reject_logits = policy_logits[:half], policy_logits[half:]
        policy_chosen_logprobs, policy_reject_logprobs = policy_logprobs[:half], policy_logprobs[half:]

        ref_chosen_logprobs, ref_reject_logprobs = ref_logprobs[:half], ref_logprobs[half:]

        loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps=policy_chosen_logprobs,
            policy_rejected_logps=policy_reject_logprobs,
            reference_chosen_logps=ref_chosen_logprobs,
            reference_rejected_logps=ref_reject_logprobs,
            reference_free=False,
        )

        if self.sft_loss:
            sft_loss = sft_loss_on_logits(policy_chosen_logits, labels[:half], self.config.pad_token_id)
            loss += self.sft_loss_weight * sft_loss
        else:
            sft_loss = None

        return DPOModelOutput(
            loss=loss,
            chosen_reward=chosen_rewards.mean(),
            rejected_reward=rejected_rewards.mean(),
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_reject_logits,
            sft_loss=sft_loss,
        )

    def save_pretrained(
            self,
            save_directory: Union[str, os.PathLike],
            is_main_process: bool = True,
            state_dict: Optional[dict] = None,
            save_function: Callable = torch.save,
            push_to_hub: bool = False,
            max_shard_size: Union[int, str] = "5GB",
            safe_serialization: bool = True,
            variant: Optional[str] = None,
            token: Optional[Union[str, bool]] = None,
            save_peft_format: bool = True,
            **kwargs,
    ):
        super().save_pretrained(save_directory, is_main_process, state_dict, save_function, push_to_hub, max_shard_size, safe_serialization, variant, token,
                                **kwargs)

        if mpu.model_parallel_is_initialized():
            mp_rank = mpu.get_tensor_model_parallel_rank()
            save_directory = os.path.join(save_directory, f"mp_{mp_rank}-of-{mpu.get_tensor_model_parallel_world_size()}")

        if is_main_process:
            config = self.config
            config.architectures = ["qwen2ForCausalLM"]
            config.save_pretrained(save_directory)
            logger.warning("Config architecture is override to qwen2ForCausalLM")
