import os
from typing import Optional, List, Tuple, Union, Callable

import torch
from fairscale.nn.model_parallel import initialize as mpu
from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from fairscale.nn.model_parallel.utils import VocabUtility
from torch import nn
from transformers.models.mistral import modeling_mistral
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM as HfMistralForCausalLM,
    CausalLMOutputWithPast,
    MistralAttention,
    MistralFlashAttention2,
    MistralSdpaAttention,
    MistralModel,
    is_flash_attn_greater_or_equal_2_10,
    MistralMLP,
    MistralConfig,
    MISTRAL_ATTENTION_CLASSES,
)

from general_util.logger import get_child_logger
from models.dpo_utils import (
    llama_dpo_batch_forward,
    llama_batch_forward,
    sft_loss_on_logits,
    tdpo_get_batch_logps,
)
from models.mixin import PretrainedModelParallelPreSplitMixin, return_reference_model
from models.utils import DPOModelOutput

logger = get_child_logger(__name__)


def attention_tp_init(self: MistralAttention, config: MistralConfig):
    self.q_proj = ColumnParallelLinear(
        self.hidden_size,
        self.num_heads * self.head_dim,
        bias=False,
        gather_output=False,
        init_method=lambda x: x
    )
    self.k_proj = ColumnParallelLinear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=False,
        gather_output=False,
        init_method=lambda x: x
    )
    self.v_proj = ColumnParallelLinear(
        self.hidden_size,
        self.num_key_value_heads * self.head_dim,
        bias=False,
        gather_output=False,
        init_method=lambda x: x
    )
    self.o_proj = RowParallelLinear(
        self.hidden_size,
        self.hidden_size,
        bias=False,
        input_is_parallel=True,
        init_method=lambda x: x
    )

    self.num_heads = self.num_heads // mpu.get_model_parallel_world_size()
    self.num_key_value_heads = self.num_key_value_heads // mpu.get_model_parallel_world_size()
    self.hidden_size = self.hidden_size // mpu.get_model_parallel_world_size()


class MistralAttentionTensorParallel(MistralAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        attention_tp_init(self, config)


class MistralFlashAttentionTensorParallel(MistralFlashAttention2):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        attention_tp_init(self, config)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


class MistralSdpaAttentionTensorParallel(MistralSdpaAttention):
    def __init__(self, config: MistralConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        attention_tp_init(self, config)


class MistralMLPTensorParallel(MistralMLP):
    def __init__(self, config: MistralConfig):
        super().__init__(config)

        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            init_method=lambda x: x
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x
        )


MISTRAL_ATTENTION_CLASSES["eager"] = MistralAttentionTensorParallel
MISTRAL_ATTENTION_CLASSES["flash_attention_2"] = MistralFlashAttentionTensorParallel
MISTRAL_ATTENTION_CLASSES["sdpa"] = MistralSdpaAttentionTensorParallel
modeling_mistral.MistralMLP = MistralMLPTensorParallel


class MistralModelTensorParallel(MistralModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            config.vocab_size, mpu.get_model_parallel_rank(), mpu.get_model_parallel_world_size()
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=self.padding_idx - self.vocab_start_index if self.vocab_start_index <= self.padding_idx < self.vocab_end_index else None,
        )
        # Initialize weights and apply final processing
        self.post_init()


class MistralForCausalLM(PretrainedModelParallelPreSplitMixin, HfMistralForCausalLM):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.model = MistralModelTensorParallel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_labels[shift_labels.eq(self.config.pad_token_id)] = -100
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


class MistralForCausalLMDPO(MistralForCausalLM):
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

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
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

        policy_logits, policy_logprobs, policy_loss_mask = llama_dpo_batch_forward(self, input_ids, attention_mask, labels)
        with torch.no_grad():
            ref_logits, ref_logprobs, ref_loss_mask = llama_dpo_batch_forward(return_reference_model(), input_ids, attention_mask, labels,
                                                                              pad_token_id=self.config.pad_token_id)

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
            mp_rank = mpu.get_model_parallel_rank()
            save_directory = os.path.join(save_directory, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")

        if is_main_process:
            config = self.config
            config.architectures = ["MistralForCausalLM"]
            config.save_pretrained(save_directory)
            logger.warning("Config architecture is override to MistralForCausalLM")


class MistralForCausalLMTDPO(MistralForCausalLM):
    def __init__(self, config, beta: float, alpha: float = 0.5, sft_loss: bool = False, sft_loss_weight: float = 1.0, if_tdpo2: bool = True, ):
        super().__init__(config)
        self.beta = beta
        self.alpha = alpha
        self.sft_loss = sft_loss
        self.sft_loss_weight = sft_loss_weight
        self.if_tdpo2 = if_tdpo2

        # Initialize weights and apply final processing
        self.post_init()

    @torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
    def tdpo_loss(self, chosen_logps_margin: torch.FloatTensor,
                  rejected_logps_margin: torch.FloatTensor,
                  chosen_position_kl: torch.FloatTensor,
                  rejected_position_kl: torch.FloatTensor,
                  ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the TDPO loss for a batch of policy and reference model log probabilities.

        Args:
            chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
            rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
            beta: Temperature parameter for the TDPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
            alpha: Temperature parameter for the TDPO loss, used to adjust the impact of sequential kl divergence.
            if_tdpo2: Determine whether to use method TDPO2, default is True; if False, then use method TDPO1.

        Returns:
            A tuple of two tensors: (losses, rewards).
            The losses tensor contains the TDPO loss for each example in the batch.
            The rewards tensors contain the rewards for response pair.
        """

        chosen_values = chosen_logps_margin + chosen_position_kl
        rejected_values = rejected_logps_margin + rejected_position_kl

        chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

        if not self.if_tdpo2:
            logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)  # tdpo1
        else:
            logits = chosen_rejected_logps_margin - self.alpha * (rejected_position_kl - chosen_position_kl.detach())  # tdpo2

        log_sigmoid = torch.nn.LogSigmoid()
        losses = -log_sigmoid(self.beta * logits)

        chosen_rewards = self.beta * chosen_values.detach()
        rejected_rewards = self.beta * rejected_values.detach()

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

        policy_logits = llama_batch_forward(self, input_ids, attention_mask).to(torch.float32)
        with torch.no_grad():
            ref_logits = llama_batch_forward(return_reference_model(), input_ids, attention_mask).to(torch.float32)

        logps_margin, position_kl, logps = tdpo_get_batch_logps(policy_logits, ref_logits, labels, self.config.pad_token_id,
                                                                average_log_prob=False)

        chosen_logps_margin, rejected_logps_margin = logps_margin[:half], logps_margin[half:]
        chosen_position_kl, rejected_position_kl = position_kl[:half], position_kl[half:]

        chosen_logps, rejected_logps = logps[:half].detach(), logps[half:].detach()

        loss, chosen_rewards, rejected_rewards = self.tdpo_loss(
            chosen_logps_margin=chosen_logps_margin,
            rejected_logps_margin=rejected_logps_margin,
            chosen_position_kl=chosen_position_kl,
            rejected_position_kl=rejected_position_kl,
        )

        if self.sft_loss:
            sft_loss = sft_loss_on_logits(policy_logits[:half], labels[:half], self.config.pad_token_id)
            loss += self.sft_loss_weight * sft_loss
        else:
            sft_loss = None

        return DPOModelOutput(
            loss=loss,
            chosen_reward=chosen_rewards.mean(),
            rejected_reward=rejected_rewards.mean(),
            policy_chosen_logits=policy_logits[:half],
            policy_rejected_logits=policy_logits[half:],
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
            mp_rank = mpu.get_model_parallel_rank()
            save_directory = os.path.join(save_directory, f"mp_{mp_rank}-of-{mpu.get_model_parallel_world_size()}")

        if is_main_process:
            config = self.config
            config.architectures = ["MistralForCausalLM"]
            config.save_pretrained(save_directory)
            logger.warning("Config architecture is override to MistralForCausalLM")
