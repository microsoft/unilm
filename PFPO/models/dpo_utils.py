import torch
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.models.gemma.modeling_gemma import (
    GemmaForCausalLM,
    GemmaModel,
)
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM,
    MistralPreTrainedModel,
)
from typing import Union


def sft_loss_on_logits(logits: torch.FloatTensor, labels: torch.LongTensor, pad_token_id: int, macro_average: bool = False, row_weights: torch.Tensor = None):
    batch_size = labels.size(0)
    labels = labels[:, 1:].contiguous()
    logits = logits[:, :-1].contiguous()

    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)

    pad_mask = labels.eq(pad_token_id)
    if pad_mask.sum() == labels.numel():  # To tackle all problems that the response are empty, or the pad_token equals eos_token so no response.
        return 0.

    labels[pad_mask] = -100
    if macro_average:
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits, labels)

        row_num_element = (~pad_mask).reshape(batch_size, -1).sum(-1).float()
        row_mask = row_num_element > 0
        loss = loss.view(batch_size, -1)
        loss = loss.sum(-1) / row_num_element
        if row_weights is not None:
            loss = loss * row_weights
        loss = loss[row_mask].mean()
    else:
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
    return loss


def llama_dpo_batch_forward(model: Union[LlamaForCausalLM, GemmaForCausalLM, MistralForCausalLM],
                            input_ids: torch.LongTensor, attention_mask: torch.Tensor, labels: torch.LongTensor, pad_token_id: int = None,
                            average_log_prob: bool = False):
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)
    logits = logits.float()

    labels = labels[:, 1:].clone()

    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id

    loss_mask = labels.ne(pad_token_id)
    labels[~loss_mask] = 0

    per_token_logprobs = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        log_ps = (per_token_logprobs * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        log_ps = (per_token_logprobs * loss_mask).sum(-1)

    return logits, log_ps, loss_mask


def llama_batch_forward(model: Union[LlamaForCausalLM, GemmaForCausalLM, MistralForCausalLM], input_ids: torch.LongTensor, attention_mask: torch.Tensor):
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    hidden_states = outputs[0]
    logits = model.lm_head(hidden_states)

    return logits


def tdpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor, pad_token_id: int,
                         average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        pad_token_id: The id of the padding token.
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the
            (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = labels.ne(pad_token_id)

    # dummy token; we'll ignore the losses on these tokens later
    labels[~loss_mask] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * loss_mask).sum(-1), \
            (per_position_kl * loss_mask).sum(-1), \
            (per_token_logps * loss_mask).sum(-1)


def llama_last_token_cls_batch_forward(model: Union[LlamaModel, GemmaForCausalLM, MistralPreTrainedModel], linear: nn.Linear,
                                       input_ids: torch.LongTensor, attention_mask: torch.Tensor,
                                       pad_token_id: int, return_full_logits: bool = False):
    transformer_outputs = model(
        input_ids,
        attention_mask=attention_mask,
    )
    hidden_states = transformer_outputs[0]

    batch_size = input_ids.shape[0]
    sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(device=hidden_states.device)

    if return_full_logits:
        rewards = linear(hidden_states)
        return rewards, sequence_lengths

    last_token_states = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
    rewards = linear(last_token_states)
    return rewards, sequence_lengths


def llama_token_batch_forward(model: Union[LlamaModel, GemmaModel], linear: nn.Linear,
                              input_ids: torch.LongTensor, attention_mask: torch.Tensor, pad_token_id: int = None, average: bool = False):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    hidden_states = outputs[0]
    logits = linear(hidden_states).squeeze(-1)
    logits = logits.float()

    if pad_token_id is None:
        pad_token_id = model.config.pad_token_id

    loss_mask = input_ids.ne(pad_token_id)

    if average:
        rewards = (logits * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        rewards = (logits * loss_mask).sum(-1)

    return rewards


def llama_last_token_forward_value(model: Union[LlamaModel, GemmaForCausalLM], linear: nn.Linear,
                                   input_ids: torch.LongTensor, attention_mask: torch.Tensor,
                                   pad_token_id: int):
    transformer_outputs = model(
        input_ids,
        attention_mask=attention_mask,
    )
    hidden_states = transformer_outputs[0]

    batch_size = input_ids.shape[0]
    sequence_lengths = (torch.eq(input_ids, pad_token_id).long().argmax(-1) - 1).to(device=hidden_states.device)

    values = linear(hidden_states)
    rewards = values[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]
    return values, rewards, sequence_lengths
