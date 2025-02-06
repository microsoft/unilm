import os
from dataclasses import dataclass
from typing import Optional

import hydra.utils
import omegaconf
import torch
from omegaconf import DictConfig
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from general_util.logger import get_child_logger
from general_util.training_utils import get_rank

logger = get_child_logger(__name__)

LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

try:
    import bitsandbytes as bnb
except:
    bnb = None


def find_all_linear_names(model, bits: int, add_lm_head: bool = False):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    lora_module_names.add("lm_head")

    if 'lm_head' in lora_module_names and not add_lm_head:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def initialize_peft_model(model: PreTrainedModel, lora_config: DictConfig, load_in_8bit: bool = False, load_in_4bit: bool = False,
                          torch_dtype: torch.dtype = torch.bfloat16):
    if lora_config is None:
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                 lora_dropout=0.1)

    logger.warning(lora_config)
    logger.info(lora_config.target_modules.__class__)
    if isinstance(lora_config.target_modules, omegaconf.listconfig.ListConfig):
        lora_config.target_modules = list(lora_config.target_modules)
    elif isinstance(lora_config.target_modules, omegaconf.DictConfig):
        lora_config.target_modules = hydra.utils.instantiate(lora_config.target_modules, model=model)
    else:
        raise ValueError(f"Unsupported type of target modules: {lora_config.target_modules.__class__}")

    if isinstance(lora_config.modules_to_save, omegaconf.listconfig.ListConfig):
        lora_config.modules_to_save = list(lora_config.modules_to_save)

    logger.warning(lora_config.target_modules)
    gradient_checkpointing = model.model.gradient_checkpointing
    if load_in_8bit or load_in_4bit:
        logger.warning(f"Rank {get_rank()} is being loaded in 8-{load_in_8bit} | 4-{load_in_4bit} bit.")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    model = get_peft_model(model, lora_config)

    compute_dtype = torch_dtype
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if compute_dtype == torch.bfloat16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if compute_dtype and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()

    return model


def enable_gradient_checkpointing(model: PreTrainedModel):
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    return model


@dataclass
class DPOModelOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    chosen_reward: torch.FloatTensor = None
    rejected_reward: torch.FloatTensor = None
    policy_chosen_logits: Optional[torch.FloatTensor] = None
    policy_rejected_logits: Optional[torch.FloatTensor] = None
    batch_chosen_reward: Optional[torch.FloatTensor] = None
    batch_rejected_reward: Optional[torch.FloatTensor] = None
    sft_loss: Optional[torch.FloatTensor] = None


@dataclass
class RewardModelOutput(ModelOutput):
    values: torch.FloatTensor = None
    chosen_end_scores: torch.FloatTensor = None
    sequence_lengths: torch.LongTensor = None


def return_single_device_map():
    return {"": "cuda:" + str(int(os.environ.get("LOCAL_RANK") or 0))}


def reward_logit2prob(reduction_ids):
    if isinstance(reduction_ids, omegaconf.ListConfig):
        reduction_ids = list(reduction_ids)

    def func(logits):
        probs = torch.softmax(logits, dim=-1)
        if len(logits.size()) == 3:
            probs = probs[:, :, reduction_ids].sum(dim=-1)
        elif len(logits.size()) == 2:
            probs = probs[:, reduction_ids].sum(dim=-1)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.size()}")
        return probs

    return func


def reward_logit(reduction_ids):
    if isinstance(reduction_ids, omegaconf.ListConfig):
        reduction_ids = list(reduction_ids)

    def func(logits):
        if len(logits.size()) == 3:
            logits = logits[:, :, reduction_ids].sum(dim=-1)
        elif len(logits.size()) == 2:
            logits = logits[:, reduction_ids].sum(dim=-1)
        else:
            raise ValueError(f"Unsupported logits shape: {logits.size()}")
        return logits

    return func


def squeeze_reduce_return_fn():
    def func(logits: torch.Tensor):
        return logits.squeeze(-1)

    return func
