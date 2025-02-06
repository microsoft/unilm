import collections
import copy
import json
import os.path
import random
from glob import glob
from typing import List, Dict, Tuple, Union, Any, Callable, Optional

import torch
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from data.math import decompose_deepseek_math_cot_v2

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class DPOCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, padding: str = "longest"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding

    def __call__(self, batch):
        chosen = [item["chosen"] for item in batch]
        reject = [item["reject"] for item in batch]
        indices = [item["index"] for item in batch]
        text_inputs = chosen + reject

        text_prompts = []
        for item in batch:
            if "chosen_prompt" in item:
                text_prompts.append(item["chosen_prompt"])
            else:
                text_prompts.append(item["prompt"])
        for item in batch:
            if "reject_prompt" in item:
                text_prompts.append(item["reject_prompt"])
            else:
                text_prompts.append(item["prompt"])

        encoded_prompts = self.tokenizer(text_prompts, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        # TODO: @2024/09/13
        #   There is another case that the chosen prompt is sth. like <prompt> + <space> + <eos>
        #   Since usually I also set pad_token as eos_token, then the labels here could be all pad_token.
        #   This could cause NAN loss when computing SFT loss.
        if prompt_mask.sum() == labels.numel():  # FIXME: This could also induce NAN loss during DPO with SFT loss. @2024/08/09
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": text_prompts,
            "chosen": chosen,
            "reject": reject,
        }
        return encoded_inputs


class DPODataSFTCollator:
    """
    Note that when you are using the DPO pair dataset, you may overlook the oversampling of chosen samples.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        indices = [item["index"] for item in batch]

        text_prompts = prompt
        text_inputs = chosen

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        if prompt_mask.sum() == labels.numel():
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "chosen": chosen,
            "response": chosen,
        }
        if "label" in batch[0]:
            encoded_inputs["meta_data"]["label"] = [item["label"] for item in batch]
        return encoded_inputs


class DPOCollatorWithExtraInputs:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, padding: str = "longest", extra_keys: List[str] = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.padding = padding
        self.extra_keys = extra_keys

    def __call__(self, batch):
        chosen = [item["chosen"] for item in batch]
        reject = [item["reject"] for item in batch]
        indices = [item["index"] for item in batch]
        text_inputs = chosen + reject

        text_prompts = []
        for item in batch:
            if "chosen_prompt" in item:
                text_prompts.append(item["chosen_prompt"])
            else:
                text_prompts.append(item["prompt"])
        for item in batch:
            if "reject_prompt" in item:
                text_prompts.append(item["reject_prompt"])
            else:
                text_prompts.append(item["prompt"])

        encoded_prompts = self.tokenizer(text_prompts, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding=self.padding, truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        # TODO: @2024/09/13
        #   There is another case that the chosen prompt is sth. like <prompt> + <space> + <eos>
        #   Since usually I also set pad_token as eos_token, then the labels here could be all pad_token.
        #   This could cause NAN loss when computing SFT loss.
        if prompt_mask.sum() == labels.numel():  # FIXME: This could also induce NAN loss during DPO with SFT loss. @2024/08/09
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels

        for k in self.extra_keys:
            _ex_inputs = [item[k] for item in batch]
            _ex_inputs = torch.tensor(_ex_inputs, dtype=torch.float)
            encoded_inputs[k] = _ex_inputs

        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": text_prompts,
            "chosen": chosen,
            "reject": reject,
        }
        return encoded_inputs


class Trajectory2ValueCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        inputs = [item["input"] for item in batch]
        indices = [item["index"] for item in batch]
        values = [item["value"] for item in batch]

        text_prompts = prompt
        text_inputs = inputs

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        if prompt_mask.sum() == labels.numel():
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["values"] = torch.tensor(values, dtype=torch.long)
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "input": inputs,
            "response": inputs,
            "label": values,
        }
        return encoded_inputs


class StepEndingsCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        indices = [item["index"] for item in batch]

        text_prompts = prompt
        text_inputs = chosen

        encoded_prompts = self.tokenizer(text_prompts, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        input_lens = torch.sum(encoded_prompts["attention_mask"], dim=-1)

        encoded_inputs = self.tokenizer(text_inputs, padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")
        if self.tokenizer.padding_side == "left":
            padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
            input_lens = input_lens + padding_len
        else:
            padding_len = torch.zeros(len(batch), dtype=torch.long)

        labels = encoded_inputs["input_ids"].clone()
        prompt_mask = torch.arange(encoded_inputs["input_ids"].size(1))[None, :] < input_lens[:, None]
        if prompt_mask.sum() == labels.numel():
            logger.warning(f"Prompt mask is all True. Indices: {indices}")
            prompt_mask[0, -1] = False

        labels[prompt_mask] = self.tokenizer.pad_token_id

        endings = []
        for b, item in enumerate(batch):
            ending = decompose_deepseek_math_cot_v2(item["prompt"], item["response"], self.max_seq_length, self.tokenizer)
            ending = [e + padding_len[b].item() for e in ending]
            endings.append(ending)

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "chosen": chosen,
            "response": [item["response"] for item in batch],
            "ending": endings,
            "type": [None] * len(endings),
        }
        if "label" in batch[0]:
            encoded_inputs["meta_data"]["label"] = [item["label"] for item in batch]
        return encoded_inputs


def iterative_mask(text_segment_list: List[List[str]], masks: List[int], tokenizer: PreTrainedTokenizer, **tokenize_kwargs):
    if len(text_segment_list) == 0:
        raise ValueError("Input groups should be greater than 0.")
    if len(text_segment_list) == 1:
        return tokenizer(text_segment_list[0], **tokenize_kwargs), None
    assert len(masks) == 1 or masks[0] == 0, "The prefix should always be masked if there are multiple groups of inputs"

    all_input_lens = []
    all_inputs = []
    for group in text_segment_list:
        group_inputs = tokenizer(group, **tokenize_kwargs)
        all_inputs.append(group_inputs)
        all_input_lens.append(torch.sum(group_inputs["attention_mask"], dim=-1, keepdim=True))

    if tokenizer.padding_side == "left":
        # If left padding, we should first compute the padding length at last
        padding_len = torch.sum(1 - all_inputs[-1]["attention_mask"], dim=-1, keepdim=True)
    else:
        padding_len = 0

    last_len = torch.zeros(all_inputs[-1]["input_ids"].size(0), 1, dtype=torch.long)
    prompt_mask = torch.zeros(all_inputs[-1]["input_ids"].shape, dtype=torch.long)
    seq_range = torch.arange(all_inputs[-1]["input_ids"].size(1))
    for _acc_lens, mask in zip(all_input_lens, masks):
        _condition_lens = padding_len + _acc_lens
        if mask == 0:
            group_mask = (seq_range[None, :] < _condition_lens) & (last_len <= seq_range[None, :])
            prompt_mask += group_mask
        last_len = _condition_lens
    return all_inputs[-1], prompt_mask


class SFTFoldAttnMaskCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, text_keys: List[str], text_masks: List[int]):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.text_keys = text_keys
        self.text_masks = text_masks

    def __call__(self, batch):
        text_segment_list = []

        for b in batch:
            assert self.text_keys[0] in b, f"At least the first group of inputs is contained in the batch, got {list(b.keys())}"
            last_input = b[self.text_keys[0]]
            for k in self.text_keys[1:]:
                if k not in b:
                    b[k] = last_input
                last_input = b[k]

        for i, key in enumerate(self.text_keys):
            batch_item = [b[key] for b in batch]
            text_segment_list.append(batch_item)

        indices = [item["index"] for item in batch]

        encoded_inputs, prompt_mask = iterative_mask(text_segment_list, self.text_masks, self.tokenizer,
                                                     padding="longest", truncation=True, max_length=self.max_seq_length, return_tensors="pt")

        labels = encoded_inputs["input_ids"].clone()
        if prompt_mask is not None:
            if prompt_mask.sum() == labels.numel():
                logger.warning(f"Prompt mask is all True. Indices: {indices}")
                prompt_mask[0, -1] = False

            labels[prompt_mask] = self.tokenizer.pad_token_id

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
        }
        encoded_inputs["meta_data"].update({
            k: text_segment_list[i] for i, k in enumerate(self.text_keys)
        })
        if "label" in batch[0]:
            encoded_inputs["meta_data"]["label"] = [item["label"] for item in batch]
        return encoded_inputs


class TextPromptCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, padding: str = "longest", extra_text_inputs: DictConfig[str, bool] = None,
                 **kwargs):
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.max_seq_length = max_seq_length
        self.padding = padding

        self.extra_text_inputs = extra_text_inputs

    def __call__(self, batch):
        inputs = [b["input"] for b in batch]
        index = [b["index"] for b in batch]

        model_inputs = self.tokenizer(inputs, padding=self.padding, truncation=True, max_length=self.max_seq_length,
                                      return_tensors="pt")

        if self.extra_text_inputs is not None:
            for k, v in self.extra_text_inputs.items():
                _ex_inputs = [b[k] for b in batch]
                if v:
                    model_inputs[k] = self.tokenizer(_ex_inputs, padding=self.padding, truncation=True, max_length=self.max_seq_length,
                                                     return_tensors="pt")
                else:
                    model_inputs[k] = _ex_inputs

        model_inputs["meta_data"] = {
            "inputs": inputs,
            "index": index,
        }
        return model_inputs
