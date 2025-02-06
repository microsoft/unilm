import collections
import json
import os.path
import random
from typing import List, Dict, Callable

import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from data.input_aligner import empty_aligner
from data.input_utils import json_read_fn
from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


class ResponseAlignDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 template: str,
                 aligner: Callable = empty_aligner,
                 instruction: str = "",
                 few_shot_prompt: str = "",
                 api_based: bool = False, message_compose_fn: Callable = None,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None,
                 split_size: int = -1,
                 split_id: int = 0,
                 index_field: str = "id",
                 max_data_num: int = -1,
                 read_fn: Callable = json_read_fn,
                 replacement: Dict[str, str] = None,
                 ):
        self.tokenizer = tokenizer
        self.template = template
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.api_based = api_based
        self.message_compose_fn = message_compose_fn
        self.service_based = service_based
        self.service_processor = service_processor
        self.flush_file = flush_file
        self.split_size = split_size
        self.split_id = split_id
        self.index_field = index_field
        self.max_data_num = max_data_num
        self.replacement = replacement

        data = read_fn(file_path)
        self.data: List[Dict] = aligner(data)

        for item in self.data:
            if self.instruction:
                item["instruction"] = self.instruction
            if self.few_shot_prompt:
                item["few_shot_prompt"] = self.few_shot_prompt

        flushed_data = set()
        flushed_full_data = []
        if flush_file is not None and os.path.exists(flush_file):
            tmp = open(flush_file, "r", encoding="utf-8").readlines()
            has_reading_error = False
            for line in tmp:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error reading line: {line}")
                    has_reading_error = True
                    continue
                if "response" in item and item["response"]:
                    if item["id"] in flushed_data:
                        continue
                    flushed_data.add(item["id"])
                    flushed_full_data.append(item)
            logger.info(f"Loaded flushed data: {len(flushed_data)} from {flush_file}")
            if has_reading_error:
                logger.warning("There are reading errors in the flush file")
                with open(flush_file, "w", encoding="utf-8") as f:
                    for item in flushed_full_data:
                        f.write(json.dumps(item) + "\n")

        if split_size > 0:
            batch_size = (len(self.data) + split_size - 1) // split_size
            self.data = self.data[split_id * batch_size: (split_id + 1) * batch_size]

        self.data = [item for item in self.data if item[self.index_field] not in flushed_data and str(item[self.index_field]) not in flushed_data]

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.data))
        return len(self.data)

    def api_getitem(self, index):
        item = self.data[index]
        if self.replacement:
            tmp = self.template
            for k, v in self.replacement.items():
                tmp = tmp.replace(k, item[v])
            text = tmp
        else:
            text = self.template.format(**item)
        if self.message_compose_fn is not None:
            text = self.message_compose_fn(text)
        item["text"] = text
        return {
            "text": text,
            "meta_data": item,
        }

    def service_getitem(self, index):
        inputs = self.api_getitem(index)
        response = self.service_processor(inputs["text"])
        inputs["response"] = response
        return inputs

    def __getitem__(self, idx):
        if self.api_based:
            return self.api_getitem(idx)
        if self.service_based:
            return self.service_getitem(idx)
        item = self.data[idx]
        if self.replacement:
            tmp = self.template
            for k, v in self.replacement.items():
                tmp = tmp.replace(k, item[v])
            text = tmp
        else:
            text = self.template.format(**item)
        item["text"] = text
        return {
            "text": text,
            "meta_data": item,
        }


class PromptResponseDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 prompt_template: str,
                 response_template: str,
                 aligner: Callable = empty_aligner,
                 instruction: str = "",
                 few_shot_prompt: str = "",
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None,
                 split_size: int = -1,
                 split_id: int = 0,
                 index_field: str = "id",
                 max_data_num: int = -1,
                 read_fn: Callable = json_read_fn,
                 kv_mapping: Dict[str, str] = None,
                 ):
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor
        self.flush_file = flush_file
        self.split_size = split_size
        self.split_id = split_id
        self.index_field = index_field
        self.max_data_num = max_data_num
        self.kv_mapping = kv_mapping

        data = read_fn(file_path)
        self.data: List[Dict] = aligner(data)

        for item in self.data:
            if self.instruction:
                item["instruction"] = self.instruction
            if self.few_shot_prompt:
                item["few_shot_prompt"] = self.few_shot_prompt

        flushed_data = set()
        if flush_file is not None and os.path.exists(flush_file):
            tmp = open(flush_file, "r", encoding="utf-8").readlines()
            for line in tmp:
                item = json.loads(line)
                if "response" in item and item["response"].strip() != "":
                    flushed_data.add(item["id"])
            logger.info(f"Loaded flushed data: {len(flushed_data)} from {flush_file}")

        self.data = [item for item in self.data if item[self.index_field] not in flushed_data]

        if split_size > 0:
            batch_size = (len(self.data) + split_size - 1) // split_size
            self.data = self.data[split_id * batch_size: (split_id + 1) * batch_size]

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.data))
        return len(self.data)

    def api_getitem(self, index):
        raise NotImplementedError

    def service_getitem(self, index):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.api_based:
            return self.api_getitem(idx)
        if self.service_based:
            return self.service_getitem(idx)
        item = self.data[idx]
        prompt = self.prompt_template.format(**item)
        response = self.response_template.format(**item)
        text = prompt + response
        item["text"] = text
        item["prompt"] = prompt

        if not self.kv_mapping:
            return {
                "text": text,
                "meta_data": item,
            }

        res = {v: item[k] for k, v in self.kv_mapping.items()}
        res["meta_data"] = item
        return res


class MultiMappingDataset(Dataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 template: Dict[str, str],
                 aligner: Callable = empty_aligner,
                 instruction: str = "",
                 few_shot_prompt: str = "",
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None,
                 split_size: int = -1,
                 split_id: int = 0,
                 index_field: str = "id",
                 max_data_num: int = -1,
                 read_fn: Callable = json_read_fn,
                 kv_mapping: Dict[str, str] = None,
                 replacement: Dict[str, str] = None,
                 ):
        self.tokenizer = tokenizer
        self.template = template
        self.instruction = instruction
        self.few_shot_prompt = few_shot_prompt
        self.api_based = api_based
        self.service_based = service_based
        self.service_processor = service_processor
        self.flush_file = flush_file
        self.split_size = split_size
        self.split_id = split_id
        self.index_field = index_field
        self.max_data_num = max_data_num
        self.kv_mapping = kv_mapping
        self.replacement = replacement

        data = read_fn(file_path)
        self.data: List[Dict] = aligner(data)

        for item in self.data:
            if self.instruction:
                item["instruction"] = self.instruction
            if self.few_shot_prompt:
                item["few_shot_prompt"] = self.few_shot_prompt

        flushed_data = set()
        if flush_file is not None and os.path.exists(flush_file):
            tmp = open(flush_file, "r", encoding="utf-8").readlines()
            for line in tmp:
                item = json.loads(line)
                if "response" in item and item["response"]:
                    flushed_data.add(item["id"])
            logger.info(f"Loaded flushed data: {len(flushed_data)} from {flush_file}")

        self.data = [item for item in self.data if item[self.index_field] not in flushed_data]

        if split_size > 0:
            batch_size = (len(self.data) + split_size - 1) // split_size
            self.data = self.data[split_id * batch_size: (split_id + 1) * batch_size]

    def __len__(self):
        if self.max_data_num > 0:
            return min(self.max_data_num, len(self.data))
        return len(self.data)

    def api_getitem(self, index):
        raise NotImplementedError

    def service_getitem(self, index):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.api_based:
            return self.api_getitem(idx)
        if self.service_based:
            return self.service_getitem(idx)
        item = self.data[idx]

        inputs = {}
        if self.replacement:
            for name, tem in self.template.items():
                tmp = tem
                for k, v in self.replacement.items():
                    tmp = tmp.replace(k, item[v])
                inputs[name] = tmp
                item[name] = tmp
        else:
            for k, v in self.template.items():
                item[k] = v.format(**item)
                inputs[k] = item[k]
        inputs["meta_data"] = item

        if not self.kv_mapping:
            return inputs

        res = {v: item[k] for k, v in self.kv_mapping.items()}
        res["meta_data"] = item
        return res


class MultiMappingDatasetGrouping(MultiMappingDataset):
    def __init__(self,
                 file_path: str,
                 tokenizer: PreTrainedTokenizer,
                 template: Dict[str, str],
                 aligner: Callable = empty_aligner,
                 instruction: str = "",
                 few_shot_prompt: str = "",
                 api_based: bool = False,
                 service_based: bool = False, service_processor: Callable = None,
                 flush_file: str = None,
                 split_size: int = -1,
                 split_id: int = 0,
                 index_field: str = "id",
                 max_data_num: int = -1,
                 read_fn: Callable = json_read_fn,
                 kv_mapping: Dict[str, str] = None,
                 group_field: str = "id",
                 ):
        super().__init__(file_path,
                         tokenizer,
                         template,
                         aligner,
                         instruction,
                         few_shot_prompt,
                         api_based,
                         service_based,
                         service_processor,
                         flush_file,
                         split_size,
                         split_id,
                         index_field,
                         max_data_num,
                         read_fn,
                         kv_mapping)

        random.shuffle(self.data)
        groups = collections.defaultdict(list)
        for item in self.data:
            groups[item[group_field]].append(item)
        new_data = []
        for group in groups.values():
            new_data.extend(group)
        self.data = new_data


class ReplayDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, new_dataset_cfg: DictConfig, old_dataset_cfg: DictConfig, replay_ratio: float = 0.1):
        logger.info(f"Loading new dataset from {file_path}")
        self.new_dataset = hydra.utils.instantiate(new_dataset_cfg, file_path=file_path, tokenizer=tokenizer)
        logger.info(f"Loading old dataset from {old_dataset_cfg}")
        old_dataset = hydra.utils.instantiate(old_dataset_cfg, tokenizer=tokenizer)
        logger.info(f"Replay ratio: {replay_ratio}")
        self.replay_ratio = replay_ratio
        self.old_data = random.sample([item for item in old_dataset], int(len(old_dataset) * replay_ratio))

    def __len__(self):
        return len(self.new_dataset) + len(self.old_data)

    def __getitem__(self, index):
        if index < len(self.new_dataset):
            return self.new_dataset[index]
        else:
            return self.old_data[index - len(self.new_dataset)]
