import copy
import json
import os.path
import random
from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from typing import Dict, List, Callable, Union

from omegaconf.listconfig import ListConfig
from tqdm import tqdm

from general_util.logger import get_child_logger

logger = get_child_logger(__name__)


def _format_option_list(option_list: List[str], _rank2option: List[str]) -> str:
    res = []
    for op_id, op in enumerate(option_list):
        res.append(f"{_rank2option[op_id]}. {op}")
    return "\n".join(res)


def option_id2str_aligner():
    option_id2str = ["A", "B", "C", "D", "E"]

    def func(data: List[Dict]):
        for sample in data:
            sample["str_label"] = option_id2str[sample["label"]]
        return data

    return func


def key_based_filter_aligner(key, value):
    if isinstance(value, ListConfig):
        value = list(value)
    if not isinstance(value, list):
        value = [value]

    def func(data: List[Dict]):
        return [item for item in data if item[key] in value]

    return func


def dpo_confidence_ratio_filter(lower_bound: float, upper_bound: float, pos_field: str, response_field: str):
    def func(item):
        pos_num = len(item[pos_field])
        total_num = len(item[response_field])
        ratio = pos_num / total_num
        if lower_bound <= ratio <= upper_bound:
            return True
        return False

    return func


def filter_aligner(filter_func: Callable):
    def func(data: List[Dict]):
        return [item for item in data if filter_func(item)]

    return func


def json_field2str(key, val: str = None, indent: int = 4):
    def func(data: List[Dict]):
        for item in data:
            if val:
                item[val] = json.dumps(item[key], indent=indent, ensure_ascii=False)
            else:
                item[key] = json.dumps(item[key], indent=indent, ensure_ascii=False)
        return data

    return func


def starts_with_filter(key, value):
    def func(data: List[Dict]):
        return [item for item in data if item[key].startswith(value)]

    return func


def not_none_filter(key):
    def func(data: List[Dict]):
        return [item for item in data if key in item and item[key] not in ["", None, []]]

    return func


def rename_field_aligner(kv_pair: Dict):
    def func(data: List[Dict]):
        for item in data:
            for k, v in kv_pair.items():
                tmp = item.pop(k)
                item[v] = tmp
        return data

    return func


def field_extract_aligner(input_index_field: str, extract_index_field: str, extract_fields: List[str], extra_file: str, renamed_fields: Dict[str, str] = None):
    if os.path.exists(extra_file):
        extra_data = json.load(open(extra_file, encoding="utf-8"))
    else:
        extra_data = []
        for file in glob(extra_file):
            extra_data += json.load(open(file))
        if len(extra_data) == 0:
            raise ValueError(f"No data found in {extra_file}")
    id2extra_data = {str(item[extract_index_field]): item for item in extra_data}
    renaming = {}
    for _field in extract_fields:
        if renamed_fields and _field in renamed_fields:
            renaming[_field] = renamed_fields[_field]
        else:
            renaming[_field] = _field

    def func(data: List[Dict]):
        missing = 0
        missing_field = 0
        outputs = []
        for item in data:
            item_id = str(item[input_index_field])
            if item_id not in id2extra_data:
                missing += 1
                continue
            extra_item = id2extra_data[item_id]
            if any(x not in extra_item for x in extract_fields):
                missing_field += 1
                continue
            for field in extract_fields:
                item[renaming[field]] = extra_item[field]
            outputs.append(item)

        logger.info(f"Extracted {len(outputs)} items from {extra_file}")
        logger.info(f"Missing {missing} items in {extra_file}")
        logger.info(f"Missing {missing_field} fields in {extra_file}")

        return outputs

    return func


def flat_aligner(input_index_field: str, extract_field: Union[str, List[str]], mode: str = "single"):
    if isinstance(extract_field, str):
        extract_field = [extract_field]

    def func(data: List[Dict]):
        outputs = []
        for item in data:
            item_id = item[input_index_field]
            # if not all(item[_field] for _field in extract_field):
            #     continue
            if any(item[_field] in [None, "", []] for _field in extract_field):
                continue

            num = len(item[extract_field[0]])
            for _field in extract_field[1:]:
                assert len(item[_field]) == num, f"Length not match: {item[_field]}"

            tmp_list = []
            for i in range(num):
                new_item = copy.deepcopy(item)
                if any(item[_field][i] in [None, "", []] for _field in extract_field):
                    continue
                for _field in extract_field:
                    new_item[_field] = item[_field][i]
                new_item[input_index_field] = f"{item_id}_{i}"
                tmp_list.append(new_item)
                if mode == "single":
                    break

            if len(tmp_list) == 0:
                continue

            if mode == "single":
                outputs.append(tmp_list[0])
            elif mode == "random":
                outputs.append(random.choice(tmp_list))
            else:
                outputs += tmp_list

        return outputs

    return func


def option_flatten_aligner():
    def func(data: List[Dict]):
        for sample in data:
            sample["option_list"] = _format_option_list(sample["options"], ["A", "B", "C", "D"])
        return data

    return func


def empty_aligner(data: List[Dict]):
    return data


def add_id_aligner(id_field: str = "id"):
    def func(data: List[Dict]):
        for i, item in enumerate(data):
            item[id_field] = i
        return data

    return func


def concat_aligner(aligners: List[Callable]):
    def func(data: List[Dict]):
        for aligner in aligners:
            data = aligner(data)
        return data

    return func


def dpo_pair_aligner_cleaned(response_field: str = "response",
                             id_field: str = "id",
                             do_sample: bool = False, ):
    """
    This aligner only accepts the cleaned file, which has removing all empty responses and combined with original data.
    :return: Callable
    """

    def func(data: List[Dict]):
        outputs = []
        for item in data:
            pos_resp = []
            neg_resp = []
            for i, (resp, pred) in enumerate(zip(item[response_field], item["pred"])):
                assert resp
                # assert pred
                if isinstance(resp, list):
                    assert isinstance(resp[0], str)
                    # assert "The answer is" in resp[-1], resp
                    resp = "".join(resp)

                if isinstance(item["label"], str):
                    if pred == item["label"]:
                        pos_resp.append((i, resp))
                    else:
                        neg_resp.append((i, resp))
                elif isinstance(item["label"], int):
                    if pred and ord(pred) - ord("A") == item["label"]:
                        pos_resp.append((i, resp))
                    else:
                        neg_resp.append((i, resp))
                else:
                    raise ValueError(f"Unknown type of label: {type(item['label'])}")

            if not (len(pos_resp) and len(neg_resp)):
                continue

            if do_sample:
                pos = random.choice(pos_resp)
                neg = random.choice(neg_resp)
                pos_resp = [pos]
                neg_resp = [neg]

            for pos in pos_resp:
                for neg in neg_resp:
                    new_item = copy.deepcopy(item)
                    new_item["pos"] = pos[1]
                    new_item["neg"] = neg[1]
                    new_item["pos_id"] = f"{item[id_field]}_{pos[0]}"
                    new_item["neg_id"] = f"{item[id_field]}_{neg[0]}"
                    outputs.append(new_item)

        logger.info(f"Counted {len(outputs)} DPO contrastive pairs.")
        return outputs

    return func


def dpo_pair_aligner(pos_field: Union[str, ListConfig], neg_field: Union[str, ListConfig]):
    def func(data: List[Dict]):
        outputs = []
        if isinstance(pos_field, str):
            _pos_fields = [pos_field]
        else:
            _pos_fields = list(pos_field)
        if isinstance(neg_field, str):
            _neg_fields = [neg_field]
        else:
            _neg_fields = list(neg_field)

        for item in tqdm(data, desc="DPO pair aligner", total=len(data)):
            pos_resp = []
            neg_resp = []
            for _field in _pos_fields:
                pos_resp += item[_field]
            for _field in _neg_fields:
                neg_resp += item[_field]

            for pos in pos_resp:
                for neg in neg_resp:
                    new_item = copy.deepcopy(item)

                    # To save memory
                    for _field in _pos_fields:
                        new_item.pop(_field)
                    for _field in _neg_fields:
                        new_item.pop(_field)

                    new_item["pos"] = pos
                    new_item["neg"] = neg
                    outputs.append(new_item)

        logger.info(f"Counted {len(outputs)} DPO contrastive pairs.")
        return outputs

    return func


def eval_multiple_choice(item):
    if isinstance(item, dict):
        pred = item["prediction"]
        label = item["answer"]
    elif isinstance(item, tuple):
        pred, label = item
    else:
        raise ValueError(f"Unknown type of item: {type(item)}")

    if isinstance(label, str):
        if pred == label:
            return True
        return False
    if isinstance(label, int):
        if pred and ord(pred) - ord("A") == label:
            return True
        return False

    raise ValueError(f"Unknown type of label: {type(item['label'])}")


def prompt_fill_aligner(prompt_file: str, mapping: Dict[str, str], prompt_field: str = "prompt"):
    full_prompt = open(prompt_file).read()

    def func(data: List[Dict]):
        for item in data:
            prompt = copy.deepcopy(full_prompt)
            for k, v in mapping.items():
                prompt = prompt.replace(k, item[v])
            item[prompt_field] = prompt

        return data

    return func


def value2pair_aligner(field: str, pos_field: str, neg_field: str, value_field: str):
    def func(data: List[Dict]):
        for item in data:
            pair_data = item.pop(field)
            values = item.pop(value_field)

            pos = []
            neg = []
            for x, v in zip(pair_data, values):
                if v:
                    pos.append(x)
                else:
                    neg.append(x)
            item[pos_field] = pos
            item[neg_field] = neg

        return data

    return func


def return_threshold_mapping(value_threshold: float):
    def func(v):
        if v >= value_threshold:
            return True
        return False

    return func


def return_binary_mapping():
    def func(v):
        if v is True:
            return 1
        return 0

    return func


def value_mapping_aligner(value_field: str, value_mapping_func: Callable, return_int: bool = False):
    def func(data: List[Dict]):
        for item in data:
            if return_int:
                item[value_field] = int(value_mapping_func(item[value_field]))
            else:
                item[value_field] = value_mapping_func(item[value_field])
        return data

    return func


def dpo_pair2value_aligner(pos_field: str, neg_field: str, seq_field: str, value_field: str, flatten: bool = True):
    def func(data: List[Dict]):
        outputs = []
        for item in data:
            pos = item.pop(pos_field)
            neg = item.pop(neg_field)
            if flatten:
                item_a = copy.deepcopy(item)
                item_a[seq_field] = pos
                item_a[value_field] = 1

                item_b = copy.deepcopy(item)
                item_b[seq_field] = neg
                item_b[value_field] = 0

                outputs.append(item_a)
                outputs.append(item_b)
            else:
                item[seq_field] = [pos, neg]
                item[value_field] = [1, 0]
                outputs.append(item)

        return outputs

    return func


def value2pair_mapping_aligner(field: str, pos_field: str, neg_field: str, value_field: str, value_mapping_func: Callable):
    def func(data: List[Dict]):
        for item in data:
            pair_data = item.pop(field)
            values = item.pop(value_field)

            pos = []
            neg = []
            for x, v in zip(pair_data, values):
                if value_mapping_func(v):
                    pos.append(x)
                else:
                    neg.append(x)
            item[pos_field] = pos
            item[neg_field] = neg

        return data

    return func


def dpo_random_choice_aligner(anchor_field: str, paired_field: str):
    def func(data: List[Dict]):
        outputs = []
        for item in tqdm(data, desc="DPO random choice aligner", total=len(data)):
            if len(item[anchor_field]) == 0:
                continue
            if len(item[paired_field]) == 0:
                continue
            for anchor in item[anchor_field]:
                new_item = copy.deepcopy(item)
                new_item[anchor_field] = anchor
                new_item[paired_field] = random.choice(item[paired_field])
                outputs.append(new_item)
        return outputs

    return func


def dpo_flat_random_choice_aligner(paired_field: str):
    def func(data: List[Dict]):
        outputs = []
        for item in tqdm(data, desc="DPO random choice aligner", total=len(data)):
            if len(item[paired_field]) == 0:
                continue
            new_item = copy.deepcopy(item)
            new_item[paired_field] = random.choice(item[paired_field])
            outputs.append(new_item)
        return outputs

    return func


def dpo_paired_random_choice_aligner(anchor_field: str, paired_field, sort_accord_to_len: bool = False, top_k: int = 5, num_workers: int = 16):
    def func(data: List[Dict]):
        outputs = []
        for item in tqdm(data, desc="processing dpo pairs", total=len(data)):
            if len(item[anchor_field]) == 0:
                continue
            if len(item[paired_field]) == 0:
                continue
            assert len(item[anchor_field]) == len(item[paired_field]), (item[anchor_field], item[paired_field])
            for anchor, targets in zip(item[anchor_field], item[paired_field]):
                if len(targets) == 0:
                    continue
                new_item = copy.deepcopy(item)
                new_item[anchor_field] = anchor
                new_item[paired_field] = random.choice(targets)
                outputs.append(new_item)

        return outputs

    def func_sorted(data: List[Dict]):
        outputs_before_sort = []
        for item in tqdm(data, desc="DPO paired random choice aligner", total=len(data)):
            if len(item[anchor_field]) == 0:
                continue
            if len(item[paired_field]) == 0:
                continue
            assert len(item[anchor_field]) == len(item[paired_field]), (item[anchor_field], item[paired_field])
            for anchor, targets in zip(item[anchor_field], item[paired_field]):
                if len(targets) == 0:
                    continue
                new_item = copy.deepcopy(item)
                new_item[anchor_field] = anchor
                new_item[paired_field] = targets
                outputs_before_sort.append(new_item)

        _annotate = partial(_sort_worker, _pos_field=anchor_field, _neg_field=paired_field, _top_k=top_k)
        with Pool(num_workers) as p:
            outputs_after_sort = list(tqdm(p.imap(_annotate, outputs_before_sort), total=len(outputs_before_sort)))

        for item in outputs_after_sort:
            item[paired_field] = random.choice(item[paired_field])

        return outputs_after_sort

    if sort_accord_to_len:
        return func_sorted

    return func


def sample_steps(response: str):
    lines = response.split("\n")
    lines = [line for line in lines if line.strip()]
    return len(lines)


def _sort_worker(item, _pos_field: str, _neg_field: str, _top_k: int = 5):
    anchor = item[_pos_field]
    targets = item[_neg_field]
    anchor_steps = sample_steps(anchor)

    sorted_targets = sorted(targets, key=lambda x: abs(anchor_steps - sample_steps(x)))
    item[_neg_field] = sorted_targets[:_top_k]
    return item


def dpo_bi_random_choice_aligner(pos_field: str, neg_field: str, sort_accord_to_len: bool = False, top_k: int = 5, num_workers: int = 16):
    def func(data: List[Dict]):
        outputs = []
        for item in tqdm(data, desc="DPO random choice aligner", total=len(data)):
            if len(item[pos_field]) == 0:
                continue
            if len(item[neg_field]) == 0:
                continue
            pos = random.choice(item[pos_field])
            neg = random.choice(item[neg_field])
            item[pos_field] = pos
            item[neg_field] = neg
            outputs.append(item)
        return outputs

    def func_sorted(data: List[Dict]):
        outputs_before_sort = []
        for item in tqdm(data, desc="DPO random choice aligner", total=len(data)):
            if len(item[pos_field]) == 0:
                continue
            if len(item[neg_field]) == 0:
                continue
            new_item = copy.deepcopy(item)
            new_item["pos"] = random.choice(item[pos_field])
            new_item["neg"] = item[neg_field]
            outputs_before_sort.append(new_item)

        _annotate = partial(_sort_worker, _pos_field="pos", _neg_field="neg", _top_k=top_k)
        with Pool(num_workers) as p:
            outputs_after_sort = list(tqdm(p.imap(_annotate, outputs_before_sort), total=len(outputs_before_sort)))

        for item in outputs_after_sort:
            item["neg"] = random.choice(item["neg"])

        return outputs_after_sort

    if sort_accord_to_len:
        return func_sorted

    return func
