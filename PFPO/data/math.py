# Copied from https://github.com/meta-math/MetaMath/blob/main/eval_math.py
# and https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
import re
import sys
from typing import List, Dict

from fraction import Fraction
from data.math_util import is_equiv, last_boxed_only_string
from transformers import PreTrainedTokenizer
from general_util.logger import get_child_logger
from data.deepseek_math_utils.answer_extraction import extract_math_answer, extract_last_single_answer
import torch

logger = get_child_logger(__name__)

MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion, separator: str = "The answer is: "):
    text = completion.split(separator)
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def gsk8k_answer_cleaner(separator: str = "The answer is: "):
    def func(completion: str):
        res = extract_answer_number(completion, separator=separator)
        if res is None:
            return ""
        return str(res)  # To be compatible with `OpenAICallBack`

    return func


def number_answer_extractor(separator: str = "The answer is: ", completion_field: str = "response"):
    def func(data: List[Dict]):
        for item in data:
            res = extract_answer_number(item[completion_field], separator=separator)
            if res is None:
                item["label"] = ""
            else:
                item["label"] = str(res)
        return data

    return func


def gsm8k_gold_answer_extractor(response_field: str = "response"):
    def func(data: List[Dict]):
        for i, item in enumerate(data):
            temp_ans = item[response_field].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            item["label"] = str(temp_ans)
            item["index"] = i

        return data

    return func


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def math_gold_answer_extractor(response_field: str = "output", kv_mapping: Dict = None):
    def func(data: List[Dict]):
        for item in data:
            item["label"] = remove_boxed(last_boxed_only_string(item[response_field]))

            if kv_mapping is not None:  # "instruction" is maintained for composition. So the `instruction` key in MATH dataset should be changed.
                for k, v in kv_mapping.items():
                    item[v] = item.pop(k)

        return data

    return func


def math_boxed_answer_cleaner():
    def func(s):
        return remove_boxed(last_boxed_only_string(s))

    return func


def math_boxed_answer_cleaner_proxy():
    def func(question, reasoning, task):
        return remove_boxed(last_boxed_only_string(reasoning))

    return func


def math_gold_answer_extractor_deepseek(query_field: str = "instruction", response_field: str = "output", kv_mapping: Dict = None):
    def func(data: List[Dict]):
        for item in data:
            item["label"] = extract_math_answer(item[query_field], item[response_field], "cot")

            if kv_mapping is not None:  # "instruction" is maintained for composition. So the `instruction` key in MATH dataset should be changed.
                for k, v in kv_mapping.items():
                    item[v] = item.pop(k)

        return data

    return func


def gsm8k_gold_answer_extractor_deepseek(query_field: str = "instruction", response_field: str = "output", kv_mapping: Dict = None):
    def func(data: List[Dict]):
        for item in data:
            item["label"] = extract_last_single_answer(item[query_field], item[response_field], "cot")

            if kv_mapping is not None:
                for k, v in kv_mapping.items():
                    item[v] = item.pop(k)

        return data

    return func


# This is the original one from MetaMath repository.
def process_results(doc, completion, answer):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        # invalid_outputs.append(temp)
        return False


def math_answer_cleaner(separator: str = "The answer is: "):
    def func(completion: str):
        split_ans = completion.split(separator)
        if len(split_ans) > 1:
            ans = split_ans[-1]
            extract_ans_temp = ans.split('.\n')[0]
            extract_ans_temp = extract_ans_temp.strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip()

            if "$" in extract_ans:  # Added new filtering
                extract_ans = extract_ans.replace("$", "")

            if "=" in extract_ans:
                extract_ans = extract_ans.split('=')[-1].strip()

            return extract_ans
        else:
            return ""

    return func


def meta_math_gold_answer_extractor(response_field: str = "response"):
    cleaner = math_answer_cleaner(separator="The answer is: ")

    def func(data: List[Dict]):
        outputs = []
        cnt = 0
        for item in data:
            label = cleaner(item[response_field])
            if label:
                item["label"] = label
                outputs.append(item)
            else:
                cnt += 1
        logger.info(f"Counted {len(outputs)} items, {cnt} items are invalid")

        return outputs

    return func


def decompose_rap(prompt: str, response: str, max_seq_length: int, tokenizer: PreTrainedTokenizer):
    # raw_steps = response.strip().split("\n")
    raw_steps = response.split("\n")
    steps = [raw_steps[0]]
    for line in raw_steps[1:]:
        if line.replace("#", "").strip() == "":
            continue
        if not (line.startswith("SubQuestion ") or line.startswith("Answer ")):
            steps[-1] += "\n" + line
        else:
            steps.append(line)

    endings = []
    acc_step = prompt
    for i, step in enumerate(steps):
        if i == 0:
            acc_step = acc_step + step
        else:
            acc_step = acc_step + "\n" + step
        input_ids = tokenizer(acc_step, truncation=True, max_length=max_seq_length)["input_ids"]
        endings.append(len(input_ids) - 1)

    assert len(endings) > 0, (prompt, response)
    return endings


def decompose_cot(prompt: str, response: str, max_seq_length: int, tokenizer: PreTrainedTokenizer):
    steps = response.split("\n")

    endings = []
    acc_step = prompt
    for i, step in enumerate(steps):
        if i == 0:
            acc_step = acc_step + step
        else:
            acc_step = acc_step + "\n" + step
        input_ids = tokenizer(acc_step, truncation=True, max_length=max_seq_length)["input_ids"]
        endings.append(len(input_ids) - 1)

    assert len(endings) > 0, (prompt, response)
    return endings


def decompose_deepseek_math_cot_v2(prompt: str, response: str, max_seq_length: int, tokenizer: PreTrainedTokenizer):
    assert isinstance(prompt, str), prompt
    assert isinstance(response, str), response
    steps = response.split("\n")

    endings = []
    acc_step = prompt
    for i, step in enumerate(steps):
        if i == 0:
            acc_step = acc_step + step
        else:
            acc_step = acc_step + "\n" + step
        if step.strip():
            input_ids = tokenizer(acc_step, truncation=False)["input_ids"]
            endings.append(len(input_ids) - 1)

    full_text = prompt + response
    true_input_ids = tokenizer(full_text, truncation=True, max_length=max_seq_length)["input_ids"]
    endings = [e for e in endings if e < len(true_input_ids)]

    # assert len(endings) > 0, (prompt, response)
    if len(endings) == 0:
        logger.warning(f"Warning: Bad response:\n\n=========================Prompt====================\n{prompt}\n\n"
                       f"=======================Response=====================\n{response}\n\n")
    return endings


# def decompose_deepseek_math_cot_v2_aligner(tokenizer: PreTrainedTokenizer, max_seq_length: int, response_field: str, prompt_field: str):
#     def func(data: List[Dict]):
#         for item in data:
#             item["ending"] = decompose_deepseek_math_cot_v2(item[prompt_field], item[response_field], max_seq_length, tokenizer)
#         return data
#
#     return func


class RAPResponseStepRewardCollator:
    _decompose_fns = {
        "rap": decompose_rap,
        "cot": decompose_cot,
    }

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, cot_type: str = "rap"):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.decompose_fn = self._decompose_fns[cot_type]

    def __call__(self, batch):
        prompt = [item["prompt"] for item in batch]
        inputs = [item["input"] for item in batch]
        indices = [item["index"] for item in batch]

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
        labels[prompt_mask] = self.tokenizer.pad_token_id

        endings = []
        padding_len = torch.sum(1 - encoded_inputs["attention_mask"], dim=-1)
        for b, item in enumerate(batch):
            # ending = decompose_rap(item["prompt"], item["response"], self.max_seq_length, self.tokenizer)
            ending = self.decompose_fn(item["prompt"], item["response"], self.max_seq_length, self.tokenizer)
            if self.tokenizer.padding_side == "left":
                ending = [e + padding_len[b].item() for e in ending]
            endings.append(ending)

        encoded_inputs["labels"] = labels
        encoded_inputs["meta_data"] = {
            "index": indices,
            "prompt": prompt,
            "input": inputs,
            "response": [item["response"] for item in batch],
            "ending": endings,
            "type": [None] * len(endings),
        }
        return encoded_inputs
