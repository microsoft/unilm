import json
import argparse
import os
from glob import glob
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.set_int_max_str_digits(0)

from post_processors.code.clean import get as get_code_cleaner


def sample_step(response: str, code: str, upper_step_ratio: float, sample_ratio: float):
    if code:
        assert code in response, (code, "\n\n========================\n\n", response)
        end_of_response = response.find(code) + len(code)
        response = response[:end_of_response]

    orig_lines = response.split("\n")
    lines = [(i, line) for i, line in enumerate(orig_lines)]
    lines = [(i, line) for i, line in lines if line.strip()]

    if len(lines) < 5:
        return []

    upper_step = int(len(lines) * upper_step_ratio)
    if upper_step == 0:
        return []

    sample_step_num = int(upper_step * sample_ratio)
    if sample_step_num == 0:
        return []

    sample_steps = random.sample(lines[:upper_step], sample_step_num)
    if len(sample_steps) == 0:
        return []

    step_prefixes = []
    for i, line in sample_steps:
        step_prefixes.append("\n".join(orig_lines[:(i + 1)]))

    return step_prefixes


def load_files(file_path):
    data = []
    if os.path.exists(file_path):
        print(f"Loading pseudo test cases from {file_path}")
        if file_path.endswith(".json"):
            data.extend(json.load(open(file_path)))
        else:
            data.extend([json.loads(line) for line in open(file_path).readlines()])
    else:
        for file in glob(file_path):
            print(file)
            if file.endswith(".json"):
                data.extend(json.load(open(file)))
            else:
                data.extend([json.loads(line) for line in open(file).readlines()])

    return data


def merge_key(item, value):
    assert isinstance(item, list)
    if isinstance(value, list):
        item = item + value
    else:
        item.append(value)
    return item


def merge_seed_sampled_data(data):
    id2data = {}
    large_mem = 0
    for item in data:
        if isinstance(item["response"], str):
            item["response"] = [item["response"]]
            assert isinstance(item["pred"], str) or item["pred"] is None
            item["pred"] = [item["pred"]]

        if item["id"] not in id2data:
            id2data[item["id"]] = item
            continue

        tmp = id2data[item["id"]]

        tmp["response"] = merge_key(tmp["response"], item["response"])
        tmp["pred"] = merge_key(tmp["pred"], item["pred"])
        assert isinstance(tmp["pred"], list), tmp["pred"]
        id2data[item["id"]] = tmp

    print(f"Too large outputs: {large_mem}")
    return list(id2data.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, )
    parser.add_argument("--output_file", type=str, )
    parser.add_argument("--upper_step_ratio", type=float)
    parser.add_argument("--sample_ratio", type=float)
    parser.add_argument("--filter_all_correct", action="store_true")
    parser.add_argument("--sample_over_p", type=int, default=-1)
    args = parser.parse_args()

    data = load_files(args.input_file)
    data = merge_seed_sampled_data(data)

    outputs = []
    num = 0
    for item in data:
        if args.filter_all_correct and all(item["res"]):
            continue

        prefixes = []
        prefix_ids = []
        if not item["response"]:
            item["prefix"] = []
            continue

        for resp_id, resp in enumerate(item["response"]):
            response_prefixes = sample_step(resp, item["pred"][resp_id], args.upper_step_ratio, args.sample_ratio)
            prefixes.extend(response_prefixes)
            prefix_ids.extend([f"{item['id']}_{resp_id}_{i}" for i in range(len(response_prefixes))])

        if args.sample_over_p > 0:
            if len(prefixes) > args.sample_over_p:
                prefixes = random.sample(prefixes, args.sample_over_p)
                prefix_ids = random.sample(prefix_ids, args.sample_over_p)

        item["prefix"] = prefixes
        item["prefix_id"] = prefix_ids
        outputs.append(item)
        num += len(prefixes)

    json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Total number of samples: {num}")


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/prm/sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.?-of-4.v2.0.json" \
    --upper_step_ratio 0.8 --sample_ratio 0.3 \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.prefix.upper0.8.r0.3.json

Total number of samples: 470010
"""
