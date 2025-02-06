import copy
import json
import re
import sys
from argparse import ArgumentParser
from datasets import load_dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from glob import glob
import os
from pympler import asizeof

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from apps.utils_execute import run_inference_process


def _worker(item, test_case_field: str):
    results = []
    full_results = []
    all_outputs = []
    all_errors = []
    if not item["pred"]:
        if "res" in item:
            item.pop("res")
        if "full_res" in item:
            item.pop("full_res")
        if "outputs" in item:
            item.pop("outputs")
        if "errors" in item:
            item.pop("errors")
        return item

    for pred in item["pred"]:
        gen_solution = pred
        if gen_solution is None:
            results.append(False)
            # full_results.append([False] * 3)
            full_results.append([-2] * 21)
            all_outputs.append([None] * 21)
            continue

        if "Hello, World!" in gen_solution:
            results.append(False)
            full_results.append([-2] * 21)
            all_outputs.append([None] * 21)
            continue

        if not item[test_case_field]:
            continue

        all_results = run_inference_process(item[test_case_field], gen_solution, timeout=1, debug=False, return_output=True)
        res, outputs, errors = all_results
        for tmp in res:
            if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)):
                print(tmp, tmp.__class__.__name__)
        res = [bool(tmp) if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)) else tmp for tmp in res]
        if all(item is True for item in res) is True:
            results.append(True)
        else:
            results.append(False)
        full_results.append(res)
        try:
            json.dumps(outputs)
            all_outputs.append(outputs)
            all_errors.append(errors)
        except:
            print(f"Cannot dump outputs for {outputs}")
            all_outputs.append([])
            all_errors.append([])

    if results:
        item["res"] = results
        item["full_res"] = full_results
        item["outputs"] = all_outputs
        item["errors"] = all_errors

    return item


# Function to check if a string contains surrogate characters
def has_surrogate_characters(text):
    return bool(re.search(r'[\ud800-\udfff]', text))


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


def merge_seed_sampled_data(data, id_field: str = "id"):
    id2data = {}
    for item in data:
        if isinstance(item["response"], str):
            print(f"Warning: {item[id_field]} has only one response. ---- {item['response']} \n\n {item['pred']}")
            item["response"] = [item["response"]]
            assert isinstance(item["pred"], str) or item["pred"] is None
            item["pred"] = [item["pred"]]

        if item[id_field] not in id2data:
            id2data[item[id_field]] = item
            continue

        tmp = id2data[item[id_field]]

        tmp["response"] = merge_key(tmp["response"], item["response"])
        tmp["pred"] = merge_key(tmp["pred"], item["pred"])
        assert isinstance(tmp["pred"], list), tmp["pred"]
        id2data[item[id_field]] = tmp

    return list(id2data.values())


def main():
    parser = ArgumentParser()
    parser.add_argument("--completion_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument("--test_case_field", type=str, default="test_cases")
    args = parser.parse_args()

    data = load_files(args.completion_file)
    data = merge_seed_sampled_data(data, args.id_field)
    # for item in data:
    #     item["response"] = list(set(item["response"])) # No need to remove duplicates, since the `pred` field is aligned.

    print(f"Total number of items: {len(data)}")

    missing = 0
    corr = 0
    corr_at_k = 0
    pbar = tqdm(data)

    outputs = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        _annotate = partial(_worker, test_case_field=args.test_case_field)
        for _input in pbar:
            future = executor.submit(_annotate, _input)
            futures.append(future)
            pbar.update()

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            outputs.append(future.result())

    large_mem = 0
    for item in outputs:
        if "res" in item and item["res"]:
            if item["res"][0] is True:
                corr += 1
            if any(item["res"]):
                corr_at_k += 1

            try:
                size_in_bytes = asizeof.asizeof(item["outputs"])
                if size_in_bytes / (1024 ** 2) > 10:  # 10MB
                    if "res" in item:
                        item.pop("res")
                    if "full_res" in item:
                        item.pop("full_res")
                    if "outputs" in item:
                        item.pop("outputs")
                    if "errors" in item:
                        item.pop("errors")
                    large_mem += 1
            except:
                print("failed to compute size. Still abandon the outputs.")
                if "res" in item:
                    item.pop("res")
                if "full_res" in item:
                    item.pop("full_res")
                if "outputs" in item:
                    item.pop("outputs")
                if "errors" in item:
                    item.pop("errors")
                large_mem += 1
        else:
            missing += 1

    new_outputs = []
    for item in tqdm(outputs):
        tmp = json.dumps(item, ensure_ascii=False)
        if has_surrogate_characters(tmp):
            print(f"Surrogate characters found in {item[args.id_field]}")
            continue
        new_outputs.append(item)

    outputs = new_outputs
    print(f"Missing: {missing / len(outputs)}")
    print(f"Large memory: {large_mem / len(outputs)}")
    print(f"Correct: {corr / len(outputs)}")
    print(f"Correct at k: {corr_at_k / len(outputs)}")
    json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), ensure_ascii=False)


if __name__ == '__main__':
    main()

"""
>>>  python scripts/apps/solution_run_outputs_local.py --completion_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.s43.json --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.s43.run_outputs.json --num_workers 24

>>> python scripts/apps/solution_run_outputs_local.py --completion_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.0shot.tem1.0.n10.?-of-8.v2.0.json --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.0shot.tem1.0.n10.v2.0.run_outputs.json --num_workers 24 --id_field "problem_id" --test_case_field "input_output"

"""
