import copy
import json
import re
import sys
from argparse import ArgumentParser
from datasets import load_dataset
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import os

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from apps.utils_execute import check_correctness


def _worker(item):
    if not item["pred"]:
        if "res" in item:
            item.pop("res")
        if "full_res" in item:
            item.pop("full_res")
        return item

    codes = item["pred"]
    if isinstance(codes, str):
        codes = [codes]

    results = []
    full_results = []
    for gen_solution in codes:
        if not gen_solution:
            results.append(False)
            full_results.append([False] * 3)
            continue

        res = check_correctness(item["pseudo_input_output"], gen_solution, timeout=10, debug=False, return_output=False)
        for tmp in res:
            if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)):
                print(tmp, tmp.__class__.__name__)
        res = [bool(tmp) if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)) else tmp for tmp in res]
        if all(item is True for item in res) is True:
            results.append(True)
        else:
            results.append(False)
        full_results.append(res)

    item["res"] = results
    item["full_res"] = full_results

    return item


def main():
    """
    This file require the input file is in json format and has `pred` field to annotate the corresponding program solution.
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("--completion_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pseudo_test_cases", type=str)
    args = parser.parse_args()

    if os.path.exists(args.completion_file):
        print(args.completion_file)
        if args.completion_file.endswith(".json"):
            data = json.load(open(args.completion_file))
        else:
            data = [json.loads(line) for line in open(args.completion_file).readlines()]
    else:
        data = []
        for file in glob(args.completion_file):
            print(file)
            if file.endswith(".json"):
                data += json.load(open(file))
            else:
                data += [json.loads(line) for line in open(file).readlines()]

    pseudo_test_cases = json.load(open(args.pseudo_test_cases))
    pseudo_test_cases = {item["problem_id"]: item for item in pseudo_test_cases}
    print(len(pseudo_test_cases))
    print(list(pseudo_test_cases.keys())[:10])
    aux_ps_test_cases = {f"apps-train-{k}": v for k, v in pseudo_test_cases.items()}
    pseudo_test_cases.update(aux_ps_test_cases)

    new_data = []
    for item in data:
        if item["id"] in pseudo_test_cases:
            item["pseudo_input_output"] = pseudo_test_cases[item["id"]]["input_output"]
            new_data.append(item)
    data = new_data

    print(f"Total number of items: {len(data)}")

    missing = 0
    corr = 0
    corr_at_k = 0
    pbar = tqdm(data)
    cnt = Counter()

    outputs = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for _input in pbar:
            future = executor.submit(_worker, _input)
            futures.append(future)
            pbar.update()

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            outputs.append(future.result())

    for item in outputs:
        if "res" in item:
            if item["res"][0] is True:
                corr += 1
            if any(item["res"]):
                corr_at_k += 1
            cnt.update(item["res"])
        else:
            missing += 1

    print(f"Missing: {missing / len(outputs)}")
    print(f"Correct: {corr / len(outputs)}")
    print(f"Correct at k: {corr_at_k / len(outputs)}")
    print(cnt)
    json.dump(outputs, open(args.output_file, "w"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/solution_fail_extract_pseudo_label.py \
    --completion_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.?-of-8.v1.1.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.pseudo_test_case.exec.json \
    --num_workers 24 --pseudo_test_cases outputs/apps/apps.train.r2c.vanilla.gpt-4o.tem1.0.n11.pseudo_test_cases.json

>>> python scripts/apps/solution_fail_extract_pseudo_label.py \
    --completion_file "../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.?-of-8.v1.0.json" \
    --output_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v1.0.pseudo_test_case.exec.sc.json \
    --num_workers 24 --pseudo_test_cases outputs/apps/apps.train.r2c.vanilla.gpt-4o.tem1.0.n11.pseudo_test_cases.json 
    
>>> python scripts/apps/solution_fail_extract_pseudo_label.py \
    --completion_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.?-of-8.v1.1.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.self_s43_pseudo_cases.exec.json \
    --num_workers 24 \
    --pseudo_test_cases ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.s43.run_outputs.pseudo_cases.sc.json

Missing: 0.0
Correct: 0.37857900318133614
Correct at k: 0.4721102863202545
Counter({False: 30141, True: 17009})

>>> 
"""
