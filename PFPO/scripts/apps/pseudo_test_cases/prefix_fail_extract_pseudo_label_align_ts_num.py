import json
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
from glob import glob

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from scripts.apps.utils_execute import check_correctness

"""
This script is used to control the amount of pseudo test cases are not more than the amount of ground truth test cases.
"""


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
    parser.add_argument("--gd_test_case_field", type=str, default="test_cases")
    args = parser.parse_args()

    print("CPU Cores:", os.cpu_count())

    if os.path.exists(args.completion_file):
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

    print(len(data))
    pseudo_test_cases = json.load(open(args.pseudo_test_cases))
    pseudo_test_cases = {item["id"]: item for item in pseudo_test_cases}

    new_data = []
    num_test_cases = 0
    for item in data:
        problem_id, resp_id, prefix_id = item["prefix_id"].split("_")
        problem_id = int(problem_id)
        item["problem_id"] = problem_id
        if problem_id in pseudo_test_cases:
            ps_test_cases = pseudo_test_cases[problem_id]["pseudo_test_cases"]
            if not item[args.gd_test_case_field]:
                continue
            if len(ps_test_cases["inputs"]) > len(item[args.gd_test_case_field]["inputs"]):
                idx = list(range(len(ps_test_cases["inputs"])))
                random.shuffle(idx)
                num_test_cases = len(item[args.gd_test_case_field]["inputs"])
                ps_test_cases["inputs"] = [ps_test_cases["inputs"][i] for i in idx[:num_test_cases]]
                ps_test_cases["outputs"] = [ps_test_cases["outputs"][i] for i in idx[:num_test_cases]]
            item["pseudo_input_output"] = ps_test_cases
            num_test_cases += len(item[args.gd_test_case_field]["inputs"])
            new_data.append(item)
    data = new_data

    print(f"Total number of items: {len(data)}")
    if len(data) == 0:
        print("No data to process.")
        return
    print(f"Average number of test cases: {num_test_cases / len(data)}")

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
>>> python scripts/apps/pseudo_test_cases/prefix_fail_extract_pseudo_label.py \
    --completion_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n5.163-of-256.v2.0.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.163-of-256.pseudo_test_case.exec.json \
    --num_workers 24 \
    --pseudo_test_cases ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json

Missing: 0.0
Correct: 0.467940813810111
Correct at k: 0.6097410604192355
Counter({False: 4524, True: 3586})

>>> python scripts/apps/pseudo_test_cases/prefix_fail_extract_pseudo_label.py \
    --completion_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n5.{split_id}-of-256.v2.0.json \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.{split_id}-of-256.pseudo_test_case.exec.json \
    --num_workers 64 \
    --pseudo_test_cases ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json
"""
