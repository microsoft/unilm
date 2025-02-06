import json
import json
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from scripts.apps.utils_execute import check_correctness


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

    if any(x and y in x for x in item['pred'] for y in ["os.makedirs", "views.contact", "extract_metadata.py"]):
        print(f"======================================================== item {item['problem_id']} ====================================================")
        if "res" in item:
            item.pop("res")
        if "full_res" in item:
            item.pop("full_res")
        return item

    results = []
    full_results = []
    for gen_solution in codes:
        if not gen_solution:
            results.append(False)
            full_results.append([False] * 3)
            continue

        try:
            res = check_correctness(item["input_output"], gen_solution, timeout=10, debug=False, return_output=False)
        except Exception as e:
            print(f"======================================================== item {item['problem_id']} ====================================================")
            print(e)
            if "res" in item:
                item.pop("res")
            if "full_res" in item:
                item.pop("full_res")
            return item

        for tmp in res:
            if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)):
                print(tmp, tmp.__class__.__name__)
        new_res = []
        # res = [bool(tmp) if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)) else tmp for tmp in res]
        for tmp in res:
            try:
                if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)):
                    new_res.append(bool(tmp))
                else:
                    new_res.append(tmp)
            except Exception as e:
                print(e)
                new_res.append(False)
        res = new_res

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

    depreciated_ids = {"oss-instruct-9410"}

    pseudo_test_cases = json.load(open(args.pseudo_test_cases))
    pseudo_test_cases = {item["problem_id"]: item for item in pseudo_test_cases}

    new_data = []
    for item in data:
        problem_id, resp_id, prefix_id = item["prefix_id"].split("_")
        # problem_id = int(problem_id)
        item["problem_id"] = problem_id
        if problem_id in pseudo_test_cases:
            item["input_output"] = pseudo_test_cases[problem_id]["input_output"]
            new_data.append(item)
    data = new_data

    print(f"Total number of items: {len(data)}")
    data = [item for item in data if item["problem_id"] not in depreciated_ids]

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


"""
