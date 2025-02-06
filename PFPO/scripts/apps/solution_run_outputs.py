import copy
import json
import re
import sys
from argparse import ArgumentParser
from datasets import load_dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import os

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.apps.utils_execute import run_inference_process


def extract_solution_between_tags(text):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'<BEGIN>(.*?)<END>'

    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def extract_test_cases(text):
    # Regular expression to match the contents between <TEST INPUT X> and </TEST INPUT X>
    pattern = r'<TEST INPUT (\d+)>(.*?)</TEST INPUT \1>'

    # Find all matches in the text, the re.DOTALL flag allows . to match newline characters
    test_cases = re.findall(pattern, text, re.DOTALL)

    # Extract only the content part from the matches
    test_cases = [case[1].strip() for case in test_cases]

    return test_cases


def _worker(item):
    if isinstance(item["completion"], str):
        completions = [item["completion"]]
    else:
        completions = item["completion"]

    if item["input_output"]:
        item["input_output"] = json.loads(item["input_output"])

    solutions = []
    results = []
    full_results = []
    all_outputs = []
    all_errors = []
    for completion in completions:
        gen_solution = extract_solution_between_tags(completion)
        if gen_solution is None:
            solutions.append("")
            results.append(False)
            full_results.append([False] * 3)
            continue

        solutions.append(gen_solution)

        if not item["input_output"]:
            continue

        all_results = run_inference_process(item["input_output"], gen_solution, timeout=10, debug=False, return_output=True)
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
        # assert len(res) == len(outputs) == len(errors), (len(res), len(outputs), len(errors))

    item["pred"] = solutions if len(solutions) > 1 else solutions[0]
    if results:
        item["res"] = results
        item["full_res"] = full_results
        item["outputs"] = all_outputs
        item["errors"] = all_errors

    return item


def main():
    parser = ArgumentParser()
    parser.add_argument("--completion_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if os.path.exists(args.completion_file):
        if args.completion_file.endswith(".json"):
            data = json.load(open(args.completion_file))
        else:
            data = [json.loads(line) for line in open(args.completion_file).readlines()]
    else:
        data = []
        item_id2data_id = {}
        for file in glob(args.completion_file):
            print(file)
            tmp = [json.loads(line) for line in open(file).readlines()]
            for item in tmp:
                item_id = item["problem_id"]
                if item_id not in item_id2data_id:
                    item_id2data_id[item_id] = len(data)
                    data.append(item)
                else:
                    new_completions = item["completion"]
                    if isinstance(new_completions, str):
                        new_completions = [new_completions]

                    if isinstance(data[item_id2data_id[item_id]]["completion"], list):
                        data[item_id2data_id[item_id]]["completion"].extend(new_completions)
                    else:
                        data[item_id2data_id[item_id]]["completion"] = [data[item_id2data_id[item_id]]["completion"]] + new_completions

    for item in data:
        item["completion"] = list(set(item["completion"]))

    print(f"Total number of items: {len(data)}")

    missing = 0
    corr = 0
    corr_at_k = 0
    pbar = tqdm(data)

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
        else:
            missing += 1

    print(f"Missing: {missing / len(outputs)}")
    print(f"Correct: {corr / len(outputs)}")
    print(f"Correct at k: {corr_at_k / len(outputs)}")
    json.dump(outputs, open(args.output_file, "w"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
