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

from scripts.apps.utils_execute import check_correctness


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
    solution = item["solutions"][0]

    all_results = check_correctness(item["input_output"], solution, timeout=10, debug=False, return_output=True)
    res, outputs, errors = all_results
    # res = all_results
    for tmp in res:
        if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)):
            print(tmp, tmp.__class__.__name__)
    res = [bool(tmp) if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)) else tmp for tmp in res]
    if all(item is True for item in res) is True:
        result = True
    else:
        result = False

    return result, res


def main():
    parser = ArgumentParser()
    parser.add_argument("--test_case_file", type=str)
    parser.add_argument("--test_case_field", type=str)
    parser.add_argument("--test_case_id_field", type=str, default="problem_id")
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    data = load_dataset("codeparrot/apps", split="train", trust_remote_code=True).to_list()
    print(f"Total number of items before: {len(data)}")
    train_sub_val_ids = set(json.load(open("apps_train_sub_val_ids.json")))

    data = [item for item in data if item["problem_id"] not in train_sub_val_ids]
    print(len(data))

    test_cases = json.load(open(args.test_case_file))
    # id2test_cases = {item[args.test_case_id_field]: item[args.test_case_field] for item in test_cases}
    missing = 0
    repeat = 0
    id2test_cases = {}
    for item in test_cases:
        if item[args.test_case_id_field] in id2test_cases:
            repeat += 1
        if args.test_case_field not in item:
            missing += 1
            continue
        id2test_cases[item[args.test_case_id_field]] = item[args.test_case_field]
    print(f"Missing: {missing}")
    print(f"Repeat: {repeat}")

    missing_solutions = 0
    missing_test_cases = 0
    new_data = []
    for item in data:
        if item["solutions"]:
            item["solutions"] = json.loads(item["solutions"])
            assert isinstance(item["solutions"], list)
            if len(item["solutions"]) > 0:
                if item["problem_id"] in id2test_cases:
                    item["input_output"] = id2test_cases[item["problem_id"]]
                    new_data.append(item)
                elif f"apps-train-{item['problem_id']}" in id2test_cases:
                    item["input_output"] = id2test_cases[f"apps-train-{item['problem_id']}"]
                    new_data.append(item)
                else:
                    missing_test_cases += 1
            else:
                missing_solutions += 1
        else:
            missing_solutions += 1

    print(f"Missing solutions: {missing_solutions}")
    print(f"Missing test cases: {missing_test_cases}")
    data = new_data

    print(f"Total number of items: {len(data)}")

    missing = 0
    corr = 0
    micro_corr = 0
    micro_all = 0
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
        result, full_res = item
        if result:
            corr += 1
        for r in full_res:
            if r:
                micro_corr += 1
            micro_all += 1

    print(f"Missing: {missing} / {len(outputs)} = {missing / len(outputs)}")
    print(f"Correct: {corr} / {len(outputs)} = {corr / len(outputs)}")
    print(f"Micro Correct: {micro_corr} / {micro_all} = {micro_corr / micro_all}")
    json.dump(outputs, open(args.output_file, "w"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
