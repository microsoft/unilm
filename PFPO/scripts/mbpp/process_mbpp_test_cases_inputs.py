import os.path

from datasets import load_dataset
import argparse
import json
from tqdm import tqdm
from glob import glob

import re


def extract_test_case_inputs(text):
    # Regular expression to match content between <TEST CASE INPUTS> and </TEST CASE INPUTS>
    pattern = r'<TEST CASE INPUTS>(.*?)</TEST CASE INPUTS>'

    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = []
    if os.path.exists(args.input_file):
        data.extend([json.loads(line) for line in open(args.input_file).readlines()])
    else:
        for file in glob(args.input_file):
            data.extend([json.loads(line) for line in open(file).readlines()])

    outputs = {}
    for item in data:
        cases = extract_test_case_inputs(item["completion"])
        if cases is not None:
            cases = cases.split("\n")
            cases = [case.strip() for case in cases if case.strip()]
            cases = [case for case in cases if "==" not in case]
            cases = [case for case in cases if "assert " in case]

            if item["task_id"] not in outputs:
                outputs[item["task_id"]] = {
                    "source_file": item["source_file"],
                    "task_id": item["task_id"],
                    "code": item["code"],
                    "test_imports": item["test_imports"],
                    "test_list": item["test_list"],
                }
                outputs[item["task_id"]]["aug_cases"] = set()

            outputs[item["task_id"]]["aug_cases"].update(cases)

    cnt = 0
    for item_id in outputs:
        outputs[item_id]["aug_cases"] = list(outputs[item_id]["aug_cases"])
        cnt += len(outputs[item_id]["aug_cases"])

    print(f"Averaged {cnt/len(outputs)} test cases per task.")

    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
