import copy
import json
import re
import sys
from argparse import ArgumentParser
from datasets import load_dataset
from collections import defaultdict
import os

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from scripts.apps.utils_execute import run_test


def extract_content_between_tags(text):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'\[BEGIN\](.*?)\[END\]'

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


def main():
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--test_case_inputs", type=str, required=True, help="The completion file.")
    parser.add_argument("--test_case_outputs", type=str, required=True, help="The completion file.")
    # parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data = load_dataset("codeparrot/apps", split="train").to_list()
    print(len(data))
    # for item in data:  # Currently, we do not require the solutions and input_output fields
    #     if item["solutions"]:
    #         item["solutions"] = json.loads(item["solutions"])
    #     if item["input_output"]:
    #         item["input_output"] = json.loads(item["input_output"])

    case_inputs = [json.loads(line) for line in open(args.test_case_inputs).readlines()]
    case_outputs = [json.loads(line) for line in open(args.test_case_outputs).readlines()]

    problem_id2case = defaultdict(list)
    for item in case_inputs:
        cases = extract_test_cases(item["completion"])
        if len(cases):
            for i, case in enumerate(cases):
                problem_id2case[item["problem_id"]].append({
                    "input": case,
                    "case_id": i,
                })

    for item in case_outputs:
        output = extract_content_between_tags(item["completion"])
        if output:
            assert isinstance(output, str), output
            case_id = int(item["case_id"].split("_")[1])
            assert case_id == problem_id2case[item["problem_id"]][case_id]["case_id"]
            problem_id2case[item["problem_id"]][case_id]["output"] = output

    # Currently, we can directly verify
    outputs = []
    for item in data:
        if item["solutions"]:
            item["solutions"] = json.loads(item["solutions"])

            problem_id = item["problem_id"]
            cases = {
                "inputs": [],
                "outputs": [],
            }
            for case in problem_id2case[problem_id]:
                if "output" in case:
                    cases["inputs"].append(case["input"])
                    cases["outputs"].append(case["output"])
                    assert isinstance(case["input"], str), case["input"]
                    assert isinstance(case["output"], str), case["output"]
            if cases["inputs"]:
                item["gen_test_cases"] = cases
                outputs.append(item)

    # print(outputs[0]["gen_test_cases"])

    cnt = 0
    num_program = 0
    for i, item in tqdm(enumerate(outputs)):
        if i in [7]:
            continue  # Some bad cases.
        print(item["gen_test_cases"])
        for solution in item["solutions"]:
            try:
                res = run_test(item["gen_test_cases"], solution)
                pass_num = sum([1 for item in res if item is True])
                cnt += pass_num
            except:
                pass
            num_program += 1

    print(f"Pass rate: {cnt / num_program}")

    # print(json.dumps(data[0], indent=2))
    # print(data[0]["question"])
    # print(data[0]["solutions"][0])
    # print(data[0]["input_output"]["inputs"][0])
    # print(data[0]["input_output"]["outputs"][0])

    # with open(args.output_file, "w") as f:
    #     for item in tqdm(outputs, desc="Writing prompts"):
    #         f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()
