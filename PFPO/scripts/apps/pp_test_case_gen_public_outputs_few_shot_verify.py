import json
import os
import re
import sys
from argparse import ArgumentParser
from collections import Counter

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.apps.utils_execute import run_test, check_correctness


def extract_content_between_tags(text):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'\[BEGIN\](.*?)\[END\]'

    match = re.search(pattern, text, re.DOTALL)
    if match:
        # return match.group(1).strip()
        return match.group(1)
    else:
        return None


def main():
    parser = ArgumentParser()
    parser.add_argument("--completion_file", type=str, required=True)
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.completion_file).readlines()]
    print(len(data))

    flag = "Now, let's get started with the following problem:"

    problem_pass_cnt = Counter()
    problem_case_cnt = Counter()
    difficulty_cnt = Counter()
    for item in tqdm(data, total=len(data)):
        # if item["problem_id"] > 2000:
        #     continue
        # if item["difficulty"] not in ["introductory", "interview"]:
        #     continue

        difficulty_cnt[item["difficulty"]] += 1
        print(f"=========================={item['problem_id']}")
        # if item['problem_id'] in [160, 379, 438, 4096]:
        #     print(f"Skip problem {item['problem_id']}")
        #     continue
        prompt = item["prompt"]
        assert flag in prompt, prompt
        prompt = prompt.split(flag)[1].strip()

        if "## TEST INPUTS\n\n" not in prompt:
            continue
        if "\n\n## SIMULATE" not in prompt:
            continue

        inputs_s = prompt.index("## TEST INPUTS") + len("## TEST INPUTS\n\n")
        inputs_e = prompt.index("\n\n## SIMULATE")
        # inputs = prompt[inputs_s:inputs_e].strip()
        inputs = prompt[inputs_s:inputs_e]
        # print(inputs)

        outputs = extract_content_between_tags(item["completion"])
        problem_case_cnt[item["problem_id"]] += 1

        if outputs is None:
            continue

        mapping = {str(_input): _input for i, _input in enumerate(item["input_output"]["inputs"])}

        # if "fn_name" in item["input_output"]:
        #     _cases = {
        #         "inputs": []
        #     }
        assert inputs in mapping, (inputs, json.dumps(mapping))
        _cases = {
            # "inputs": [inputs],
            # "outputs": [outputs]
            "inputs": [mapping[inputs]],
            "outputs": [outputs]
        }
        if "fn_name" in item["input_output"]:
            _cases["fn_name"] = item["input_output"]["fn_name"]

        solutions = json.loads(item["solutions"])
        for solution in solutions[:1]:  # I think one positive solution is enough
            try:
                # res = run_test(_cases, solution)
                res = check_correctness(_cases, solution, timeout=10, debug=False)
                pass_num = sum([1 for item in res if item is True])
                problem_pass_cnt[item["problem_id"]] += pass_num
            except:
                pass

    print(difficulty_cnt)

    print(f"Averaged pass cnt: {sum(problem_pass_cnt.values()) / sum(problem_case_cnt.values())}")
    print(f"Absolute pass rate over all generated cases: {sum(problem_pass_cnt.values()) / sum(problem_case_cnt.values())}")


if __name__ == '__main__':
    main()

"""
"""
