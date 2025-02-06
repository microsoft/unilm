import copy
import json
import re
import sys
from argparse import ArgumentParser
import json
from datasets import load_dataset

from tqdm import tqdm

sys.set_int_max_str_digits(0)


def main():
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, default="vanilla")
    args = parser.parse_args()

    _prompt_template = open(args.prompt_file).read()

    data = load_dataset("codeparrot/apps", split=args.split).to_list()
    print(len(data))

    inputs = []
    for item in data:
        if item["input_output"]:
            item["input_output"] = json.loads(item["input_output"])
            inputs.append(item)

    with open(args.output_file, "w") as f:
        for item in tqdm(inputs, total=len(inputs)):
            problem = item["question"]
            for case in item["input_output"]["inputs"]:
                # assert isinstance(case, str), (problem, case, json.loads(item["solutions"])[0])  # For leetcode-format, this is not applied.
                new_item = copy.deepcopy(item)
                prompt = _prompt_template.replace("<<PROBLEM>>", problem).replace("<<TEST INPUTS>>", str(case))
                new_item["prompt"] = prompt
                f.write(json.dumps(new_item) + "\n")


if __name__ == '__main__':
    main()

"""
"""
