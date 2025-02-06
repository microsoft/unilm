import sys
import json
import os
import argparse

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.apps import APPsWithFunctionName


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    # reader = APPsWithFunctionName(split="test", use_starter_code=True)
    reader = APPsWithFunctionName(split="train", train_sub_split="train", use_starter_code=True)

    data = reader("codeparrot/apps")

    # prompt_template = "{question}\n\nPlease write a program to solve the above problem under the given time constraints and memory limits."
    prompt_template = "{question}\n\nPlease write a Python program to solve the above problem under the given time constraints and memory limits."

    inputs = []
    for item in data:
        prompt = prompt_template.format(question=item["question"])
        item["prompt"] = prompt
        inputs.append(item)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in inputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(len(inputs))


if __name__ == "__main__":
    main()
