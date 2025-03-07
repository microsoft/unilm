import sys
import json
import os
import argparse

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.human_eval import MBPPReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    reader = MBPPReader()
    data = reader()

    prompt_template = "You are an expert Python programmer, and here is your task: {prompt}\nYour code should pass these tests:\n\n{test_list}\n\n"

    inputs = []
    for item in data:
        prompt = prompt_template.format(**item)
        item["prompt"] = prompt
        inputs.append(item)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in inputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(len(inputs))


if __name__ == "__main__":
    main()
