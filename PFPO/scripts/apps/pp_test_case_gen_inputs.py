import json
import sys
from argparse import ArgumentParser

from datasets import load_dataset
from tqdm import tqdm

sys.set_int_max_str_digits(0)

_vanilla_prompt = """You are an expert programmer. Here is a programming problem:
{}

Please carefully comprehend the requirements in the problem, and generate **10** more test cases for me. You can find some examples in the problem description.

**REMEMBER** (1) I only need the inputs, please do not show me the outputs. (2) Please follow the format below to provide the test cases:

<TEST INPUT 1>
input 1
</TEST INPUT 1>
<TEST INPUT 2>
input 2
</TEST INPUT 2>
...
<TEST INPUT 10>
input 10
</TEST INPUT 10>
"""

PROMPTS = {
    "vanilla": _vanilla_prompt,
}


def main():
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_type", type=str, default="vanilla")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data = load_dataset("codeparrot/apps", split="train").to_list()
    print(len(data))
    # for item in data:  # Currently, we do not require the solutions and input_output fields
    #     if item["solutions"]:
    #         item["solutions"] = json.loads(item["solutions"])
    #     if item["input_output"]:
    #         item["input_output"] = json.loads(item["input_output"])

    # print(json.dumps(data[0], indent=2))
    # print(data[0]["question"])
    # print(data[0]["solutions"][0])
    # print(data[0]["input_output"]["inputs"][0])
    # print(data[0]["input_output"]["outputs"][0])

    with open(args.output_file, "w") as f:
        for item in tqdm(data, desc="Writing prompts"):
            prompt = PROMPTS[args.prompt_type].format(item["question"])
            item["prompt"] = prompt
            f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()
