import json
import sys
from argparse import ArgumentParser

from datasets import load_dataset
from tqdm import tqdm

sys.set_int_max_str_digits(0)

_vanilla_prompt = """You are an expert programmer. Here is a programming problem:
--------
{}
--------
Please carefully comprehend the requirements in the problem, and write down the solution program to pass it under the given time and memory constraints.

**REMEMBER** to strictly follow the steps below to help reduce the potential flaws:
(1) According to the input scale and the time/memory constraints, think about the time complexity and space complexity of your solution.
(2) Think **step-by-step** to design the algorithm.
(3) Translate your thoughts into Python program to solve it.

Besides, your Python solution program should be located between <BEGIN> and <END> tags. For example:
<BEGIN>
t = int(input())
...
print(ans)
<END>

Now, remember my instruction and let's get started:
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

    with open(args.output_file, "w") as f:
        for item in tqdm(data, desc="Writing prompts"):
            prompt = PROMPTS[args.prompt_type].format(item["question"])
            item["prompt"] = prompt
            f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()
