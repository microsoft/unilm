import copy
import json
import re
import sys
from argparse import ArgumentParser
import json
from datasets import load_dataset

from tqdm import tqdm

sys.set_int_max_str_digits(0)

_vanilla_prompt = """You are an expert programmer. Here is a programming problem:

{}

------------------------

And here is one group of test case inputs:
```
{}
```
------------------------

Please carefully comprehend the process described in the problem, and simulate it **step-by-step** to obtain the expected outputs the above test case inputs.

**REMEMBER** to follow the format below to provide the outputs:
[BEGIN]
outputs
[END]
"""

_vanilla_prompt_2 = """You are an expert programmer. Here is a programming problem:

{}

------------------------

And here is one group of test case inputs:
```
{}
```
------------------------

Please carefully comprehend the process described in the problem, and derive the expected outputs the above test case inputs. You should follow the steps below:
(1) Simulate the algorithm/process described in the problem with the given inputs **step-by-step**, and record the step-level intermediate results.
(2) Derive the correct outcome.
(3) Put the final outputs in the format below:

[BEGIN]
*put your outputs here*
[END]

and return back it to me.
"""

_vanilla_prompt_3 = """You are an expert programmer. Here is a programming problem:

{}

------------------------

And here is one group of test case inputs:
```
{}
```
------------------------

Please carefully comprehend the process described in the problem, and derive the expected outputs the above test case inputs. You should follow the steps below:
(1) Simulate the algorithm/process described in the problem with the given inputs **step-by-step**, and record the step-level intermediate results.
(2) Derive the correct outcome.

Response format:
[BEGIN]
*The expected outputs*
[END]
"""

PROMPTS = {
    "vanilla": _vanilla_prompt,
    "vanilla_2": _vanilla_prompt_2,
    "vanilla_3": _vanilla_prompt_3,
}


def main():
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="vanilla")
    args = parser.parse_args()

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
                new_item = copy.deepcopy(item)
                prompt = PROMPTS[args.prompt_type].format(problem, case)
                new_item["prompt"] = prompt
                f.write(json.dumps(new_item) + "\n")


if __name__ == '__main__':
    main()

"""
"""
