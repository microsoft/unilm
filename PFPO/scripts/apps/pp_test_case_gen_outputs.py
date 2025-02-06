import copy
import json
import re
import sys
from argparse import ArgumentParser

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


PROMPTS = {
    "vanilla": _vanilla_prompt,
    "vanilla_2": _vanilla_prompt_2,
}


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
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="vanilla")
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.input_file).readlines()]

    outputs = []
    for item in tqdm(data):
        cases = extract_test_cases(item["completion"])
        if len(cases):
            for i, case in enumerate(cases):
                new_item = copy.deepcopy(item)
                new_id = f"{item['problem_id']}_{i}"

                prompt = PROMPTS[args.prompt_type].format(item["question"], case)

                new_item["prompt"] = prompt
                new_item["case_id"] = new_id
                outputs.append(new_item)

    with open(args.output_file, "w") as f:
        for item in tqdm(outputs, desc="Writing prompts"):
            f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()

"""
python scripts/apps/pp_test_case_gen_outputs.py --input_file outputs/apps/apps.train.test_case_inputs.500.outputs.vanilla.v1.0.gpt-35-turbo.s42.jsonl --output_file outputs/apps/apps.train.test_case_outputs.500.vanilla.gpt-35-turbo.s42.v1.0.jsonl

python scripts/apps/pp_test_case_gen_outputs.py --input_file outputs/apps/apps.train.test_case_inputs.500.outputs.vanilla.v1.0.gpt-35-turbo.s42.jsonl --output_file outputs/apps/apps.train.test_case_outputs.500.vanilla.gpt-35-turbo.s42.v1.0.jsonl --prompt_type vanilla_2

python scripts/apps/pp_test_case_gen_outputs.py --input_file outputs/apps/apps.train.test_case_inputs.500.outputs.vanilla.v1.0.gpt-35-turbo.s42.jsonl --output_file outputs/apps/apps.train.test_case_outputs.500.vanilla.gpt-35-turbo.s42.v1.1.jsonl --prompt_type vanilla_2
"""
