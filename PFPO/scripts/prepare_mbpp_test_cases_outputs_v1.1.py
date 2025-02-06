import copy

from datasets import load_dataset
import argparse
import json
from tqdm import tqdm

_vanilla_prompt = """You are an expert Python programmer, and here is your task: 

{}

Please design **10** test cases for the task above. Please **STRICTLY** follow the following steps to complete this:
(1) Write down one group of inputs.
(2) Simulate the algorithm described by the task, and derive the expected results.
(3) Write down the expected results.
(4) Repeat step (1) - (3) for **10** times.
(5) Put all the test cases into the following format:

[BEGIN]
assertion 1
assertion 2
...
assertion 10
[END]

where each assertion should follow the format:

assert <function>(inputs) == outputs

For example:
assert <function>(2, 3) == 5

Since you do not know the name of the function, please just use <function> as a placeholder.

**REMEMBER**: Please first generate enough test cases following step (1) - (4), and then organize all the test cases in the above assertion format and return back them to me.
"""

_guideline_prompt = """You are an expert Python programmer, and here is your task: 

{}

And here is **one** group of inputs:

{}

which is in the assertion format like:

assert <function>(inputs)

The outputs are unknown and waiting for your completion. <function> is the name of function and you should not change this.

Please simulate the running process of the task mentioned above and get the expected outputs by **STRICTLY** following the following steps:
(1) Generate a **step-by-step** guideline of the algorithm or task for efficient and accurate simulation.
(2) Select one group of inputs.
(3) Simulate the algorithm described by the task **following your guideline**, and derive the expected results.
(4) Write down the expected results.
(5) Put the input-output pair together as the following standard assertion format:

[BEGIN]
assert <function>(inputs) == outputs
[END]

Here is a example for assertion format:

assert <function>(2, 3) == 5

where <function> is the function, `2, 3` is the inputs, and `5` is the expected outputs.

**REMEMBER**: Please first obtain the expected outputs following step (1) - (4), and then reorganize the test case in the above assertion format and return back it to me.
"""

PROMPTS = {
    "vanilla": _vanilla_prompt,
    "guideline": _guideline_prompt,
}


def extract_test_case_inputs(item):
    cases = item["results"]
    inputs = []
    for case in cases:
        tmp = case["case"].split("==")[0].strip()
        inputs.append(tmp)

    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_case_file", type=str, required=True)
    parser.add_argument("--sanitized", default=False, action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="vanilla")
    args = parser.parse_args()

    data = json.load(open(args.test_case_file))
    if args.sanitized:
        dataset = load_dataset("mbpp", "sanitized", split="test").to_list()
    else:
        dataset = load_dataset("mbpp", split="test").to_list()
    prompt_key = "prompt" if args.sanitized else "text"
    task_id2orig_data = {item["task_id"]: item for item in dataset}

    outputs = []
    for item in tqdm(data):
        query = task_id2orig_data[item["task_id"]][prompt_key]
        case_inputs = extract_test_case_inputs(item)
        for x, case in enumerate(case_inputs):
            new_item = copy.deepcopy(item)
            prompt = PROMPTS[args.prompt_type].format(query, case)
            new_item["program_prompt"] = item["prompt"]
            new_item["program"] = item["completion"]
            new_item["prompt"] = prompt
            new_item["task_id"] = f"{item['task_id']}_{x}"
            new_item["case_input"] = case
            outputs.append(new_item)

    with open(args.output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
