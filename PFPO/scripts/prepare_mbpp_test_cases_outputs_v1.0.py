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

And here are several group of inputs here:

{}

where each group of inputs is in the assertion format like:

assert <function>(inputs)

The outputs are unknown and waiting for your completion. <function> is the name of function and you should not change this.

Please simulate the running process of the task and get the expected outputs by **STRICTLY** following the following steps:
(1) Generate a **step-by-step** guideline of the algorithm or task for efficient and accurate simulation.
(2) Select one group of inputs.
(3) Simulate the algorithm described by the task **following your guideline**, and derive the expected results.
(4) Write down the expected results.
(5) Repeat step (2) - (4) until you have got the expected outputs for all inputs.
(6) Put all the test cases (input-output pair) together into the following format:

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

**REMEMBER**: Please first obtain all the expected outputs following step (1) - (5), and then organize all the test cases in the above assertion format and return back them to me.
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

    return "\n".join(inputs)


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

    for item in tqdm(data):
        query = task_id2orig_data[item["task_id"]][prompt_key]
        case_inputs = extract_test_case_inputs(item)
        prompt = PROMPTS[args.prompt_type].format(query, case_inputs)
        item["program_prompt"] = item["prompt"]
        item["program"] = item["completion"]
        item["prompt"] = prompt

    with open(args.output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
