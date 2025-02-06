from datasets import load_dataset
import argparse
import json

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

Please design **10** test cases for the task above. Please **STRICTLY** follow the following steps to complete this:
(1) Generate a **step-by-step** guideline of the algorithm or task for efficient and accurate simulation.
(2) Write down one group of inputs.
(3) Simulate the algorithm described by the task **following your guideline**, and derive the expected results.
(4) Write down the expected results.
(5) Repeat step (2) - (4) for **10** times.
(6) Put all the test cases into the following format:

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

**REMEMBER**: Please first generate enough test cases following step (1) - (5), and then organize all the test cases in the above assertion format and return back them to me.
"""


PROMPTS = {
    "vanilla": _vanilla_prompt,
    "guideline": _guideline_prompt,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitized", default=False, action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="vanilla")
    args = parser.parse_args()

    if args.sanitized:
        dataset = load_dataset("mbpp", "sanitized", split="test").to_list()
    else:
        dataset = load_dataset("mbpp", split="test").to_list()
    prompt_key = "prompt" if args.sanitized else "text"

    outputs = []
    for item in dataset:
        task_id = item["task_id"]
        query = item[prompt_key]
        prompt = PROMPTS[args.prompt_type].format(query)
        outputs.append({
            "prompt": prompt,
            "task_id": task_id,
        })

    with open(args.output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
