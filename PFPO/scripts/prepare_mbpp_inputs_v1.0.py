from datasets import load_dataset
import argparse
import json


_vanilla_prompt = """You are an expert Python programmer, and here is your task: {}
Your code should pass these tests:

{}

Please put your code into [BEGIN]<Your Code>[END] block.
"""


_vanilla_prompt_emp = """You are an expert Python programmer, and here is your task: {}
Your code should pass these tests:

{}

**REMEMBER** to tag your solution program with the following format:
[BEGIN]
<Your Program>
[END]
and then return back it to me.
"""


_description_prompt = """You are an expert Python programmer, and here is your task: {}
Your code should pass these tests:

{}

**DO NOT** show me any code. Please use **NATURAL LANGUAGE** to describe your thoughts and implementation in details. Take care if your algorithm can pass the test cases by simulation.

**REMEMBER** to tag your description with the following format:
[BEGIN]
<Your Description>
[END]
and then return back it to me.
"""


PROMPTS = {
    "vanilla": _vanilla_prompt,
    "vanilla_emp": _vanilla_prompt_emp,
    "description": _description_prompt,
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
        test_cases = "\n".join(item["test_list"])
        prompt = PROMPTS[args.prompt_type].format(query, test_cases)
        outputs.append({
            "prompt": prompt,
            "task_id": task_id,
        })

    with open(args.output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
