import argparse
import copy
import json

from datasets import load_dataset

_code_judge_template = """I will show you a programming problem. You will first be given the instruction about the program, and then I will show you the program implementation. Please determine if the completed program is correct or not. You can think step by step before determine its correctness. 

After your reasoning process, **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.

Instruction: {}

Program:
```
{}
```
"""


def main():
    parser = argparse.ArgumentParser(description='Completion Judge')
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    results = json.load(open(args.result_file))
    task_id2passed = {}
    for task_id, res in results.items():
        attempt = res[0]  # Pass@1
        attempt_res = attempt[1]
        passed = attempt_res["passed"]
        task_id2passed[task_id] = passed
    print(len(task_id2passed))

    predictions = json.load(open(args.prediction_file))
    print(len(predictions))
    outputs = []
    for item in predictions:
        if str(item["task_id"]) not in task_id2passed:
            continue

        orig_prompt = item["prompt"]
        new_prompt = _code_judge_template.format("{}", "{}", orig_prompt, item["completion"])
        item["orig_prompt"] = orig_prompt
        item["prompt"] = new_prompt
        item["passed"] = task_id2passed[str(item["task_id"])]  # FIXME: This is not aligned.
        outputs.append(item)

    with open(args.output_file, "w", encoding="utf8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
