import argparse
import copy
import json

from datasets import load_dataset

_code_judge_template = """I will show you a competitive programming problem. You will first be given the description of the problem, as well as the running time and memory limits. Then I will show you one program solution. Please determine if the completed program can pass the problem or not. You can think step by step before determine its correctness. 

After your reasoning process, **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.

Problem Title: {}

Description:
{}

Time Limits:
{} Seconds
Memory Limits:
{} Bytes

Solution Program:
```
{}
```
"""


def main():
    parser = argparse.ArgumentParser(description='Completion Judge')
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data = load_dataset("parquet", data_files=args.file_path)["train"]
    outputs = []
    for item in data:
        description = item["description"]
        title = item["name"]
        time_limits = item["time_limit"]["seconds"]
        mem_limits = item["memory_limit_bytes"]
        if len(item["solutions"]["solution"]) == 0 or len(item["incorrect_solutions"]["solution"]) == 0:
            continue

        pos_solution = item["solutions"]["solution"][0]
        neg_solution = item["incorrect_solutions"]["solution"][0]

        item = {
            "title": title,
            "description": description,
            "time_limits": time_limits,
            "mem_limits": mem_limits,
        }
        pos_item = copy.deepcopy(item)
        pos_item["prompt"] = _code_judge_template.format("{}", "{}", title, description, time_limits, mem_limits,
                                                         pos_solution)
        pos_item["passed"] = True
        outputs.append(pos_item)

        neg_item = copy.deepcopy(item)
        neg_item["prompt"] = _code_judge_template.format("{}", "{}", title, description, time_limits, mem_limits,
                                                         neg_solution)
        neg_item["passed"] = False
        outputs.append(neg_item)

    with open(args.output_file, "w", encoding="utf8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
