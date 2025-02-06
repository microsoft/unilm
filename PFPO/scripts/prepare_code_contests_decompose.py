import argparse
import copy
import json

from datasets import load_dataset

_decompose_judge_template = """I will show you a competitive programming problem as well as its one solution. Note that the solution can be correct or incorrect. Please:
(1) Use **natural language** to clarify the thoughts behind the solution. You may consider including: the limitation about running time and memory, potential data structure, potential algorithms, etc. Please put your thoughts in <THOUGHTS></THOUGHTS>.
(2) After that, please split the above program solution into several segments. Each segment should follow the following format:
<SEGMENT 1>
<CODE SEGMENT>
*One segment of code from the original solution. It has to be a continuous part of the original solution, and should implement ONE basic and meaningful function.*
</CODE SEGMENT>
<NL DESCRIPTION>
*The corresponding description or interpretation of the above code snippet/segment. You should also include the clarification about the definition of the involved variables.*
</NL DESCRIPTION>
<WRAPPED CODE SEGMENT>
*According to the original code solution, the split code snippet, and the corresponding sub-clarification about the code segment, please wrap the code snippet into a function. The function should be a standalone function that can be executed independently. You should also include necessary docstring/comments in the function to define the input variables and return values.*
</WRAPPED CODE SEGMENT>
</SEGMENT 1>
<SEGMENT 2>
*Same structure as SEGMENT 1*
</SEGMENT 2>

Now, let's get started:

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
    parser.add_argument("--sample_num", default=-1, type=int, help="Number of samples to generate")
    args = parser.parse_args()

    data = load_dataset("parquet", data_files=args.file_path)["train"]
    outputs = []
    if args.sample_num > 0:
        sample_num = args.sample_num
    else:
        sample_num = len(data)
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
        pos_item["solution"] = pos_solution
        pos_item["prompt"] = _decompose_judge_template.format(title, description, time_limits, mem_limits,
                                                              pos_solution)
        pos_item["passed"] = True
        outputs.append(pos_item)

        neg_item = copy.deepcopy(item)
        neg_item["solution"] = neg_solution
        neg_item["prompt"] = _decompose_judge_template.format(title, description, time_limits, mem_limits,
                                                              neg_solution)
        neg_item["passed"] = False
        outputs.append(neg_item)

        if len(outputs) >= sample_num:
            break

    if args.sample_num > 0:
        args.output_file = args.output_file.replace(".jsonl", f"_{sample_num}.jsonl")

    with open(args.output_file, "w", encoding="utf8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
