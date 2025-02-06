import argparse
import json

_decompose_judge_template = """I will show you a competitive programming problem as well as its one solution. To reduce the difficulty for your understanding, I will help you to split it into several segments, each segment is tagged with <SEGMENT X></SEGMENT X>, where X is the id, and it follows the following format:
-   <CODE SEGMENT>
    *One segment of code from the original solution. It is a continuous part of the original solution, and implements ONE basic and meaningful function.*
    </CODE SEGMENT>
-   <NL DESCRIPTION>
    *The corresponding description or interpretation of the above code snippet/segment. It may also includes the clarification about the definition of the involved variables.*
    </NL DESCRIPTION>
-   <WRAPPED CODE SEGMENT>
    *According to the original code solution, the split code snippet, and the corresponding sub-clarification about the code segment, the code is wrapped into a function as shown here to facilitate isolated checking. It is standalone that can be executed independently. And it also includes necessary docstring/comments in the function to define the input variables and return values.*
-   </WRAPPED CODE SEGMENT>.

Now, your task is to:
(1) Examine the correctness of each code segment. You can refer the implementation in wrapped function for isolated checking. According to the docstring, think about its role in the whole solution, and carefully check its internal correctness and if it necessary and suitable for solving the whole problem.
(2) According to your decisions to each of the segment, decide if the composed one, i.e., the original complete solution, can well solve the problem with the given test cases, time and memory limits. You can think step-by-step to think about the implementation. Remember that sometimes the overall solution can still be incorrect although the divided programs are all good, since you still need to examine the overall implementation logic, and the running time/memory limits.

Finally, after the above process, **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.

Now, let's get started:

Problem Title: {}

Description:
{}

Time Limits:
{} Seconds
Memory Limits:
{} Bytes

Original Solution Program:
```
{}
```

Divided Code Segments:
{}

Now, please start your examination.
"""

MAX_SEGMENT_NUM = 10


def completion_parsing(completion: str):
    segments = []
    for i in range(MAX_SEGMENT_NUM):
        s_str = f"<SEGMENT {i + 1}>"
        e_str = f"</SEGMENT {i + 1}>"
        if s_str not in completion:
            break

        s = completion.index(s_str)
        e = completion.index(e_str) + len(e_str)
        segments.append(completion[s:e])
    return segments


def main():
    parser = argparse.ArgumentParser(description='Completion Judge')
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    # parser.add_argument("--sample_num", default=-1, type=int, help="Number of samples to generate")
    args = parser.parse_args()

    # data = load_dataset("parquet", data_files=args.file_path)["train"]
    data = [json.loads(line) for line in open(args.file_path).readlines()]
    outputs = []
    # if args.sample_num > 0:
    #     sample_num = args.sample_num
    # else:
    #     sample_num = len(data)
    for item in data:
        description = item["description"]
        title = item["title"]
        time_limits = item["time_limits"]
        mem_limits = item["mem_limits"]
        # if len(item["solutions"]["solution"]) == 0 or len(item["incorrect_solutions"]["solution"]) == 0:
        #     continue

        # pos_solution = item["solutions"]["solution"][0]
        # neg_solution = item["incorrect_solutions"]["solution"][0]
        if "solution" in item:
            solution = item["solution"]
        else:
            s = item["prompt"].index("Solution Program:\n```")
            solution = item["prompt"][s + len("Solution Program:\n```"):-3]

        # segments = item["completion"]
        segments = completion_parsing(item["completion"])
        segments = "\n\n".join(segments)

        item = {
            "title": title,
            "description": description,
            "time_limits": time_limits,
            "mem_limits": mem_limits,
            "solution": solution,
            "segments": segments,
            "prompt": _decompose_judge_template.format("{}", "{}", title, description, time_limits,
                                                       mem_limits, solution, segments),
            "passed": item["passed"],
        }

        outputs.append(item)

        # if len(outputs) >= sample_num:
        #     break

    # if args.sample_num > 0:
    #     args.output_file = args.output_file.replace(".jsonl", f"_{sample_num}.jsonl")

    with open(args.output_file, "w", encoding="utf8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
