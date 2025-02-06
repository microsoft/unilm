import json
import argparse
import io
import sys
from tqdm import tqdm

_vanilla_code_judge_template = """I will show you a programming problem. You will first be given the instruction about the program, and then I will show you the program implementation. Please determine if the completed program is correct or not. You can think step by step before determine its correctness. 

After your reasoning process, **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.

Instruction: {}

Program:
```
{}
```
"""

_running_code_judge_template_0 = """I will show you a programming problem. You will first be given the instruction about the program, and then I will show you the program implementation. Please determine if the completed program is correct or not. Besides, in order to reduce the difficulty, I will also provide you with a version that the same program is equipped with print statements and several test cases to check the intermediate running variables.

After your reasoning process, **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.
Instruction: {}

Program:
```
{}
```

The program with necessary print statements and test cases to check intermediate variables:
```
{}
```

The corresponding execution results of the program with print statements and test cases:
```
{}
```

Now, please determine its correctness. You can think step by step, but **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.
"""


_running_code_judge_template_1 = """I will show you a programming problem. You will first be given the instruction about the program, and then I will show you the program implementation. Please determine if the completed program is correct or not. Besides, in order to reduce the difficulty, I will also provide you with a version that the same program is equipped with print statements and several test cases to check the intermediate running variables. You can thus verify the correctness of the program by (1) implement your own program following the instruction; (2) Simulate the running process of your program with the given test cases, and then check if the expected results are consistent with the given values (including those of the intermediate variables).

After your reasoning process, **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.

Instruction: {}

Program:
```
{}
```

The program with necessary print statements and test cases to check intermediate variables:
```
{}
```

The corresponding execution results of the program with print statements and test cases:
```
{}
```

Now, please determine its correctness. You can think step by step, but **REMEMBER** to use **### The program is {}.** to repeat your judgement, where {} should only be **correct** or **incorrect**.
"""


def capture_print_output(code):
    # Create a StringIO stream to capture output
    output = io.StringIO()

    # Save the current stdout (standard output)
    current_stdout = sys.stdout

    # Set the stdout to the StringIO stream
    sys.stdout = output

    try:
        exec_globals = {}
        # Run the function
        exec(code, exec_globals)
    except Exception as e:
        print("Error:", e)
    finally:
        # Restore stdout to its original setting
        sys.stdout = current_stdout

    # Get the content of the StringIO stream
    return output.getvalue()


def _process_item(item):
    completion = item["completion"]

    program_s = completion.find("<NEW PROGRAM>")
    if program_s == -1:
        return False

    program_s += len("<NEW PROGRAM>")
    program_e = completion.find("</NEW PROGRAM>")
    program = completion[program_s:program_e].strip()

    test_cases = []
    case_s = completion.find("<TEST CASE>")
    while case_s != -1:
        case_s += len("<TEST CASE>")
        case_e = completion.find("</TEST CASE>", case_s)
        if case_e == -1:
            break
        case = completion[case_s:case_e].strip()
        test_cases.append(case)
        case_s = completion.find("<TEST CASE>", case_e)

    if len(test_cases) == 0:
        return False

    composed_program = "\n".join([program] + [f"print(\"Execution Results:\", {case})" for case in test_cases])

    outputs = capture_print_output(composed_program)

    return composed_program, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str)
    parser.add_argument("--print_prediction_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--prompt_version", type=str, default="0")
    args = parser.parse_args()

    _running_code_judge_template = {
        "0": _running_code_judge_template_0,
        "1": _running_code_judge_template_1,
    }[args.prompt_version]

    predictions = json.load(open(args.prediction_file))
    task_id2pred = {str(item["task_id"]): item for item in predictions}

    print_predictions = [json.loads(line) for line in open(args.print_prediction_file).readlines()]

    for item in tqdm(print_predictions, total=len(print_predictions)):
        res = _process_item(item)

        task_id = str(item["task_id"])
        orig_program = task_id2pred[task_id]["completion"]
        item["orig_program"] = orig_program
        item["print_completion"] = item.pop("completion")

        if res:
            composed_program, outputs = res
            item["exec_outputs"] = outputs
            new_prompt = _running_code_judge_template.format("{}", "{}", item["orig_prompt"], orig_program, composed_program, outputs, "{}", "{}")
            item["prompt"] = new_prompt
        else:
            new_prompt = _vanilla_code_judge_template.format("{}", "{}", item["orig_prompt"], orig_program)
            item["prompt"] = new_prompt

    with open(args.output_file, "w", encoding="utf8") as f:
        for item in print_predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    # outputs = [json.loads(line) for line in open("outputs/oss_magicoder_mbpp257_pred_print_inter_outputs.gpt4.outputs.jsonl").readlines()]
    #
    # for item in outputs[:10]:
    #     res = _process_item(item)
    #     if res:
    #         print()
    #         print("+++++++++++++++++++++++++++++++++++++++++++++")
    #         print(res[0])
    #         print("-----------------------------------------------")
    #         print(res[1])
    #         print("++++++++++++++++++++++++++++++++++++++++++++")
    #         print()
    #     else:
    #         print("==========================================")
    #         print(item["completion"])
    #         print("=========================================")

    main()
