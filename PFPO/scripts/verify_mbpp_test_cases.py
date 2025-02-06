import json
import argparse
import io
import sys
import re
from tqdm import tqdm


def capture_print_output(code):
    # Create a StringIO stream to capture output
    output = io.StringIO()

    # Save the current stdout (standard output)
    current_stdout = sys.stdout

    # Set the stdout to the StringIO stream
    sys.stdout = output

    passed = True
    try:
        exec_globals = {}
        # Run the function
        exec(code, exec_globals)
    except Exception as e:
        print("Error:", e)
        passed = False
    finally:
        # Restore stdout to its original setting
        sys.stdout = current_stdout

    # Get the content of the StringIO stream
    return output.getvalue(), passed


def _extract_test_case(item):
    completion = item["completion"]

    s = completion.find("[BEGIN]")
    if s == -1:
        return False

    s += len("[BEGIN]")
    e = completion.find("[END]")

    if e == -1:
        e = len(completion)

    cases = completion[s:e].strip()
    cases = cases.replace("```", "")
    cases = cases.split("\n")
    cases = [case for case in cases if case.startswith("assert")]
    cases = [case for case in cases if "<function>" in case]

    return cases


def extract_test_cases(file_path):
    if file_path.endswith(".json"):
        predictions = json.load(open(file_path))
        assert isinstance(predictions, dict)
        task_id2cases = {int(k): v for k, v in predictions.items()}
        return task_id2cases

    predictions = [json.loads(line) for line in open(file_path).readlines()]

    task_id2cases = {}
    for item in predictions:
        cases = _extract_test_case(item)
        task_id2cases[item["task_id"]] = cases

    return task_id2cases


def _extract_function_names(code):
    # Regular expression to match function definitions
    function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('

    # Find all matches in the code
    function_names = re.findall(function_pattern, code)

    return function_names


def clean_orig_test_cases_and_extract_func_name(program):
    func_name = _extract_function_names(program)
    if len(func_name) < 1:
        print(f"Warning: Function name case")
        print(program)
        print("==========================================")
        return None

    func_name = func_name[0]

    lines = program.split("\n")
    orig_cases = []

    new_lines = []
    for line in lines:
        if line.startswith("assert"):
            orig_cases.append(line)
        else:
            new_lines.append(line)

    new_program = "\n".join(new_lines)

    return new_program, orig_cases, func_name


def filter_passed_programs(prediction_file):
    solutions = json.load(open(prediction_file))

    tmp = len(solutions)
    print(f"Loaded {tmp} programs from {prediction_file}")

    correct_solutions = []
    incorrect_solutions = []
    for item in solutions:
        if item["passed"]:
            correct_solutions.append(item)
        else:
            incorrect_solutions.append(item)

    print(f"Filtered out {len(correct_solutions)} correct solutions and {len(incorrect_solutions)} incorrect solutions.")

    return correct_solutions, incorrect_solutions


def verify_program(program, test_cases):
    res = clean_orig_test_cases_and_extract_func_name(program)
    if res is None:
        return None

    pure_program, _, func_name = res

    cases = [case.replace("<function>", func_name) for case in test_cases]

    new_program = pure_program + "\n" + "\n".join(cases)

    output, passed = capture_print_output(new_program)

    return output, passed, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_prediction_file", type=str)
    parser.add_argument("--case_prediction_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    task_id2cases = extract_test_cases(args.case_prediction_file)

    correct_programs, incorrect_programs = filter_passed_programs(args.program_prediction_file)

    corr_pass = 0
    corr_fail = 0
    in_corr_pass = 0
    in_corr_fail = 0
    for program in tqdm(correct_programs):
        task_id = program["task_id"]
        test_cases = task_id2cases[task_id]

        if test_cases is False:
            print(f"Warning: No test cases for task {task_id}")
            program["results"] = []
            continue

        res = []
        for case in test_cases:
            tmp = verify_program(program["completion"], [case])
            if tmp is None:
                output = "Bad Program"
                passed = False
                cases = [case]
            else:
                output, passed, cases = tmp
                if passed:
                    corr_pass += 1
                else:
                    corr_fail += 1

            res.append({
                "case": cases[0],
                "output": output,
                "passed": passed
            })

        program["results"] = res

    for program in tqdm(incorrect_programs):
        task_id = program["task_id"]
        test_cases = task_id2cases[task_id]

        if test_cases is False:
            print(f"Warning: No test cases for task {task_id}")
            program["results"] = []
            continue

        res = []
        for case in test_cases:
            tmp = verify_program(program["completion"], [case])
            if tmp is None:
                output = "Bad Program"
                passed = False
                cases = [case]
            else:
                output, passed, cases = tmp
                if passed:
                    in_corr_pass += 1
                else:
                    in_corr_fail += 1

            res.append({
                "case": cases[0],
                "output": output,
                "passed": passed
            })

        program["results"] = res

    print(f"Correct programs: {corr_pass} passed, {corr_fail} failed.")
    print(f"Incorrect programs: {in_corr_pass} passed, {in_corr_fail} failed.")

    outputs = correct_programs + incorrect_programs
    with open(args.output_file, "w") as fout:
        json.dump(outputs, fout)


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

"""
python scripts/verify_mbpp_test_cases.py --program_prediction_file outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json --case_prediction_file outputs/mbpp_257_test_case_gen.outputs.gpt-4.v1.0.1.jsonl --output_file outputs/mbpp_257_test_case_gen.gpt-4.v1.0.1.verify.w_gpt-4-programs-v1.1.results.json
Loaded 257 programs from outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json
Filtered out 217 correct solutions and 40 incorrect solutions.
  0%|                                                                                                                                                                                                                                                                                                                                                                                                 | 0/217 [00:00<?, ?it/s]Warning: No test cases for task 67
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 217/217 [00:00<00:00, 2031.82it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:00<00:00, 198.23it/s]
Correct programs: 1262 passed, 542 failed.
Incorrect programs: 172 passed, 196 failed.
"""

"""
python scripts/verify_mbpp_test_cases.py --program_prediction_file outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json --case_prediction_file outputs/mbpp_257_test_case_gen.outputs.gpt-4.v1.1.jsonl --output_file outputs/mbpp_257_test_case_gen.gpt-4.v1.1.verify.w_gpt-4-programs-v1.1.results.json
Loaded 257 programs from outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json
Filtered out 217 correct solutions and 40 incorrect solutions.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 217/217 [00:00<00:00, 1794.48it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:00<00:00, 208.21it/s]
Correct programs: 1562 passed, 600 failed.
Incorrect programs: 212 passed, 188 failed.


python scripts/verify_mbpp_test_cases.py --program_prediction_file outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json --case_prediction_file outputs/mbpp_257_test_case_gen.gpt-4.v1.1.case_inputs.v1.1.outputs.sc.0611.json --output_file outputs/mbpp_257_test_case_gen.gpt-4.v1.1.sc0611.verify.w_gpt-4-programs-v1.1.results.json
Loaded 257 programs from outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json
Filtered out 217 correct solutions and 40 incorrect solutions.
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 217/217 [00:00<00:00, 2950.43it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:00<00:00, 4273.14it/s]
Correct programs: 1057 passed, 220 failed.
Incorrect programs: 96 passed, 47 failed.

python scripts/verify_mbpp_test_cases.py --program_prediction_file outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json --case_prediction_file outputs/mbpp_257_test_case_gen.gpt-4.v1.1.case_inputs.v1.1.outputs.sc.0612.json --output_file outputs/mbpp_257_test_case_gen.gpt-4.v1.1.sc0612.verify.w_gpt-4-programs-v1.1.results.json
Loaded 257 programs from outputs/mbpp_sanitized_v1.1.outputs.gpt-4.eval/mbpp_eval_predictions.json
Filtered out 217 correct solutions and 40 incorrect solutions.
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 217/217 [00:00<00:00, 3030.46it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:00<00:00, 4740.00it/s]
Correct programs: 1116 passed, 234 failed.
Incorrect programs: 108 passed, 44 failed.
"""
