import argparse
import io
import json
import multiprocessing
import os.path
import sys
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool

import resource
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.mbpp_eval.execute import time_limit, TimeoutException


# Function to set memory limits (e.g., 100 MB)
def set_memory_limit(memory_limit_mb):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (memory_limit_mb * 1024 * 1024, hard))


def unsafe_execute(check_program, result, timeout):
    set_memory_limit(256)

    # Create a StringIO stream to capture output
    output = io.StringIO()

    # Save the current stdout (standard output)
    current_stdout = sys.stdout

    # Set the stdout to the StringIO stream
    sys.stdout = output

    # Run program.
    try:
        exec_globals = {}
        # with swallow_io():
        with time_limit(timeout):
            exec(check_program, exec_globals)
        result.append("passed")
    except TimeoutException:
        result.append("timed out")
    except BaseException as e:
        result.append(f"failed: {e}")
    finally:
        # Restore stdout to its original setting
        sys.stdout = current_stdout

    result.append(output.getvalue())


# def capture_print_output(check_program, timeout=1.0):
#     """
#     Evaluates the functional correctness of a completion by running the test
#     suite provided in the problem.
#
#     :param completion_id: an optional completion ID so we can match
#         the results later even if execution finishes asynchronously.
#     """
#     # manager = multiprocessing.Manager()
#     # result = manager.list()
#     # p = multiprocessing.Process(target=unsafe_execute, args=(check_program, result, timeout))
#     # p.start()
#     # p.join(timeout=timeout + 1)
#     # if p.is_alive():
#     #     p.kill()
#
#     result = []
#     unsafe_execute(check_program, result, timeout)
#
#     if not result:
#         result.append("timed out")
#         result.append("")
#
#     passed = result[0] == "passed"
#     output = result[1]
#
#     return output, passed


def capture_print_output(code, timeout=2):
    # output_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    output_queue = manager.list()
    process = multiprocessing.Process(target=unsafe_execute, args=(code, output_queue, timeout))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return "Error: Timeout", False

    if not output_queue or len(output_queue) < 2:  # TODO: I don't know the case when the length of `output_queue` is 1.
        return "Error: Unknown", False
    else:
        passed, output = output_queue
        return output, passed


def test_single_case(program, test_case):
    if "assert" in test_case:
        test_case = test_case.replace("assert ", "").strip()

    program = program + "\n\n" + f"print({test_case})"
    output, passed = capture_print_output(program, timeout=3)
    return output, passed


def _worker(_input):
    task_id, program_id, case_id, program, test_case = _input
    output, passed = test_single_case(program, test_case)
    return task_id, program_id, case_id, output, passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_prediction_file", type=str)
    parser.add_argument("--test_cases", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--top_k_prog", type=int, default=100)
    parser.add_argument("--top_k_case", type=int, default=100)
    args = parser.parse_args()

    program_predictions = json.load(open(args.program_prediction_file))
    programs = {}
    for item in program_predictions:
        if item["task_id"] not in programs:
            programs[item["task_id"]] = {
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "programs": set(),
            }
        programs[item["task_id"]]["programs"].add(item["completion"])

    test_cases = json.load(open(args.test_cases))
    test_cases = {int(task_id): item["aug_cases"][:args.top_k_case] for task_id, item in test_cases.items()}

    for task, item in programs.items():
        programs[task] = {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "programs": list(item["programs"])[:args.top_k_prog],
        }

    mp_inputs = []
    for task in programs:
        for program_id, program in enumerate(programs[task]["programs"]):
            for case_id, test_case in enumerate(test_cases[task]):
                mp_inputs.append((task, program_id, case_id, program, test_case))

    pbar = tqdm(mp_inputs, total=len(mp_inputs))
    task_results = defaultdict(dict)
    # with Pool(args.num_workers) as p:
    #     # for result in p.imap(_worker, mp_inputs):
    #     for result in p.imap_unordered(_worker, mp_inputs):
    #         task_id, program_id, case_id, output, passed = result
    #         if passed is False:
    #             continue
    #         if case_id not in task_results[task_id]:
    #             task_results[task_id][case_id] = {}
    #
    #         if not output:
    #             print(f"Warning: empty output with task id {task_id} and case id {case_id}\n\nProgram:\n{programs[task_id]['programs'][program_id]}")
    #
    #         if output not in task_results[task_id][case_id]:
    #             task_results[task_id][case_id][output] = []
    #         task_results[task_id][case_id][output].append(program_id)
    #
    #         pbar.update()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for _input in pbar:
            future = executor.submit(_worker, _input)
            futures.append(future)
            pbar.update()

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            result = future.result()
            task_id, program_id, case_id, output, passed = result
            if passed is False:
                continue
            if case_id not in task_results[task_id]:
                task_results[task_id][case_id] = {}

            if not output:
                print(f"Warning: empty output with task id {task_id} and case {test_cases[task_id][case_id]}\n\n"
                      f"Program:\n{programs[task_id]['programs'][program_id]}\n\nOutput: {output}")
                continue

            # print(f"Task: {task_id}, Program: {program_id}, Case: {test_cases[task_id][case_id]}, Passed: {passed}")
            # print(f"Program: {programs[task_id]['programs'][program_id]}")
            # print(f"Output: {output}")
            # print("========================================================================")

            if output not in task_results[task_id][case_id]:
                task_results[task_id][case_id][output] = []
            task_results[task_id][case_id][output].append(program_id)

    # for _input in pbar:
    #     result = _worker(_input)
    #     task_id, program_id, case_id, output, passed = result
    #     print(f"Task: {task_id}, Program: {program_id}, Case: {case_id}, Passed: {passed}")
    #     print(f"Program: {programs[task_id]['programs'][program_id]}")
    #     print(f"Output: {output}")
    #     if passed is False:
    #         continue
    #     if case_id not in task_results[task_id]:
    #         task_results[task_id][case_id] = {}
    #
    #     if not output:
    #         print(f"Warning: empty output with task id {task_id} and case id {case_id}\n\nProgram:\n{programs[task_id]['programs'][program_id]}")
    #
    #     if output not in task_results[task_id][case_id]:
    #         task_results[task_id][case_id][output] = []
    #     task_results[task_id][case_id][output].append(program_id)

    sc_results = []
    visited = {}
    for task_id, cases in tqdm(task_results.items(), total=len(task_results)):
        program_pass_cnt = Counter()
        for case_id, outputs in cases.items():
            if not outputs:
                continue
            tmp = sorted([(len(v), k) for k, v in outputs.items()], reverse=True)
            maj_program_ids = outputs[tmp[0][1]]
            program_pass_cnt.update(maj_program_ids)

        if not program_pass_cnt:
            continue
        best_program_id = program_pass_cnt.most_common(1)[0][0]
        sc_results.append({
            "task_id": task_id,
            "completion": f"[BEGIN]\n{programs[task_id]['programs'][best_program_id]}\n[END]",
        })
        visited[task_id] = True

    cnt = 0
    for task, programs in programs.items():
        cnt += 1
        if task not in visited:
            sc_results.append({
                "task_id": task,
                "completion": f"[BEGIN]\n{programs['programs'][0]}\n[END]",
            })

    print(f"Missing {cnt} programs.")
    print(len(sc_results))
    with open(args.program_prediction_file.replace(".json", f"_sc_{args.top_k_prog}_{args.top_k_case}.jsonl"), "w") as f:
        for item in sc_results:
            f.write(json.dumps(item) + "\n")

    json.dump(task_results, open(args.program_prediction_file.replace(".json", f"_sc_{args.top_k_prog}_{args.top_k_case}_outputs.json"), "w"))


if __name__ == "__main__":
    main()

"""
python scripts/mbpp/run_test_case_v1.0.py --program_prediction_file ../msranlpintern/instruction_tuning/experiments/h100/oos_sc2_magicdoer_mix/model_lr3e-6_batch512_epochs3_gpus8_linearSchedule/evaluation/open_instruct_results_local/mbpp_257_n100/mbpp_eval_predictions.json --test_cases outputs/mbpp/mbpp_test_case_inputs.w_test.v1.0.compl.gpt-4-32k.tem1.0.combine.json 
"""
