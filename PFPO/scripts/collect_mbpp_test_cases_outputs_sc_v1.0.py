import collections

from datasets import load_dataset
import argparse
import json
from glob import glob
from collections import Counter


def process_completion(completion: str):
    if "[BEGIN]" in completion or "[ BEGIN ]" in completion:
        if "[BEGIN]" in completion:
            s = completion.index("[BEGIN]") + len("[BEGIN]")
            e = completion.index("[END]")
        else:
            s = completion.index("[ BEGIN ]") + len("[ BEGIN ]")
            e = completion.index("[ END ]")
        case = completion[s:e].strip()
        # outputs = case.split("==")[1].strip()
        outputs = case
    else:
        lines = completion.split()
        lines = [line for line in lines if line.startswith("assert")]
        if len(lines) == 0:
            outputs = None
        else:
            line = lines[-1]
            # outputs = line.split("==")[1].strip()
            outputs = line.strip()
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    # parser.add_argument("--sanitized", default=False, action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    # parser.add_argument("--prompt_type", type=str, default="vanilla")
    args = parser.parse_args()

    # print(glob(args.input_file))
    # return

    # if args.sanitized:
    #     dataset = load_dataset("mbpp", "sanitized", split="test").to_list()
    # else:
    #     dataset = load_dataset("mbpp", split="test").to_list()
    # prompt_key = "prompt" if args.sanitized else "text"

    results = dict()
    missing = 0
    for file in glob(args.input_file):
        predictions = [json.loads(line) for line in open(file).readlines()]

        for item in predictions:
            task_id = item["task_id"].split("_")[0]
            if task_id not in results:
                results[task_id] = {}

            case_input = item["case_input"].strip()
            outputs = process_completion(item["completion"])

            # ================ Original version =====================
            # if outputs is None or outputs.split("==")[0].strip() != case_input:
            #     print(f"Task ID: {task_id}")
            #     print(f"Case Input: {case_input}")
            #     print(f"Outputs: {outputs}")
            #     print("Skip due to mismatch prediction.")
            #     missing += 1

            if outputs is None:
                print(f"Task ID: {task_id}")
                print(f"Case Input: {case_input}")
                print(f"Outputs: {outputs}")
                print("Skip due to mismatch prediction.")
                missing += 1
                continue

            if case_input.lower() != outputs.split("==")[0].strip().lower():
                print(f"Task ID: {task_id}")
                print(f"Case Input: {case_input}")
                print(f"Outputs: {outputs}")
                print("Skip due to mismatch prediction.")
                missing += 1
                continue

            if case_input not in results[task_id]:
                results[task_id][case_input] = Counter()

            if outputs.split("==")[0].strip() != case_input:
                tmp = outputs.split("==")
                outputs = case_input + " == " + tmp[1].strip()

            results[task_id][case_input][outputs] += 1

    outputs = {}
    cnt = 0
    for task_id, cases in results.items():
        outputs[task_id] = []
        for case_input in cases:
            times = sorted(cases[case_input].items(), key=lambda x: x[1], reverse=False)
            if times[0][1] < 2:
                continue
            outputs[task_id].append(times[0][0])
            cnt += 1

    json.dump(outputs, open(args.output_file, "w"))
    print(f"Missing: {missing}")
    print(f"Total: {cnt}")


if __name__ == "__main__":
    main()


"""
python scripts/collect_mbpp_test_cases_outputs_sc_v1.0.py --input_file "outputs/mbpp_257_test_case_gen.gpt-4.v1.1.case_inputs.v1.1.outputs.gpt-*.tem1.0.s4[2456].jsonl" --output_file outputs/mbpp_257_test_case_gen.gpt-4.v1.1.case_inputs.v1.1.outputs.sc.0611.json
Missing: 1980
Total: 842

Missing: 1939
Total: 1420  # Changed version


python scripts/collect_mbpp_test_cases_outputs_sc_v1.0.py --input_file "outputs/mbpp_257_test_case_gen.gpt-4.v1.1.case_inputs.v1.1.outputs.gpt-*.tem1.0.s4[23456].jsonl" --output_file outputs/mbpp_257_test_case_gen.gpt-4.v1.1.case_inputs.v1.1.outputs.sc.0612.json
Missing: 2175
Total: 1502
"""

