import copy
import json
import re
import sys
from argparse import ArgumentParser
from datasets import load_dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
import os

from tqdm import tqdm

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.apps.utils_execute import run_inference_process


def extract_test_case_inputs(item):
    if item["input_output"]:
        item["input_output"] = json.loads(item["input_output"])

    if "fn_name" in item["input_output"]:
        function_call = True
    else:
        function_call = False

    full_inputs = []
    if function_call:
        try:
            response = json.loads(item["completion"])
        except Exception as e:
            # print(f"Cannot load response for {item['completion']}")
            print(e)
            return []

        for k, v in response.items():
            # if v.find(item["input_output"]["fn_name"]) != -1:
            #     inputs = v.replace(item["input_output"]["fn_name"], "").strip()
            #     inputs = inputs[1:-1]  # Remove `(` and `)`.
            #     try:
            #         inputs = eval(f"[{inputs}]")
            #     except Exception as e:
            #         print(f"Cannot eval inputs for {inputs}\t{item['input_output']['fn_name']}\t{v}\t{item['input_output']['inputs']}")
            #         print(e)
            #         continue
            #     full_inputs.append(inputs)
            if not isinstance(v, list):
                assert isinstance(v, str), v
                v = [v]
            full_inputs.append(v)
    else:
        try:
            response = json.loads(item["completion"])
        except Exception as e:
            # print(f"Cannot load response for {item['completion']}")
            print(e)
            return []
        for k, v in response.items():
            inputs = v
            full_inputs.append(inputs)

    return full_inputs


def _worker(item):
    results = []
    full_results = []
    all_outputs = []
    all_errors = []
    if not item["pred"]:
        if "res" in item:
            item.pop("res")
        if "full_res" in item:
            item.pop("full_res")
        if "outputs" in item:
            item.pop("outputs")
        if "errors" in item:
            item.pop("errors")
        return item

    for pred in item["pred"]:
        gen_solution = pred
        if gen_solution is None:
            results.append(False)
            full_results.append([-2] * 21)
            all_outputs.append([None] * 21)
            continue

        if not item["pseudo_test_cases"]:
            continue

        all_results = run_inference_process(item["pseudo_test_cases"], gen_solution, timeout=10, debug=False, return_output=True)
        res, outputs, errors = all_results
        for tmp in res:
            if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)):
                print(tmp, tmp.__class__.__name__)
        res = [bool(tmp) if (not isinstance(tmp, bool)) and (not isinstance(tmp, int)) else tmp for tmp in res]
        if all(item is True for item in res) is True:
            results.append(True)
        else:
            results.append(False)
        full_results.append(res)
        try:
            json.dumps(outputs)
            all_outputs.append(outputs)
            all_errors.append(errors)
        except:
            print(f"Cannot dump outputs for {outputs}")
            all_outputs.append([])
            all_errors.append([])

    if results:
        item["res"] = results
        item["full_res"] = full_results
        item["outputs"] = all_outputs
        item["errors"] = all_errors

    return item


def main():
    parser = ArgumentParser()
    parser.add_argument("--completion_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pseudo_test_case", type=str)
    parser.add_argument("--completion_test_field", type=str, default="test_cases")
    args = parser.parse_args()

    if os.path.exists(args.completion_file):
        if args.completion_file.endswith(".json"):
            data = json.load(open(args.completion_file))
        else:
            data = [json.loads(line) for line in open(args.completion_file).readlines()]
    else:
        data = []
        item_id2data_id = {}
        for file in glob(args.completion_file):
            print(file)
            if file.endswith(".json"):
                tmp = json.load(open(file))
            else:
                tmp = [json.loads(line) for line in open(file).readlines()]
            for item in tmp:
                item_id = item["id"]
                if item_id not in item_id2data_id:
                    item_id2data_id[item_id] = len(data)
                    data.append(item)
                else:
                    new_completions = item["response"]
                    if isinstance(new_completions, str):
                        new_completions = [new_completions]

                    if isinstance(data[item_id2data_id[item_id]]["response"], list):
                        data[item_id2data_id[item_id]]["response"].extend(new_completions)
                    else:
                        data[item_id2data_id[item_id]]["response"] = [data[item_id2data_id[item_id]]["response"]] + new_completions

    print(f"Total number of items: {len(data)}")

    ps_test_cases = []
    cnt = 0
    if args.pseudo_test_case.endswith(".json"):
        test_cases = json.load(open(args.pseudo_test_case))
        for item in test_cases:
            _input = extract_test_case_inputs(item)
            if not _input:
                continue
            item["pseudo_inputs"] = _input
            ps_test_cases.append(item)
    else:
        with open(args.pseudo_test_case) as f:
            lines = f.readlines()
            for line in lines:
                try:
                    item = json.loads(line)
                except:
                    print(f"Cannot load {line}")
                    cnt += 1
                    pass
                _input = extract_test_case_inputs(item)
                if not _input:
                    continue
                item["pseudo_inputs"] = _input
                ps_test_cases.append(item)
    print(cnt)
    print(f"Total number of pseudo test cases: {len(ps_test_cases)}")

    id2item = {item["problem_id"]: item for item in ps_test_cases}
    new_data = []
    for item in data:
        if not isinstance(item["response"], list):
            item["response"] = [item["response"]]
            item["pred"] = [item["pred"]]
        if item["id"] not in id2item:
            continue
        item["pseudo_test_cases"] = {
            "inputs": id2item[item["id"]]["pseudo_inputs"],
            "outputs": [],
        }
        if item[args.completion_test_field]:
            if not isinstance(item[args.completion_test_field], dict):
                item[args.completion_test_field] = json.loads(item[args.completion_test_field])
            if "fn_name" in item[args.completion_test_field]:
                item["pseudo_test_cases"]["fn_name"] = item[args.completion_test_field]["fn_name"]
        new_data.append(item)
    data = new_data
    print(f"Total number of items: {len(data)}")

    missing = 0
    corr = 0
    corr_at_k = 0
    pbar = tqdm(data)

    outputs = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for _input in pbar:
            future = executor.submit(_worker, _input)
            futures.append(future)
            pbar.update()

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            outputs.append(future.result())

    for item in outputs:
        if "res" in item and item["res"]:
            if item["res"][0] is True:
                corr += 1
            if any(item["res"]):
                corr_at_k += 1
        else:
            missing += 1

    print(f"Missing: {missing / len(outputs)}")
    print(f"Correct: {corr / len(outputs)}")
    print(f"Correct at k: {corr_at_k / len(outputs)}")
    json.dump(outputs, open(args.output_file, "w"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

"""
>>>  python scripts/apps/solution_run_pseudo_outputs_local.py \
    --completion_file "../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.?-of-8.v2.0.json" \
    --output_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json \
    --pseudo_test_case outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json 
    
    
>>> python scripts/apps/solution_run_pseudo_outputs_local.py \
    --completion_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.?-of-4.v2.0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_test_cases.v1.0.azure.json \
    --pseudo_test_case ${DATA_PREFIX_PATH}/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json --num_workers 128
"""
