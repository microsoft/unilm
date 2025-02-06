import json
import os
import sys
from argparse import ArgumentParser

from datasets import load_dataset

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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


def main():
    parser = ArgumentParser()
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pseudo_test_case", type=str)
    parser.add_argument("--completion_test_field", type=str, default="input_output")
    args = parser.parse_args()

    data = load_dataset("codeparrot/apps", split="train").to_list()

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
    outputs = []
    for item in data:
        if item["problem_id"] not in id2item:
            continue
        test_cases = {
            "inputs": id2item[item["problem_id"]]["pseudo_inputs"],
        }
        if item[args.completion_test_field]:
            if not isinstance(item[args.completion_test_field], dict):
                item[args.completion_test_field] = json.loads(item[args.completion_test_field])
            if "fn_name" in item[args.completion_test_field]:
                test_cases["fn_name"] = item[args.completion_test_field]["fn_name"]

        # This is to align with oss data format.
        new_item = {
            "input_output": test_cases,
            "question": item["question"],
            "problem_id": f"apps-train-{item['problem_id']}",
            "starter_code": item["starter_code"],
        }
        outputs.append(new_item)

    print(f"Total number of items: {len(outputs)}")
    json.dump(outputs, open(args.output_file, "w"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/pseudo_test_cases/combine_pseudo_test_inputs.py \
    --output_file ../msranlpintern/share/dataset/magicoder/apps-train.4o-test-func.json \
    --pseudo_test_case ../msranlpintern/share/gpt-chat-examples-outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only_combine.outputs.gpt4o.n1.tem0.0.json_obj.json
"""

