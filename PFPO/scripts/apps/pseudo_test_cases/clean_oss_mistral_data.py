import json
import argparse
import os.path
import re
from glob import glob


def load_file(file_path):
    if os.path.exists(file_path):
        if file_path.endswith(".json"):
            return json.load(open(file_path, encoding="utf-8"))
        else:
            return [json.loads(line) for line in open(file_path).readlines()]
    data = []
    for file in glob(file_path):
        print(file)
        if file.endswith(".json"):
            tmp = json.load(open(file, encoding="utf-8"))
        else:
            tmp = [json.loads(line) for line in open(file).readlines()]
        data.extend(tmp)
    return data


def extract_json(completion: str):
    # Regular expression to match content between [BEGIN] and [END]
    pattern = r'```(.*?)```'

    match = re.search(pattern, completion, re.DOTALL)
    if match:
        item = match.group(1).strip()
        if item.startswith("json"):
            item = item[5:]
        item = item.replace(",\n}", "\n}").replace(",}", "}").replace("None", "null")
        item = item.replace("(", "[").replace(")", "]").replace("\'", "\"")
        item = item.replace("True", "true").replace("False", "false")
        return json.loads(item)
    else:
        return None


def load_test_cases(item):
    try:
        response = extract_json(item["response"])
    except Exception as e:
        print(e)
        print(item["response"])
        return {}

    if not response:
        return {}

    if not isinstance(response, dict):
        return {}

    test_cases = {
        "inputs": [],
    }
    for k, v in response.items():
        if not k.startswith("test_case_"):
            continue
        test_cases["inputs"].append(v)

    return test_cases


def load_func_head(item) -> str:
    try:
        response = extract_json(item["response"])
    except Exception as e:
        print(e)
        print(item["response"])
        return ""

    if not response:
        return ""

    if "func_head" not in response:
        return ""

    return response["func_head"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--test_case_file", type=str)
    parser.add_argument("--func_head_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = load_file(args.data_file)

    test_cases = load_file(args.test_case_file)
    id2test_cases = {}
    test_num = 0
    for item in test_cases:
        tmp = load_test_cases(item)
        if tmp:
            id2test_cases[item["id"]] = tmp
            test_num += 1

    func_heads = load_file(args.func_head_file)
    id2func_head = {}
    func_num = 0
    for item in func_heads:
        tmp = load_func_head(item)
        if tmp:
            id2func_head[item["id"]] = tmp
            func_num += 1

    print(f"Test cases: {test_num} / {len(test_cases)}")
    print(f"Func heads: {func_num} / {len(func_heads)}")

    outputs = []
    for item in data:
        item_id = item["index"]
        if item_id in id2test_cases:
            test_inputs = id2test_cases[item_id]
        else:
            continue

        if item_id in id2func_head:
            func_head = id2func_head[item_id]
            if not func_head:
                continue
        else:
            continue

        # item["input_output"] = test_cases
        # item["input_output"]["fn_name"] = func_head
        # outputs.append(item)
        test_inputs["fn_name"] = func_head
        outputs.append({
            "input_output": test_inputs,
            "question": item["problem"],
            "problem_id": f"oss-instruct-{item_id}",
        })

    print(len(outputs))
    json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/pseudo_test_cases/clean_oss_mistral_data.py \
    --data_file ../msranlpintern/share/dataset/magicoder/data-oss_instruct-decontaminated-python.json \
    --test_case_file "../msranlpintern/share/models/Mistral-Large-Instruct-2407/apps-test-inputs-gen/sub_dev.0shot.tem0.0.n1.*-of-16.v1.0.json" \
    --func_head_file "../msranlpintern/share/models/Mistral-Large-Instruct-2407/apps-test-inputs-gen/oss_instruct_python.func_head_extract.tem0.0.n1.*-of-16.v1.0.json" \
    --output_file ../msranlpintern/share/dataset/magicoder/data-oss_instruct-decontaminated-python.mistral-large-test-func.json

Test cases: 31134
Func heads: 23807
18214

Test cases: 32066
Func heads: 23807
18969

Test cases: 32185
Func heads: 23807
19280


Test cases: 33760
Func heads: 23807
20342

Test cases: 33760 / 38284
Func heads: 23807 / 38284
20342

"""
