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


def load_test_cases(item):
    try:
        input_output = json.loads(item["completion"])
    except Exception as e:
        print(e)
        print(item["completion"])
        return []

    inputs = list(input_output.values())
    return inputs


x_code_eval_template_v1 = """{description}

Input: {input_from}
Output: {output_to}

Time limit: {time_limit}
Memory limit: {memory_limit}

#### Input Format

{input_spec}

#### Output Format

{output_spec}

#### Notes

{notes}

#### Example Input-Output

{sample_input_output}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--test_case_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--val_file", type=str)
    args = parser.parse_args()

    data = load_file(args.data_file)

    test_data = load_file(args.test_file)
    test_ids = set([item["src_uid"] for item in test_data])

    val_data = load_file(args.val_file)
    test_ids.update([item["src_uid"] for item in val_data])

    test_cases = load_file(args.test_case_file)
    id2test_cases = {}
    test_num = 0
    for item in test_cases:
        tmp = load_test_cases(item)
        if tmp:
            id2test_cases[item["src_uid"]] = tmp
            test_num += 1

    print(f"Test cases: {test_num} / {len(test_cases)}")

    outputs = []
    in_out_template = "Input\n```{}```\n\nOutput\n```{}```"
    test_held = 0
    for item in data:
        if item["input_from"] != 'standard input':
            continue

        if item["src_uid"] in test_ids:
            test_held += 1
            continue

        item_id = item["src_uid"]
        if item_id in id2test_cases:
            test_inputs = id2test_cases[item_id]
        else:
            continue

        sample_input_output = []
        for _in, _out in zip(item["sample_inputs"], item["sample_outputs"]):
            sample_input_output.append(in_out_template.format(_in, _out))

        item["sample_input_output"] = "\n\n".join(sample_input_output)

        question = x_code_eval_template_v1.format(**item)

        test_inputs = {
            "inputs": test_inputs,
        }
        outputs.append({
            "input_output": test_inputs,
            "question": question,
            "problem_id": f"xcode-eval-{item['src_uid']}",
        })

    print(len(outputs))
    print(f"No test: {test_held}")
    json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
