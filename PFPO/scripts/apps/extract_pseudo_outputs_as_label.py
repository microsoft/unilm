import argparse
import collections
import json
import sys

sys.set_int_max_str_digits(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--use_sc", default=False, action="store_true")
    parser.add_argument("--problem_id_field", type=str, default="problem_id")
    parser.add_argument("--test_case_field", type=str, default="input_output")
    parser.add_argument("--cover", default=False, action="store_true")
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    pseudo_test_cases = []
    unaligned = 0
    total = 0
    for item in data:
        problem_id = item[args.problem_id_field]

        if not item[args.test_case_field]:
            continue

        if "res" not in item or (not item["res"]):
            continue

        if len(item["res"]) != len(item["outputs"]):
            unaligned += 1
            continue

        input_outputs = collections.defaultdict(list)
        for i, (r, outputs) in enumerate(zip(item["res"], item["outputs"])):
            if r in [-1, -2]:
                continue

            for j, o in enumerate(outputs):
                if o:
                    # _input = item["input_output"]["inputs"][j]
                    input_outputs[j].append(o)

        if args.use_sc:
            test_cases = {
                "inputs": [],
                "outputs": []
            }
            for _input_index, outputs in input_outputs.items():
                pairs = {str(o): o for o in outputs}
                cnt = collections.Counter(list(pairs.keys()))
                cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
                _output = cnt[0][0]
                _output = pairs[_output]
                test_cases["inputs"].append(item[args.test_case_field]["inputs"][_input_index])
                test_cases["outputs"].append(_output)
        else:
            test_cases = {
                "inputs": [],
                "outputs": [],
            }
            for _input, outputs in input_outputs.items():
                test_cases["inputs"].append(_input)
                test_cases["outputs"].append(outputs[0])

        total += len(test_cases["inputs"])
        if len(test_cases["inputs"]) == 0:
            continue
        if args.cover:
            item.pop("res")
            item.pop("outputs")
            item.pop("full_res")
            item.pop("errors")
            if "fn_name" in item[args.test_case_field]:
                test_cases["fn_name"] = item[args.test_case_field]["fn_name"]
            item["input_output"] = test_cases
            pseudo_test_cases.append(item)
        else:
            pseudo_test_cases.append({
                "problem_id": problem_id,
                "input_output": test_cases
            })

    print(f"Unaligned: {unaligned}")
    print(f"Total number of pseudo test cases: {len(pseudo_test_cases)}")
    print(f"Average number of test cases: {total} / {len(pseudo_test_cases)} = {total / len(pseudo_test_cases)}")
    json.dump(pseudo_test_cases, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()

"""
>>> python scripts/apps/extract_pseudo_outputs_as_label.py --input_file outputs/apps/apps.train.r2c.vanilla.gpt-4o.tem1.0.n11.pure_outputs.json --output_file outputs/apps/apps.train.r2c.vanilla.gpt-4o.tem1.0.n11.pseudo_test_cases.json --use_sc

Unaligned: 582
Total number of pseudo test cases: 4223
Average number of test cases: 13674 / 4223 = 3.2379824769121477


>>> python scripts/apps/extract_pseudo_outputs_as_label.py --input_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.s43.run_outputs.json --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.s43.run_outputs.pseudo_cases.sc.json --use_sc --problem_id_field id --test_case_field test_cases

Unaligned: 87
Total number of pseudo test cases: 4715
Average number of test cases: 17551 / 4715 = 3.72237539766702

"""
