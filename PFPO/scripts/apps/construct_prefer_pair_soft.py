import argparse
import json
import os.path
import sys
from glob import glob
from tqdm import tqdm
import collections

sys.set_int_max_str_digits(0)

"""
Soft version of constructing preference pair. This would be useful for teacher-generated pseudo test cases.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--response_field", type=str, default="response")
    parser.add_argument("--test_case_field", type=str, default="input_output")
    parser.add_argument("--pass_case_margin", type=float, default=1)
    parser.add_argument("--pass_case_lower_bound", type=float, default=0.5)
    args = parser.parse_args()

    cnt = 0

    if os.path.exists(args.input_file):
        if args.input_file.endswith(".json"):
            data = json.load(open(args.input_file))
        else:
            data = [json.loads(line) for line in open(args.input_file).readlines()]
    else:
        data = []
        for file in glob(args.input_file):
            print(file)
            if file.endswith(".json"):
                data += json.load(open(file))
            else:
                data += [json.loads(line) for line in open(file).readlines()]

    if len(data) == 0:
        raise ValueError(f"No data found in {args.input_file}")

    print(len(data))
    pass_cnt = collections.Counter()
    for item in tqdm(data):
        pos = []
        neg = []
        pos_code = []
        neg_code = []
        if isinstance(item[args.response_field], str):  # We cannot make pairs if there is only one response.
            item["pos"] = []
            item["pos_code"] = []
            item["neg"] = []
            item["neg_code"] = []
            continue

        if len(item[args.test_case_field]["inputs"]) == 0:
            item["pos"] = []
            item["pos_code"] = []
            item["neg"] = []
            item["neg_code"] = []
            continue

        if "res" in item and "full_res" in item and item[args.test_case_field]:  # If there is no test-cases, we cannot determine the correctness
            assert len(item["res"]) == len(item["full_res"]) == len(item[args.response_field]), (len(item["res"]),
                                                                                                 len(item["full_res"]),
                                                                                                 len(item[args.response_field]),
                                                                                                 item[args.response_field])

            pred_pass_cnt = []
            for pg_i, pg_res in enumerate(item["full_res"]):
                pred_pass_cnt.append(sum([1 for r in pg_res if r == 1]))
                pass_cnt[pred_pass_cnt[-1]] += 1
            num_test_cases = len(item[args.test_case_field]["inputs"])

            for i in range(len(pred_pass_cnt)):
                resp_i = item["response"][i]
                prog_i = item["pred"][i]
                pass_cnt_i = pred_pass_cnt[i]
                if pass_cnt_i / num_test_cases < args.pass_case_lower_bound:
                    continue
                for j in range(len(pred_pass_cnt)):
                    if i == j:
                        continue
                    resp_j = item["response"][j]
                    prog_j = item["pred"][j]
                    pass_cnt_j = pred_pass_cnt[j]
                    if pass_cnt_i - pass_cnt_j >= args.pass_case_margin:
                        pos.append(resp_i)
                        pos_code.append(prog_i)
                        neg.append(resp_j)
                        neg_code.append(prog_j)

        item["pos"] = pos
        item["neg"] = neg
        item["pos_code"] = pos_code
        item["neg_code"] = neg_code
        cnt += len(pos)

        if args.response_field != "response":
            item["response"] = item.pop(args.response_field)
        if args.test_case_field != "input_output":
            item["input_output"] = item.pop(args.test_case_field)

    json.dump(data, open(args.output_file, "w"), ensure_ascii=False)
    print(len(data), cnt, cnt / len(data))


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/construct_prefer_pair.py --input_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v1.0.pseudo_test_case.exec.sc.json \
    --output_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v1.0.pseudo_test_case.exec.sc.dpo_v1.0.json --test_case_field test_cases

4223
100%|████████████████████████████████████████████████████████████████████████████████████| 4223/4223 [00:00<00:00, 148189.90it/s]
4223 18383 4.353066540374142

>>> python scripts/apps/construct_prefer_pair.py --input_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.self_s43_pseudo_cases.exec.json \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.self_s43_pseudo_cases.exec_dpo.json --test_case_field test_cases

4715
100%|██████████████████████████████████████████████████████████| 4715/4715 [00:00<00:00, 107036.93it/s]
4715 15921 3.3766702014846235

>>> python ~/gpt-chat-examples/scripts/apps/construct_prefer_pair.py --input_file "train.0shot.tem1.0.n10.?-of-8.v2.0.json" \
    --output_file "train.0shot.tem1.0.n10.v2.0.dpo_v1.0.json" --test_case_field test_cases

4500
100%|██████████████████████| 4500/4500 [00:00<00:00, 142921.59it/s]
4500 42550 9.455555555555556

>>> python scripts/apps/construct_prefer_pair.py \
    --input_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.?-of-4.v2.0.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.dpo_v1.0.json \
    --test_case_field test_cases

"""
