import argparse
import collections
import copy
import json
import re
import sys
from argparse import ArgumentParser
from datasets import load_dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from tqdm import tqdm
import os

from tqdm import tqdm

sys.set_int_max_str_digits(0)

"""
Copied from `scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py`.

In the processing of oss combine data, the pseudo test cases are directly saved in `input_output` fields so I re-write this script.
"""


def worker(item, min_success_test_num: int = 1):
    inputs = item["input_output"]["inputs"]
    outputs_counter = [
        collections.Counter() for _ in inputs
    ]
    output_str2orig_pred = [
        {} for _ in inputs
    ]
    resp2outputs = [
        {} for _ in range(len(item["outputs"]))
    ]

    assert len(item["full_res"]) == len(item["outputs"]) == len(item["pred"]), (len(item["full_res"]), len(item["outputs"]), len(item["pred"]))
    # if not len(item["full_res"]) == len(item["outputs"]) == len(item["pred"]):  # TODO: Figure it out why this happens
    #     print(len(item["full_res"]), len(item["outputs"]), len(item["pred"]))
    #     return {
    #         "inputs": [],
    #         "outputs": [],
    #         "output_meta": [],
    #         "sc_match_res": [],
    #     }

    for resp_id, (full_res, pg_outputs) in enumerate(zip(item["full_res"], item["outputs"])):
        for case_j, (case_r, case_o) in enumerate(zip(full_res, pg_outputs)):
            if case_j >= len(inputs):
                break
            if case_r != 0:
                continue
            # assert case_o  # sometimes is could be `int` or `True` or `False`. We believe the `case_r` here.

            if not str(case_o):
                continue  # some outputs could be empty string

            if str(case_o) not in output_str2orig_pred[case_j]:
                output_str2orig_pred[case_j][str(case_o)] = case_o
            outputs_counter[case_j][str(case_o)] += 1
            resp2outputs[resp_id][case_j] = str(case_o)

    new_inputs = []
    new_outputs = []
    new_output_meta = []
    sc_match_res = [[] for _ in range(len(item["pred"]))]
    for case_j, output_cnt in enumerate(outputs_counter):
        if not output_cnt:
            continue

        if sum(output_cnt.values()) < min_success_test_num:
            continue

        new_inputs.append(inputs[case_j])

        sc_o = output_cnt.most_common(1)[0][0]
        sc_o_real = output_str2orig_pred[case_j][sc_o]

        new_outputs.append(sc_o_real)

        new_output_meta.append({
            "output_freq": output_cnt,
            "output_str2orig_pred": output_str2orig_pred[case_j],
        })

        for pg_i in range(len(item["pred"])):
            if case_j not in resp2outputs[pg_i]:
                sc_match_res[pg_i].append(-2)  # compilation error
                continue
            if resp2outputs[pg_i][case_j] == sc_o:
                sc_match_res[pg_i].append(1)
            else:
                sc_match_res[pg_i].append(0)

    return {
        "inputs": new_inputs,
        "outputs": new_outputs,
        "output_meta": new_output_meta,
        "sc_match_res": sc_match_res,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--pseudo_test_case_file", type=str, default=True)
    parser.add_argument("--construct_prefer_pair", default=False, action="store_true")
    parser.add_argument("--min_success_test_num", type=int, default=2)
    parser.add_argument("--pass_case_margin", type=float, default=1)
    parser.add_argument("--pass_case_lower_bound", type=float, default=0.5)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.pseudo_test_case_file):
        print(f"Loading pseudo test cases from {args.pseudo_test_case_file}")
        data = json.load(open(args.pseudo_test_case_file))
    else:
        data = []
        for file in glob(args.pseudo_test_case_file):
            print(file)
            if file.endswith(".json"):
                tmp = json.load(open(file))
            else:
                tmp = [json.loads(line) for line in open(file).readlines()]
            data.extend(tmp)

    outputs = []
    before_test_num = 0
    cnt = 0
    missing_predictions = 0
    avg_test_case_num = 0
    pass_cnt = collections.Counter()
    for item in tqdm(data):
        if "outputs" not in item:
            print(item["pred"])
            missing_predictions += 1
            continue

        before_test_num += len(item["input_output"]["inputs"])

        result = worker(item, min_success_test_num=args.min_success_test_num)

        if not result["inputs"]:
            continue

        item["input_output"]["inputs"] = result["inputs"]
        item["input_output"]["outputs"] = result["outputs"]
        item["input_output"]["output_meta"] = result["output_meta"]
        item["sc_full_res"] = result["sc_match_res"]
        avg_test_case_num += len(result["inputs"])

        if args.construct_prefer_pair:
            pred_pass_cnt = []
            for pg_i, pg_res in enumerate(item["sc_full_res"]):
                pred_pass_cnt.append(sum([1 for r in pg_res if r == 1]))
                pass_cnt[pred_pass_cnt[-1]] += 1

            pos = []
            neg = []
            pos_code = []
            neg_code = []
            num_test_cases = len(item["input_output"]["inputs"])
            assert num_test_cases == len(item["sc_full_res"][0])
            assert len(pred_pass_cnt) == len(item["response"]) == len(item["pred"])
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
            item["pos_code"] = pos_code
            item["neg"] = neg
            item["neg_code"] = neg_code
            cnt += len(pos)

        outputs.append(item)

    print(len(outputs))
    print(cnt)
    print(missing_predictions)
    print(before_test_num / len(data) if data else 0)
    print(avg_test_case_num / len(outputs) if outputs else 0)
    print(pass_cnt)

    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()

"""
>>> python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs.py \
    --pseudo_test_case_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.*-of-32.run_outputs.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 5

13033
111756
2
10.003563932998059
9.762295710887747
Counter({10: 81281, 0: 22865, 9: 4529, 1: 4140, 8: 3010, 5: 2760, 2: 2636, 7: 2300, 4: 2256, 3: 2250, 6: 2158, 11: 120, 20: 24, 16: 1})


python scripts/apps/pseudo_test_cases/oss_combine_collect_pseudo_outputs.py \
    --pseudo_test_case_file "../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.*-of-32.run_outputs.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/split-32/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.dpo_m6_low0.5_min1.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 --min_success_test_num 1
    
14899
136123
2
10.003563932998059
9.835290959124773
Counter({10: 84309, 0: 38044, 9: 4612, 1: 4199, 8: 3019, 5: 2823, 2: 2705, 7: 2356, 4: 2282, 3: 2275, 6: 2211, 11: 128, 20: 26, 16: 1})

"""
