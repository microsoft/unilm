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


def worker(item, pseudo_test_case_field):
    inputs = item[pseudo_test_case_field]["inputs"]
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
    for resp_id, (full_res, pg_outputs) in enumerate(zip(item["full_res"], item["outputs"])):
        for case_j, (case_r, case_o) in enumerate(zip(full_res, pg_outputs)):
            if case_j >= len(inputs):
                break
            if case_r != 0:
                continue
            # assert case_o  # sometimes is could be `int` or `True` or `False`. We believe the `case_r` here.

            if str(case_o) not in output_str2orig_pred[case_j]:
                output_str2orig_pred[case_j][str(case_o)] = case_o
            outputs_counter[case_j][str(case_o)] += 1
            resp2outputs[resp_id][case_j] = str(case_o)

    new_inputs = []
    new_outputs = []
    new_inputs_non_sc = []
    new_outputs_non_sc = []
    new_output_meta = []
    sc_match_res = [[] for _ in range(len(item["pred"]))]
    for case_j, output_cnt in enumerate(outputs_counter):
        if not output_cnt:
            continue
        new_inputs.append(inputs[case_j])

        sc_o = output_cnt.most_common(1)[0][0]
        sc_o_real = output_str2orig_pred[case_j][sc_o]

        new_outputs.append(sc_o_real)

        # Non-sc output
        if case_j in resp2outputs[0]:
            new_inputs_non_sc.append(inputs[case_j])
            new_outputs_non_sc.append(output_str2orig_pred[case_j][resp2outputs[0][case_j]])

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
        "inputs_non_sc": new_inputs_non_sc,
        "outputs_non_sc": new_outputs_non_sc,
        "output_meta": new_output_meta,
        "sc_match_res": sc_match_res,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument("--pseudo_test_case_file", type=str, default=True)
    parser.add_argument("--test_case_field", type=str, default="pseudo_test_cases")
    parser.add_argument("--construct_prefer_pair", default=False, action="store_true")
    parser.add_argument("--pass_case_margin", type=float, default=1)
    parser.add_argument("--pass_case_lower_bound", type=float, default=0.5)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = json.load(open(args.pseudo_test_case_file))

    outputs = []
    cnt = 0
    missing_predictions = 0
    avg_test_case_num = 0
    avg_non_test_case_num = 0
    pass_cnt = collections.Counter()
    for item in tqdm(data):
        if "outputs" not in item:
            print(item["pred"])
            missing_predictions += 1
            continue

        result = worker(item, args.test_case_field)

        if not result["inputs"]:
            continue

        item[args.test_case_field]["inputs"] = result["inputs"]
        item[args.test_case_field]["outputs"] = result["outputs"]
        item[args.test_case_field]["output_meta"] = result["output_meta"]
        item[f"{args.test_case_field}_non_sc"] = copy.deepcopy(item[args.test_case_field])
        item[f"{args.test_case_field}_non_sc"]["inputs"] = result["inputs_non_sc"]
        item[f"{args.test_case_field}_non_sc"]["outputs"] = result["outputs_non_sc"]
        item["sc_full_res"] = result["sc_match_res"]
        avg_test_case_num += len(result["inputs"])
        avg_non_test_case_num += len(result["inputs_non_sc"])

        if args.construct_prefer_pair:
            pred_pass_cnt = []
            for pg_i, pg_res in enumerate(item["sc_full_res"]):
                pred_pass_cnt.append(sum([1 for r in pg_res if r == 1]))
                pass_cnt[pred_pass_cnt[-1]] += 1

            pos = []
            neg = []
            pos_code = []
            neg_code = []
            num_test_cases = len(item[args.test_case_field]["inputs"])
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
    print(avg_test_case_num / len(outputs) if outputs else 0)
    print(avg_non_test_case_num / len(outputs) if outputs else 0)
    print(pass_cnt)

    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()

"""
>>> python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
    --pseudo_test_case_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json \
    --output_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.clean.dpo_m2_low0.5.json \
    --construct_prefer_pair --pass_case_margin 3 --pass_case_lower_bound 0.5
    
 93%|██████████████████                      | 4095/4418 [00:03<00:00, 629.19it/s]None
100%|█████████████████████████████| 4418/4418 [00:03<00:00, 1323.18it/s]
~~4299
~~82365
~~1
~~9.953477552919283
~~Counter({10: 15551, 0: 15150, 1: 2567, 2: 1722, 9: 1585, 3: 1235, 8: 1220, 7: 1060, 4: 1016, 5: 951, 6: 933})
4299
72724
1
9.953477552919283
Counter({10: 15551, 0: 15150, 1: 2567, 2: 1722, 9: 1585, 3: 1235, 8: 1220, 7: 1060, 4: 1016, 5: 951, 6: 933})

>>> python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
    --pseudo_test_case_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json \
    --output_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.clean.dpo_m2_low0.5.json \
    --construct_prefer_pair --pass_case_margin 2 --pass_case_lower_bound 0.5 

~~4299
~~82365
~~1
~~9.953477552919283
~~Counter({10: 15551, 0: 15150, 1: 2567, 2: 1722, 9: 1585, 3: 1235, 8: 1220, 7: 1060, 4: 1016, 5: 951, 6: 933})
4299
77024
1
9.953477552919283
Counter({10: 15551, 0: 15150, 1: 2567, 2: 1722, 9: 1585, 3: 1235, 8: 1220, 7: 1060, 4: 1016, 5: 951, 6: 933})


>>> python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
    --pseudo_test_case_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json \
    --output_file ../msranlpintern/share/models/deepseek-coder-7b-instruct-v1.5/apps/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.clean.dpo_m6_low0.5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 
    
4299
59226
1
9.953477552919283
Counter({10: 15551, 0: 15150, 1: 2567, 2: 1722, 9: 1585, 3: 1235, 8: 1220, 7: 1060, 4: 1016, 5: 951, 6: 933})


>>> python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
    --pseudo_test_case_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_test_cases.v1.0.azure.json \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.clean.dpo_m6_low0.5.json \
    --construct_prefer_pair --pass_case_margin 6 --pass_case_lower_bound 0.5 

4712
59606
2
9.955220713073006
Counter({10: 17718, 0: 16091, 1: 2515, 2: 1841, 9: 1807, 3: 1364, 8: 1291, 6: 1172, 4: 1125, 5: 1115, 7: 1081})


>>> python scripts/apps/pseudo_test_cases/collect_pseudo_outputs.py \
    --pseudo_test_case_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_test_cases.v1.0.azure.json \
    --output_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.0shot.tem1.0.n10.v2.0.pseudo_input_output.v1.0.json

4712
0
2
9.955220713073006
Counter()

"""
