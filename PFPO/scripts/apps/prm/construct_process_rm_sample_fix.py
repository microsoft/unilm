import json
import argparse
from glob import glob
import os
import sys
import collections
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

sys.set_int_max_str_digits(0)

"""
For self-consistency based input-output pairs, first copy the pseudo test cases into the prefix data, run `prefix_fail_extract_pseudo_label.py`,
and the run this script.


Note:
1. Sometimes the pseudo test cases can include some extremely large cases, leading to out-of-memory error.
"""


def counting_partial_response_value(full_res):
    pass_num = []
    for pred_res in full_res:
        if len(pred_res) == 0:
            pass_num.append(0)
        # elif pred_res[0] == -1:  # FIXME: This line is commented since 2024/09/25. Seems no difference.
        #     pass_num.append(0)
        else:
            pass_num.append(sum([1 for x in pred_res if x is True]))

    return pass_num


def annotate(file, exclude: str = ""):
    exclude = exclude.split(",")
    if any([e and e in file for e in exclude]):
        print(f"Excluding {file}")
        return []
    return json.load(open(file, encoding="utf-8"))


def multiprocessing_loading(files, exclude: str = "", num_workers: int = 8):
    _annotate = partial(annotate, exclude=exclude)

    # with Pool(num_workers) as p:
    #     data = list(tqdm(p.imap(_annotate, files), total=len(files)))
    # all_data = []
    # for d in data:
    #     all_data.extend(d)
    # return all_data
    data = []
    for file in tqdm(files):
        data += _annotate(file)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--pass_case_margin", type=float, default=1)
    parser.add_argument("--pass_case_lower_bound", type=float, default=0.5)
    parser.add_argument("--test_case_field", type=str, default="pseudo_input_output")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--reduction", type=str, default="max")
    parser.add_argument("--exclude", type=str, default="")
    args = parser.parse_args()

    print("Collecting data...")
    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        files = glob(args.input_file)
        files = sorted(files)
        print(len(files))
        print(files)
        data = multiprocessing_loading(files, args.exclude)

    print(len(data))

    num_prefixes = 0
    val_cnt = collections.Counter()
    outputs = []
    p_id2prefixes = collections.defaultdict(list)
    preference_pairs = []
    missing = 0
    missing_test_cases = 0
    for item in tqdm(data):
        problem_id, resp_id, prefix_id = item["prefix_id"].split("_")
        prefix = item["prefix"]
        # problem_id = int(problem_id)

        if "res" not in item:
            missing += 1
            continue

        if args.test_case_field not in item or (not isinstance(item[args.test_case_field], dict)) or "inputs" not in item[args.test_case_field]:
            missing_test_cases += 1
            continue

        test_case_num = len(item[args.test_case_field]["inputs"])
        if test_case_num == 0:
            missing_test_cases += 1
            continue

        pass_num = counting_partial_response_value(item["full_res"])
        if args.reduction == "max":
            max_pass_num = max(pass_num)
        elif args.reduction == "avg":
            max_pass_num = sum(pass_num) / len(pass_num)
        else:
            raise NotImplementedError

        outputs.append({
            "problem_id": problem_id,
            "prefix": prefix,
            "pass_num": pass_num,
            "max_pass_num": max_pass_num,
            "test_case_num": test_case_num,
        })
        num_prefixes += 1
        val_cnt.update(pass_num)

        p_id2prefixes[problem_id].append(outputs[-1])

    for problem_id, all_prefixes in tqdm(p_id2prefixes.items()):
        max_pass_num2prefixes = collections.defaultdict(list)
        for prefix in all_prefixes:
            max_pass_num2prefixes[prefix["max_pass_num"]].append(prefix)
        max_pass_num2prefixes = sorted(max_pass_num2prefixes.items(), key=lambda x: x[0])

        pos_prefixes = []
        neg_prefixes = []
        for p in all_prefixes:
            pass_ratio = p["max_pass_num"] / p["test_case_num"]
            if pass_ratio < args.pass_case_lower_bound:
                continue
            neg_upper_pass_num = p["max_pass_num"] - args.pass_case_margin

            target_neg = []
            for pass_num, prefixes in max_pass_num2prefixes:
                if pass_num < neg_upper_pass_num:
                    target_neg.extend([x["prefix"] for x in prefixes])
                else:
                    break

            pos_prefixes.append(p["prefix"])
            neg_prefixes.append(target_neg)

        preference_pairs.append({
            "problem_id": problem_id,
            "pos": pos_prefixes,
            "neg": neg_prefixes
        })

    print(f"Missing: {missing}")
    print(f"Missing test cases: {missing_test_cases}")
    print(val_cnt)
    print(f"Processed {num_prefixes} prefixes.")
    print(f"Averaged {num_prefixes / len(data)} prefixes per problem.")
    print(f"Processed {len(preference_pairs)} problems.")
    json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(preference_pairs, open(args.output_file.replace(".json", f"_low{args.pass_case_lower_bound}_m{args.pass_case_margin}_{args.reduction}.json"),
                                     "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "/mnt/fangkai_blob/reward_modeling//experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.[0-9]*-of-256.pseudo_test_case.exec.json" \
    --output_file /mnt/fangkai_blob/reward_modeling//experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.V100.w8.v3.1.dp4.tp4.s42/apps/checkpoint-200/train.tem1.0.n10.prefix.upper0.8.r0.3.completion.tem1.0.n5.v2.0.pseudo_test_case.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.8 --pass_case_margin 4 --test_case_field pseudo_input_output


>>> python scripts/apps/prm/construct_process_rm_sample_fix.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.tem1.0.n10.prefix.upper0.8.r0.3.sample20_per.completion.tem1.0.n3.pseudo_input_output.exec.*-of-256.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.sft_ps_test_case.process-dpo.V100.tp8dp16.v4.9.s42/oss-instruct-apps-train/checkpoint-700/train.tem1.0.n10.prefix.upper0.8.r0.3.sample20_per.completion.tem1.0.n3.pseudo_input_output.prefix_pass_num.fix.json \
    --pass_case_lower_bound 0.5 --pass_case_margin 4 --test_case_field pseudo_input_output --reduction avg --test_case_field input_output --exclude "204-of-256,225-of-256"

Missing: 0
Missing test cases: 0
Counter({10: 448824, 0: 153053, 9: 25420, 1: 23469, 8: 16718, 5: 15115, 2: 14803, 3: 12821, 4: 12253, 7: 12053, 6: 11947, 11: 724, 20: 112, 16: 3})  # TODO: This should be a problem. Why there are some problems have more than 10 test cases?
Processed 249105 prefixes.
Averaged 1.0 prefixes per problem.
"""
