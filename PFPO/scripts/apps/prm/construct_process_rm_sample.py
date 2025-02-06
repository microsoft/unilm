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

This script is incorrect for pseudo test cases since we cannot ensure the correctness of each test case, but it is appropriate for ground-truth test cases,
serving as hard limit, i.e., if there is one completion for some prefix has passed all test cases, then it is a gold prefix.

"""


def counting_partial_response_value(res):
    return sum([1 if x else 0 for x in res])


def parse_value(v, binary: bool):
    if binary:
        return 1 if v > 0 else 0
    return v


def _process_trajectories_worker(item, top_k: int, binary: bool):
    item_id, trajectories = item
    outputs = {
        "idx": item_id
    }
    for i in range(top_k):
        level_trajectories = [(traj["vs"][i], traj["prefix"]) for traj in trajectories if len(traj["vs"]) > i]
        if len(level_trajectories) == 0:
            continue
        prefix_vis = set()
        level_values = []
        level_prefixes = []
        for v, prefix in level_trajectories:
            if prefix in prefix_vis:
                continue
            level_values.append(parse_value(v, binary))
            level_prefixes.append(prefix)
            prefix_vis.add(prefix)
        outputs[f"traj_level_{i}_values"] = level_values
        outputs[f"traj_level_{i}_prefixes"] = level_prefixes
    return outputs


def _annotate(file):
    return json.load(open(file, encoding="utf-8"))


def multiprocessing_loading(files, num_workers: int = 8):
    with Pool(num_workers) as p:
        data = list(tqdm(p.imap(_annotate, files), total=len(files)))
    all_data = []
    for d in data:
        all_data.extend(d)
    return all_data
    # data = []
    # for file in tqdm(files):
    #     data += json.load(open(file, encoding="utf-8"))
    # return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--binary", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    print("Collecting data...")
    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        # data = []
        # for file in glob(args.input_file):
        #     print(file)
        #     data += json.load(open(file))
        files = glob(args.input_file)
        files = sorted(files)
        print(len(files))
        print(files)
        data = multiprocessing_loading(files)

    print(len(data))

    num_prefixes = 0
    val_cnt = collections.Counter()
    outputs = []
    preference_pairs = dict()
    missing = 0
    for item in tqdm(data):
        problem_id, resp_id, prefix_id = item["prefix_id"].split("_")
        prefix = item["prefix"]
        problem_id = int(problem_id)

        if "res" not in item:
            missing += 1
            continue
        v = counting_partial_response_value(item["res"])
        v = parse_value(v, args.binary)

        outputs.append({
            "problem_id": problem_id,
            "prefix": prefix,
            "value": v,
        })
        num_prefixes += 1
        val_cnt[v] += 1

        if problem_id not in preference_pairs:
            preference_pairs[problem_id] = {
                "pos": [],
                "neg": [],
            }

        if v > 0:
            preference_pairs[problem_id]["pos"].append(prefix)
        else:
            preference_pairs[problem_id]["neg"].append(prefix)

    preference_pairs = [
        {
            "problem_id": problem_id,
            "pos": pair["pos"],
            "neg": pair["neg"],
        }
        for problem_id, pair in preference_pairs.items()
    ]

    print(f"Missing: {missing}")
    print(val_cnt)
    print(f"Processed {num_prefixes} prefixes.")
    print(f"Averaged {num_prefixes / len(data)} prefixes per problem.")
    json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(preference_pairs, open(args.output_file.replace(".json", "_pairs.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()

"""
>>> 
"""
