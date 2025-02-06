import json
import argparse
import sys
import collections
import glob
import os

sys.set_int_max_str_digits(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--completion_file", type=str, required=True)
    parser.add_argument("--reward_file", type=str, required=True)
    args = parser.parse_args()

    solutions = json.load(open(args.completion_file))
    if "difficulty" in solutions[0]:
        all_diff = collections.Counter([item["difficulty"] for item in solutions])
        all_diff_success = {k: 0 for k in all_diff}
    else:
        all_diff = None
        all_diff_success = None

    if os.path.exists(args.reward_file):
        rewards = json.load(open(args.reward_file))
    else:
        rewards = []
        for file in glob.glob(args.reward_file):
            rewards += json.load(open(file))

    id2reward = collections.defaultdict(dict)
    for item in rewards:
        index = item["index"]
        orig_id, pred_id = list(map(int, index.split("_")))
        # id2reward[orig_id][pred_id] = item["reward"]
        if isinstance(item["reward"], list):
            id2reward[orig_id][pred_id] = item["reward"][0]
        else:
            id2reward[orig_id][pred_id] = item["reward"]

    success = 0
    for item in solutions:
        problem_id = item["id"]

        if not item["response"]:
            continue
        if not item["pred"]:
            continue

        tuples = []
        for j, (resp, pred) in enumerate(zip(item["response"], item["pred"])):
            if pred:
                tuples.append((j, id2reward[problem_id][j]))

        if not tuples:
            continue

        tuples = sorted(tuples, key=lambda x: x[1], reverse=True)
        pred_id = tuples[0][0]
        if item["res"][pred_id]:
            success += 1
            if all_diff is not None:
                all_diff_success[item["difficulty"]] += 1

    print(f"Success rate: {success / len(solutions)}")
    if all_diff is not None:
        for k, v in all_diff.items():
            print(f"Difficulty {k}: {all_diff_success[k] / v}")


if __name__ == '__main__':
    main()
