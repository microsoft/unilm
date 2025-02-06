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
    parser.add_argument("--lower_bound", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=1.0)
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
        if isinstance(item["reward"], list):
            id2reward[orig_id][pred_id] = item["reward"][0]
        else:
            id2reward[orig_id][pred_id] = item["reward"]

    outputs = []
    success = 0
    num_pairs = 0
    for item in solutions:
        problem_id = item["id"]

        if not item["response"]:
            continue
        if not item["pred"]:
            continue

        assert isinstance(item["response"], list)
        assert isinstance(item["pred"], list)
        assert len(item["response"]) == len(item["pred"])

        tuples = []
        for j, (resp, pred) in enumerate(zip(item["response"], item["pred"])):
            if pred:
                tuples.append((j, resp, pred, id2reward[problem_id][j]))

        if not tuples:
            continue

        tuples = sorted(tuples, key=lambda x: x[3], reverse=True)
        # Construct preference pairs: collect pairs with margin greater than args.margin
        pos = []
        neg = []
        for i in range(len(tuples)):
            if tuples[i][3] < args.lower_bound:
                break
            for j in range(i + 1, len(tuples)):
                if tuples[i][3] - tuples[j][3] < args.margin:
                    continue
                pos.append(tuples[i][1])
                neg.append(tuples[j][1])
        outputs.append({
            "id": problem_id,
            "pos": pos,
            "neg": neg
        })
        num_pairs += len(pos)

        _pred_id = tuples[0][0]
        if item["res"] and item["res"][_pred_id]:
            success += 1
            if all_diff is not None:
                all_diff_success[item["difficulty"]] += 1

    print(f"Success rate: {success / len(solutions)}")
    if all_diff is not None:
        for k, v in all_diff.items():
            print(f"Difficulty {k}: {all_diff_success[k] / v}")

    json.dump(outputs, open(args.completion_file.replace(".json", f".lower-{args.lower_bound}.mar-{args.margin}.json"), "w"), ensure_ascii=False, indent=2)
    print(f"Number of pairs: {num_pairs}")


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/construct_prefer_pair_rm.py --completion_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.json --reward_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.pair-rm.gpt4o-worsen.A100.w8.v1.2.s42/r2c.sft.step-400.train.n10/test-checkpoint-50/eval_predictions_rank0.json --lower_bound 0.0 --margin 1.0

Success rate: 0.3566
Number of pairs: 29584
"""


