import json
import argparse
import os.path
from glob import glob
from tqdm import tqdm
import collections
from multiprocessing import Pool
from functools import partial
import torch
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from post_processors.openai_api_callback import majority_voting_predict
from data.deepseek_math_utils import eval_script


def load_rewards(reward_file):
    if os.path.exists(reward_file):
        rewards = json.load(open(reward_file))
    else:
        rewards = []
        for file in glob(reward_file):
            print(file)
            rewards.extend(json.load(open(file)))

    return rewards


def _init(id2reward):
    global _id2reward
    _id2reward = id2reward


def _worker(item, sc_top_k=None):
    sorted_results = []
    if not item["response"] or not item["pred"] or not item["res"]:
        return {
            "missing": 1,
            "reward_missing": 0,
            "pred_missing": 0,
            "seq_too_long": 0,
            "sorted_results": [],
            "sc_res": False
        }
    reward_missing = 0
    pred_missing = 0
    seq_too_long = 0
    for i, (resp, pred, r) in enumerate(zip(item["response"], item["pred"], item["res"])):
        resp_id = f"{item['id']}_{i}"
        if resp_id not in _id2reward:
            reward_missing += 1
            continue
        if not pred:
            pred_missing += 1
            continue

        reward = _id2reward[resp_id]
        sorted_results.append((resp, pred, r, reward))
    sorted_results = sorted(sorted_results, key=lambda x: x[-1], reverse=True)

    sc_top_k_res = {}
    if sc_top_k and sorted_results:
        for k in sc_top_k:
            preds = [r[1] for r in sorted_results[:k]]
            sc_pred = majority_voting_predict(preds)
            if sc_pred != "":
                sc_res = eval_script.eval_math({"prediction": sc_pred, "answer": item["label"]})
            else:
                sc_res = False
            sc_top_k_res[k] = sc_res

    return {
        "missing": 0,
        "reward_missing": reward_missing,
        "pred_missing": pred_missing,
        "seq_too_long": seq_too_long,
        "sorted_results": sorted_results,
        "sc_res": item["sc_res"],
        "sc_top_k_res": sc_top_k_res,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    parser.add_argument("--reward_file", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    if os.path.exists(args.response_file):
        responses = json.load(open(args.response_file))
    else:
        responses = []
        for file in glob(args.response_file):
            print(file)
            responses.extend(json.load(open(file)))

    rewards = load_rewards(args.reward_file)
    id2reward = {item["index"]: item["reward"][1] for item in rewards}

    _k = [1, 3, 5]
    orm_pass_at_k = {k: 0 for k in _k}
    missing = 0
    missing_reward = 0
    pred_missing = 0
    seq_too_long = 0
    sc_cnt = 0
    # for item in tqdm(responses):
    #     sorted_results = []
    #     if not item["response"] or not item["pred"] or not item["res"]:
    #         missing += 1
    #         continue
    #     for i, (resp, pred, r) in enumerate(zip(item["response"], item["pred"], item["res"])):
    #         resp_id = f"{item['id']}_{i}"
    #         if resp_id not in id2reward:
    #             missing_reward += 1
    #             continue
    #
    #         reward = id2reward[resp_id]
    #         sorted_results.append((resp, pred, r, reward))
    #
    #     if not sorted_results:
    #         continue
    #     sorted_results = sorted(sorted_results, key=lambda x: x[-1], reverse=True)
    #     for k in _k:
    #         if any([r[2] for r in sorted_results[:k]]):
    #             orm_pass_at_k[k] += 1
    #
    #     if item["sc_res"]:
    #         sc_cnt += 1
    with Pool(args.num_workers, initializer=_init, initargs=(id2reward,)) as pool:
        annotate = partial(_worker, sc_top_k=(5, 10))
        results = list(tqdm(pool.imap_unordered(annotate, responses), total=len(responses)))

    sc_top_k = {k: 0 for k in (5, 10)}
    for item in results:
        missing += item["missing"]
        missing_reward += item["reward_missing"]
        pred_missing += item["pred_missing"]
        seq_too_long += item["seq_too_long"]
        sorted_results = item["sorted_results"]

        if item["sc_res"]:
            sc_cnt += 1

        if not sorted_results:
            continue

        # ultimate_results.append(sorted_results)
        for k in _k:
            if any([r[2] for r in sorted_results[:k]]):
                orm_pass_at_k[k] += 1

        for k, v in item["sc_top_k_res"].items():
            if v:
                sc_top_k[k] += 1

    print(f"Total: {len(responses)}")
    print(f"Missing: {missing}")
    print(f"Missing reward: {missing_reward}")
    print(f"Missing pred: {pred_missing}")
    print(f"Seq too long: {seq_too_long}")
    print(f"SC: {sc_cnt}")
    for k, v in orm_pass_at_k.items():
        print(f"PRM pass at {k}: {v}")
        print(f"PRM pass at {k} rate: {v / len(responses) * 100:.2f}%")
    for k, v in sc_top_k.items():
        print(f"SC pass at {k}: {v}")
        print(f"SC pass at {k} rate: {v / len(responses) * 100:.2f}%")
    print(f"SC rate: {sc_cnt / len(responses) * 100:.2f}%")


if __name__ == '__main__':
    main()
