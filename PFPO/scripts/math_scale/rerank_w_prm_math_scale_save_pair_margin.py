import json
import argparse
import os.path
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import sys
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from post_processors.openai_api_callback import majority_voting_predict
from data.mathscale.util import mathscale_is_equiv

"""
This script is used to rerank according to different measure of rewards and save the sorted responses for rejection sampling.
"""


def load_rewards(reward_file, re_index):
    if os.path.exists(reward_file):
        print(reward_file)
        rewards = json.load(open(reward_file))
    else:
        rewards = []
        for i, file in enumerate(sorted(glob(reward_file))):
            print(file)
            try:
                sub_rewards = json.load(open(file))
            except Exception as e:
                print(f"Error in {file}: {e}")
                continue
            if re_index:
                for item in sub_rewards:
                    item["index"] = f"{item['index']}_{i}"
            rewards.extend(sub_rewards)

    return rewards


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def reward_reduction(ending_logits, reduction: str = "min", norm: bool = True):
    if norm:
        ending_probs = softmax(ending_logits).tolist()
        step_rewards = [step[1] for step in ending_probs]
    else:
        step_rewards = [step[1] for step in ending_logits]
    if reduction == "min":
        reward = min(step_rewards)
    elif reduction == "product":
        reward = 1
        for prob in step_rewards:
            reward *= prob
    elif reduction == "sum":
        reward = sum(step_rewards)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    return reward


def weighted_majority_voting_predict(preds, weights):
    pred2weight = {}
    for pred, weight in zip(preds, weights):
        if pred not in pred2weight:
            pred2weight[pred] = 0
        pred2weight[pred] += weight
    return max(pred2weight, key=pred2weight.get)


def _init(id2reward):
    global _id2reward
    _id2reward = id2reward


def _worker(item, reduction, norm, sc_type: str, include_incorrect: bool, top_k: int, margin: float = 0.2):
    if not item["response"] or not item["pred"] or not item["res"]:
        return {
            "id": item["id"],
            "missing": 1,
            "reward_missing": 0,
            "pred_missing": 0,
            "seq_too_long": 0,
            "unsorted_results": [],
            "top_k_sorted_results": [],
        }
    unsorted_results = []
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

        process_rewards = _id2reward[resp_id]
        if len(process_rewards["ending_logits"]) == 0:
            seq_too_long += 1
            continue

        assert resp == process_rewards["response"], f"{resp} \n\n {process_rewards['response']} \n\n ========="

        reward = reward_reduction(process_rewards["ending_logits"], reduction, norm)
        unsorted_results.append((resp, pred, r, reward))

    if len(unsorted_results) == 0:
        return {
            "id": item["id"],
            "missing": 1,
            "reward_missing": reward_missing,
            "pred_missing": pred_missing,
            "seq_too_long": seq_too_long,
            "unsorted_results": [],
            "top_k_sorted_results": [],
        }

    # Decide the self-consistency-based pseudo label
    if sc_type == "majority":
        sc_pred = majority_voting_predict(item["pred"])
    elif sc_type == "bon":
        preds = [r[1] for r in unsorted_results]
        # weights = torch.softmax(torch.tensor([r[-1] for r in unsorted_results]), dim=0).tolist()
        weights = softmax([r[-1] for r in unsorted_results]).tolist()
        assert len(preds) == len(weights)
        sc_pred = weighted_majority_voting_predict(preds, weights)
    else:
        raise ValueError(f"Invalid self-consistency type: {sc_type}")

    sorted_results = sorted(unsorted_results, key=lambda x: x[-1], reverse=True)

    chosen = []
    reject = []
    for i in range(len(sorted_results)):
        if not include_incorrect and not mathscale_is_equiv(sorted_results[i][1], sc_pred)[0]:
            continue
        if len(chosen) >= top_k:
            break
        pos = sorted_results[i]
        neg_list = []
        for j in range(i + 1, len(sorted_results)):
            neg = sorted_results[j]
            if pos[-1] - neg[-1] >= margin:
                neg_list.append(neg[0])

        if neg_list:
            chosen.append(pos[0])
            reject.append(neg_list)

    return {
        "id": item["id"],
        "missing": 0,
        "reward_missing": reward_missing,
        "pred_missing": pred_missing,
        "seq_too_long": seq_too_long,
        "unsorted_results": unsorted_results,
        "top_k_sorted_results": sorted_results[:top_k],
        "sc_pred": sc_pred,
        "top_k_chosen": chosen,
        "top_k_reject": reject,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    parser.add_argument("--reward_file", type=str, required=True)
    parser.add_argument("--reduction", type=str, default="min")
    parser.add_argument("--raw_logits", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--re_index", action="store_true", default=False)
    parser.add_argument("--include_incorrect", action="store_true", default=False)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sc_type", type=str, choices=["majority", "bon"])
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    print(os.cpu_count())

    if os.path.exists(args.response_file):
        print(args.response_file)
        responses = json.load(open(args.response_file))
    else:
        responses = []
        for file in glob(args.response_file):
            print(file)
            responses.extend(json.load(open(file)))

    print(len(responses[0]["response"]), len(responses[0]["pred"]), len(responses[0]["res"]))
    print(responses[0]["id"])
    print(len(responses))

    rewards = load_rewards(args.reward_file, args.re_index)
    print(rewards[0]['index'])
    print(len(rewards))
    id2reward = {item["index"]: item for item in rewards}

    id2responses = {item["id"]: item for item in responses}

    missing = 0
    missing_reward = 0
    pred_missing = 0
    seq_too_long = 0
    with Pool(args.num_workers, initializer=_init, initargs=(id2reward,)) as pool:
        annotate = partial(_worker, reduction=args.reduction, norm=(not args.raw_logits), sc_type=args.sc_type, include_incorrect=args.include_incorrect,
                           top_k=args.top_k, margin=args.margin)
        results = list(tqdm(pool.imap_unordered(annotate, responses), total=len(responses)))

    outputs = []
    num_pairs = 0
    for item in results:
        missing += item["missing"]
        missing_reward += item["reward_missing"]
        pred_missing += item["pred_missing"]
        seq_too_long += item["seq_too_long"]

        sorted_results = item["top_k_sorted_results"]

        if not sorted_results:
            continue

        orig_item = id2responses[item["id"]]
        orig_item.pop("response")
        orig_item.pop("pred")
        if "sc_res" in orig_item:
            orig_item.pop("sc_res")
        if "sc_pred" in orig_item:
            orig_item.pop("sc_pred")
        orig_item.pop("res")

        orig_item["top_k_response"] = [r[0] for r in sorted_results]
        orig_item["top_k_pred"] = [r[1] for r in sorted_results]
        orig_item["top_k_reward"] = [r[3] for r in sorted_results]

        orig_item["top_k_chosen"] = item["top_k_chosen"]
        orig_item["top_k_reject"] = item["top_k_reject"]
        num_pairs += len(item["top_k_chosen"])

        outputs.append(orig_item)

    print(f"Total: {len(responses)}")
    print(f"Missing: {missing}")
    print(f"Missing reward: {missing_reward}")
    print(f"Missing pred: {pred_missing}")
    print(f"Seq too long: {seq_too_long}")
    print(f"Num pairs: {num_pairs}")
    json.dump(outputs, open(args.output_file, "w"))


if __name__ == '__main__':
    main()
