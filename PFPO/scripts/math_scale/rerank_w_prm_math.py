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


def load_rewards(reward_file, re_index):
    if os.path.exists(reward_file):
        rewards = json.load(open(reward_file))
    else:
        rewards = []
        for i, file in enumerate(sorted(glob(reward_file))):
            print(file)
            sub_rewards = json.load(open(file))
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
        # ending_logits = torch.tensor(ending_logits)
        # ending_probs = torch.softmax(ending_logits, dim=-1).tolist()
        ending_probs = softmax(ending_logits)
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


def _worker(item, reduction, norm, sc_top_k=None):
    if not item["response"] or not item["pred"] or not item["res"]:
        return {
            "missing": 1,
            "reward_missing": 0,
            "pred_missing": 0,
            "seq_too_long": 0,
            "sorted_results": [],
            "sc_res": False
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
    sorted_results = sorted(unsorted_results, key=lambda x: x[-1], reverse=True)

    sc_top_k_res = {}
    if sc_top_k and sorted_results:
        for k in sc_top_k:
            preds = [r[1] for r in sorted_results[:k]]
            sc_pred = majority_voting_predict(preds)
            if sc_pred != "":
                sc_res = mathscale_is_equiv(sc_pred, item["label"])[0]
            else:
                sc_res = False
            sc_top_k_res[k] = sc_res

    sc_k_res = {}
    if sc_top_k and unsorted_results:
        for k in sc_top_k:
            preds = [r[1] for r in unsorted_results[:k]]
            sc_pred = majority_voting_predict(preds)
            if sc_pred != "":
                sc_res = mathscale_is_equiv(sc_pred, item["label"])[0]
            else:
                sc_res = False
            sc_k_res[k] = sc_res

    weighted_best_of_k = {}
    if sc_top_k and unsorted_results:
        for k in sc_top_k:
            preds = [r[1] for r in unsorted_results[:k]]
            # weights = torch.softmax(torch.tensor([r[-1] for r in unsorted_results]), dim=0).tolist()
            weights = softmax(np.array([r[-1] for r in unsorted_results])).tolist()
            best_of_k_pred = weighted_majority_voting_predict(preds, weights)
            if best_of_k_pred != "":
                best_of_k_res = mathscale_is_equiv(best_of_k_pred, item["label"])[0]
            else:
                best_of_k_res = False
            weighted_best_of_k[k] = best_of_k_res

    return {
        "missing": 0,
        "reward_missing": reward_missing,
        "pred_missing": pred_missing,
        "seq_too_long": seq_too_long,
        "sorted_results": sorted_results,
        "sc_res": item["sc_res"],
        "sc_top_k_res": sc_top_k_res,
        "weighted_best_of_k": weighted_best_of_k,
        "sc_k_res": sc_k_res,
    }


def merge_key(item, value):
    assert isinstance(item, list)
    if isinstance(value, list):
        item = item + value
    else:
        item.append(value)
    return item


def merge_seed_sampled_data(data):
    id2data = {}
    for item in data:
        if item["id"] not in id2data:
            id2data[item["id"]] = item
            continue

        tmp = id2data[item["id"]]
        if isinstance(tmp["response"], str):
            tmp["response"] = [tmp["response"]]
        if not isinstance(tmp["res"], list):
            tmp["res"] = [tmp["res"]]
        if not isinstance(tmp["pred"], list):
            tmp["pred"] = [tmp["pred"]]

        tmp["response"] = merge_key(tmp["response"], item["response"])
        tmp["res"] = merge_key(tmp["res"], item["res"])
        tmp["pred"] = merge_key(tmp["pred"], item["pred"])
        assert isinstance(tmp["pred"], list), tmp["pred"]
        id2data[item["id"]] = tmp

    return list(id2data.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    parser.add_argument("--reward_file", type=str, required=True)
    parser.add_argument("--reduction", type=str, default="min")
    parser.add_argument("--raw_logits", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--re_index", action="store_true", default=False)
    parser.add_argument("--keep_top_k", type=str, default="")
    args = parser.parse_args()

    if os.path.exists(args.response_file):
        responses = json.load(open(args.response_file))
    else:
        responses = []
        for file in glob(args.response_file):
            print(file)
            responses.extend(json.load(open(file)))

    responses = merge_seed_sampled_data(responses)

    print(len(responses[0]["response"]), len(responses[0]["pred"]), len(responses[0]["res"]))
    print(responses[0]["id"])

    rewards = load_rewards(args.reward_file, args.re_index)
    print(rewards[0]['index'])
    id2reward = {item["index"]: item for item in rewards}

    # _k = [1, 3, 5]
    _k = [3, 5, 4, 8, 16, 32, 64, 128]
    prm_pass_at_k = {k: 0 for k in _k}
    missing = 0
    missing_reward = 0
    pred_missing = 0
    seq_too_long = 0
    sc_cnt = 0
    ultimate_results = []
    with Pool(args.num_workers, initializer=_init, initargs=(id2reward,)) as pool:
        annotate = partial(_worker, reduction=args.reduction, norm=(not args.raw_logits), sc_top_k=_k)
        results = list(tqdm(pool.imap_unordered(annotate, responses), total=len(responses)))

    sc_top_k = {k: 0 for k in _k}
    weighted_best_of_k = {k: 0 for k in _k}
    sc_k = {k: 0 for k in _k}
    outputs = []
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

        ultimate_results.append(sorted_results)
        for k in _k:
            if any([r[2] for r in sorted_results[:k]]):
                prm_pass_at_k[k] += 1

        for k, v in item["sc_top_k_res"].items():
            if v:
                sc_top_k[k] += 1

        for k, v in item["weighted_best_of_k"].items():
            if v:
                weighted_best_of_k[k] += 1

        for k, v in item["sc_k_res"].items():
            if v:
                sc_k[k] += 1

    print(f"Total: {len(responses)}")
    print(f"Missing: {missing}")
    print(f"Missing reward: {missing_reward}")
    print(f"Missing pred: {pred_missing}")
    print(f"Seq too long: {seq_too_long}")
    print(f"SC: {sc_cnt}")
    for k, v in prm_pass_at_k.items():
        print(f"PRM pass at {k}: {v}")
        print(f"PRM pass at {k} rate: {v / len(responses) * 100:.2f}%")
    for k, v in sc_top_k.items():
        print(f"SC pass at {k}: {v}")
        print(f"SC pass at {k} rate: {v / len(responses) * 100:.2f}%")
    print(f"SC rate: {sc_cnt / len(responses) * 100:.2f}%")
    for k, v in weighted_best_of_k.items():
        print(f"Weighted best of k pass at {k}: {v}")
        print(f"Weighted best of k pass at {k} rate: {v / len(responses) * 100:.2f}%")
    for k, v in sc_k.items():
        print(f"SC k pass at {k}: {v}")
        print(f"SC k pass at {k} rate {v / len(responses) * 100:.2f}%")

    json.dump(ultimate_results[:100], open("debug.json", "w"), indent=2)


if __name__ == '__main__':
    main()
