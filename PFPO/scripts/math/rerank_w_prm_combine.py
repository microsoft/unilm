import json
import argparse
import os.path
from glob import glob
from tqdm import tqdm
import collections
import torch


def load_rewards(reward_file):
    if os.path.exists(reward_file):
        rewards = json.load(open(reward_file))
    else:
        rewards = []
        for file in glob(reward_file):
            print(file)
            rewards.extend(json.load(open(file)))

    return rewards


def reward_reduction(ending_logits, reduction: str = "min", norm: bool = True):
    if norm:
        ending_logits = torch.tensor(ending_logits)
        ending_probs = torch.softmax(ending_logits, dim=-1).tolist()
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


def merge_rewards(group_logits, weights, reduction: str = "min", norm: bool = True, group_reduction: str = "min"):
    # rewards = []
    # for g_id, item in group_logits.items():
    #     if len(item["ending_logits"]) == 0:
    #         continue
    #     r = reward_reduction(item["ending_logits"], reduction, norm)
    #     rewards.append(weights[g_id] * r)
    # if len(rewards) == 0:
    #     return 0
    # if group_reduction == "min":
    #     reward = min(rewards)
    # elif group_reduction == "product":
    #     reward = 1
    #     for r in rewards:
    #         reward *= r
    # elif group_reduction == "sum":
    #     reward = sum(rewards)
    # else:
    #     raise ValueError(f"Invalid group reduction method: {group_reduction}")
    ending_logits = []
    if len(group_logits) == 0:
        return 0
    _len = len(group_logits[0]["ending_logits"])
    if _len == 0:
        return 0
    for g_id, item in group_logits.items():
        assert len(item["ending_logits"]) == _len
        ending_logits.append(item["ending_logits"])
    ending_logits = torch.tensor(ending_logits)
    assert ending_logits.size() == (len(group_logits), _len, 2), ending_logits.size()

    if norm:
        ending_logits = torch.softmax(ending_logits, dim=-1)
        ending_logits = ending_logits[:, :, 1]
    else:
        ending_logits = ending_logits[:, :, 1]

    ending_logits = ending_logits * torch.tensor(weights).view(-1, 1)

    if group_reduction == "min":
        ending_logits = torch.min(ending_logits, dim=0)[0]
    elif group_reduction == "product":
        ending_logits = torch.prod(ending_logits, dim=0)
    elif group_reduction == "sum":
        ending_logits = torch.sum(ending_logits, dim=0)
    else:
        raise ValueError(f"Invalid group reduction method: {group_reduction}")

    if reduction == "min":
        reward = torch.min(ending_logits).item()
    elif reduction == "product":
        reward = torch.prod(ending_logits).item()
    elif reduction == "sum":
        reward = torch.sum(ending_logits).item()
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")

    return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file", type=str, required=True)
    parser.add_argument("--reward_file", type=str, required=True, nargs='+')
    parser.add_argument("--weights", type=float, nargs='+', default=[1.0])
    parser.add_argument("--reduction", type=str, default="min")
    parser.add_argument("--raw_logits", action="store_true", default=False)
    parser.add_argument("--group_reduction", type=str, default="min")
    args = parser.parse_args()

    if os.path.exists(args.response_file):
        responses = json.load(open(args.response_file))
    else:
        responses = []
        for file in glob(args.response_file):
            print(file)
            responses.extend(json.load(open(file)))

    id2reward = collections.defaultdict(dict)
    assert len(args.reward_file) == len(args.weights)
    print(args.reward_file)
    print(args.weights)
    for _g_id, _reward_group in enumerate(args.reward_file):
        rewards = load_rewards(_reward_group)
        for r in rewards:
            id2reward[r["index"]][_g_id] = r

    _k = [1, 3, 5]
    prm_pass_at_k = {k: 0 for k in _k}
    missing = 0
    missing_reward = 0
    sc_cnt = 0
    ultimate_results = []
    for item in tqdm(responses):
        sorted_results = []
        if not item["response"] or not item["pred"] or not item["res"]:
            missing += 1
            continue
        for i, (resp, pred, r) in enumerate(zip(item["response"], item["pred"], item["res"])):
            resp_id = f"{item['id']}_{i}"
            if resp_id not in id2reward:
                missing_reward += 1
                continue
            process_rewards = id2reward[resp_id]
            # if len(process_rewards["ending_logits"]) == 0:
            #     continue

            # reward = merge_rewards(process_rewards["ending_logits"], args.reduction, not args.raw_logits)
            reward = merge_rewards(process_rewards, args.weights, args.reduction, not args.raw_logits, args.group_reduction)
            sorted_results.append((resp, pred, r, reward))

        if not sorted_results:
            continue
        sorted_results = sorted(sorted_results, key=lambda x: x[-1], reverse=True)
        ultimate_results.append(sorted_results)
        for k in _k:
            if any([r[2] for r in sorted_results[:k]]):
                prm_pass_at_k[k] += 1

        if item["sc_res"]:
            sc_cnt += 1

    print(f"Total: {len(responses)}")
    print(f"Missing: {missing}")
    print(f"Missing reward: {missing_reward}")
    print(f"SC: {sc_cnt}")
    for k, v in prm_pass_at_k.items():
        print(f"PRM pass at {k}: {v}")
        print(f"PRM pass at {k} rate: {v / len(responses) * 100:.2f}%")
    print(f"SC rate: {sc_cnt / len(responses) * 100:.2f}%")

    json.dump(ultimate_results[:100], open("debug.json", "w"), indent=2)


if __name__ == '__main__':
    main()
