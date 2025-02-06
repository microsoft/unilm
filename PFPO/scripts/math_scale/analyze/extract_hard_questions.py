import json
import argparse
from glob import glob
import os
import collections
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.mathscale.util import mathscale_is_equiv

"""
The output file should be processed by post_processors.openai_api_callback.MathScaleCallBack
"""


def majority_voting_frequency(preds):
    assert isinstance(preds, list), preds
    if isinstance(preds[0], list):
        tmp = []
        for pred in preds:
            tmp.append(str(sorted(pred)))
        tmp = collections.Counter(tmp)
        tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)
        sorted_preds = [(eval(pred), fre) for pred, fre in tmp]
    elif isinstance(preds[0], str):
        tmp = collections.Counter(preds)
        tmp = sorted(tmp.items(), key=lambda x: x[1], reverse=True)
        sorted_preds = [(pred, fre) for pred, fre in tmp]
    else:
        raise ValueError(f"Unknown type {type(preds[0])}")
    return sorted_preds


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
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--maj_k", type=int, default=16)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in sorted(glob(args.input_file)):
            if "metrics" in file:
                continue
            print(file)
            data += json.load(open(file))
        if len(data) == 0:
            raise ValueError(f"No data found in {args.input_file}")

    print(len(data))
    data = merge_seed_sampled_data(data)
    print(len(data))

    cnt = 0
    pass_at_k = 0
    maj_at_k = 0
    acc_data_topic = collections.Counter()
    cnt_data_topic = collections.Counter()
    avg_solution_num = 0
    outputs = []
    for item in tqdm(data):
        if not isinstance(item["res"], list):
            res = [item["res"]]
        else:
            res = item["res"]

        if res[0]:
            cnt += 1

        if "data_topic" in item:
            acc_data_topic[item["data_topic"]] += int(res[0])
            cnt_data_topic[item["data_topic"]] += 1

        if any(res):
            pass_at_k += 1

        if isinstance(item["pred"], str):
            item["pred"] = [item["pred"]]

        avg_solution_num += len(item["pred"])

        k_preds = item["pred"][:args.maj_k]
        k_sc_pred = majority_voting_frequency(k_preds)[0][0]
        k_sc_res = mathscale_is_equiv(k_sc_pred, item["label"])[0]
        if k_sc_res:
            maj_at_k += 1
        else:
            outputs.append(item)

        item[f"sc_pred@{args.maj_k}"] = k_sc_pred
        item[f"sc_res@{args.maj_k}"] = k_sc_res

    assert pass_at_k <= len(data)
    json.dump(outputs, open(args.output_file, "w"))

    metrics = {"acc": cnt / len(data), "pass@k": pass_at_k / len(data), "correct": cnt, "total": len(data)}
    if len(acc_data_topic) > 0:
        for k, v in acc_data_topic.items():
            metrics[f"acc_{k}"] = v / cnt_data_topic[k]
            metrics[f"total_{k}"] = cnt_data_topic[k]
    metrics["avg_solution_num"] = avg_solution_num / len(data)
    metrics[f"maj@{args.maj_k}"] = maj_at_k / len(data)
    json.dump(metrics, open(args.output_file.replace(".json", ".metrics.json"), "w"), indent=2)

    print(len(data))
    print(len(outputs))
    print(metrics)


if __name__ == '__main__':
    main()
