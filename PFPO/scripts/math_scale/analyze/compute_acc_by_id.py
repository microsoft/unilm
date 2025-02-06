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
    parser.add_argument("--maj_ks", type=str, default="8,16,32,64,128")
    parser.add_argument("--id_file", type=str)
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

    id_data = json.load(open(args.id_file))
    all_ids = set([item["id"] for item in id_data])
    data = [item for item in data if item["id"] in all_ids]

    maj_ks = list(map(int, args.maj_ks.split(",")))

    cnt = 0
    pass_at_k = 0
    maj_at_k = {k: 0 for k in maj_ks}
    acc_data_topic = collections.Counter()
    cnt_data_topic = collections.Counter()
    avg_solution_num = 0
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

        pred2res = {pred: res[j] for j, pred in enumerate(item["pred"])}

        avg_solution_num += len(item["pred"])

        for _k in maj_ks:
            k_preds = item["pred"][:_k]
            k_sc_pred = majority_voting_frequency(k_preds)[0][0]
            k_sc_res = pred2res[k_sc_pred]
            if k_sc_res:
                maj_at_k[_k] += 1

        sc_pred = majority_voting_frequency(item["pred"])[0][0]
        item["sc_pred"] = sc_pred
        item["sc_res"] = pred2res[sc_pred]

    assert pass_at_k <= len(data)
    json.dump(data, open(args.output_file, "w"))

    metrics = {"acc": cnt / len(data), "pass@k": pass_at_k / len(data), "correct": cnt, "total": len(data)}
    if len(acc_data_topic) > 0:
        for k, v in acc_data_topic.items():
            metrics[f"acc_{k}"] = v / cnt_data_topic[k]
            metrics[f"total_{k}"] = cnt_data_topic[k]
    metrics["avg_solution_num"] = avg_solution_num / len(data)
    for _k in maj_ks:
        metrics[f"maj@{_k}"] = maj_at_k[_k] / len(data)
    json.dump(metrics, open(args.output_file.replace(".json", ".metrics.json"), "w"), indent=2)

    print(len(data))
    print(metrics)


if __name__ == '__main__':
    main()
