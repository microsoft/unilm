import json
import argparse
from glob import glob
import os
from tqdm import tqdm
import collections

"""
The output file should be processed by post_processors.openai_api_callback.MathScaleCallBack
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in glob(args.input_file):
            data += json.load(open(file))
        if len(data) == 0:
            raise ValueError(f"No data found in {args.input_file}")

    cnt = 0
    pass_at_k = 0
    sc = 0
    acc_data_topic = collections.Counter()
    cnt_data_topic = collections.Counter()
    for item in data:
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
        if item["sc_res"]:
            sc += 1

    assert pass_at_k <= len(data)
    json.dump(data, open(args.output_file, "w"), indent=2)

    metrics = {"acc": cnt / len(data), "pass@k": pass_at_k / len(data), "maj@k": sc / len(data), "correct": cnt, "total": len(data)}
    if len(acc_data_topic) > 0:
        for k, v in acc_data_topic.items():
            metrics[f"acc_{k}"] = v / cnt_data_topic[k]
            metrics[f"total_{k}"] = cnt_data_topic[k]
    json.dump(metrics, open(args.output_file.replace(".json", ".metrics.json"), "w"), indent=2)

    print(len(data))
    print(json.dumps(metrics, indent=2))
