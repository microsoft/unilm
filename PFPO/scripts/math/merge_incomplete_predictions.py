import json
import argparse
from glob import glob
import os
import collections
import sys
from tqdm import tqdm


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.deepseek_math_utils import eval_script, answer_extraction


def pred2str(pred):
    if isinstance(pred, str):
        return pred
    if isinstance(pred, list):
        pred = sorted(pred)
        pred = str(pred)
        return pred
    raise ValueError(f"Unknown type {type(pred)}")


def load_data(file_path):
    if not file_path:
        return []
    if os.path.exists(file_path):
        data = json.load(open(file_path))
    else:
        data = []
        for file in glob(file_path):
            data += json.load(open(file))
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file1", type=str)
    parser.add_argument("--input_file2", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--n", type=int, default=-1)
    args = parser.parse_args()

    data1 = load_data(args.input_file1)
    data2 = load_data(args.input_file2)

    item_id2preds = {}
    sc = 0
    for item in tqdm(data1 + data2):
        if "prefix_id" in item:
            prefix_id = item["prefix_id"]
            tmp = prefix_id.split("_")
            item_id = "_".join(tmp[:-2])
        else:
            item_id = item["id"]

        if item_id not in item_id2preds:
            item_id2preds[item_id] = {
                "response": [],
                "pred": [],
                "label": item["label"],
            }

        item_id2preds[item_id]["response"].extend(item["response"])
        item_id2preds[item_id]["pred"].extend(item["pred"])

    for item, responses in tqdm(item_id2preds.items()):
        preds = responses["pred"]
        if args.n != -1:
            preds = preds[:args.n]
        str_preds = [pred2str(item) for item in preds]
        counter = collections.Counter(str_preds)
        sc_pred = eval(counter.most_common(1)[0][0])
        sc_res = eval_script.eval_math({"prediction": sc_pred, "answer": responses["label"]})
        if sc_res:
            sc += 1

    print(f"Self-consistency: {sc}/{len(item_id2preds)} = {sc / len(item_id2preds)}")


if __name__ == '__main__':
    main()
