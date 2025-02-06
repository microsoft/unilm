import argparse
import json
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from post_processors.openai_api_callback import majority_voting_predict
from data.deepseek_math_utils import eval_script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_format", type=str)
    parser.add_argument("--seed_list", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = []
    for seed in eval(args.seed_list):
        _file = args.input_file_format.format(seed)
        if os.path.exists(_file):
            print(f"Loading: {_file}")
            data.extend(json.load(open(_file)))
        else:
            print(f"File not found: {_file}")

    id2data = {}
    for item in tqdm(data):
        if item["id"] not in id2data:
            id2data[item["id"]] = item

            id2data[item["id"]]["response"] = [id2data[item["id"]]["response"]]
            id2data[item["id"]]["pred"] = [id2data[item["id"]]["pred"]]
            id2data[item["id"]]["res"] = [id2data[item["id"]]["res"]]
        else:
            assert item["text"] == id2data[item["id"]]["text"]

            id2data[item["id"]]["response"].append(item["response"])
            id2data[item["id"]]["pred"].append(item["pred"])
            id2data[item["id"]]["res"].append(item["res"])

    data = list(id2data.values())
    pass_at_k = 0
    maj_at_k = 0
    for item in tqdm(data):
        sc_pred = majority_voting_predict(item["pred"])
        if sc_pred != "":
            sc_res = eval_script.eval_math({"prediction": sc_pred, "answer": item["label"]})
        else:
            sc_res = False
        item["sc_pred"] = sc_pred
        item["sc_res"] = sc_res

        if any(item["res"]):
            pass_at_k += 1
        if sc_res:
            maj_at_k += 1

    print("Pass at k: ", pass_at_k / len(data))
    print("Maj at k: ", maj_at_k / len(data))
    json.dump(data, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
