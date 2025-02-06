import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from functools import partial

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from data.mathscale.util import mathscale_is_equiv


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


def worker(item, n: int):
    sc_pred, sc_freq = majority_voting_frequency(item["pred"][:n])[0]
    sc_res = mathscale_is_equiv(sc_pred, item["label"])[0]
    return {
        "sc_res": sc_res,
        "sc_pred": sc_pred,
        "sc_freq": sc_freq,
        "id": item["id"],
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
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--ps", type=str,
                        help="The probabilities of top-1 frequency over total samples, split by comma")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--n", type=int, default=128)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in sorted(glob(args.input_file)):
            print(file)
            data += json.load(open(file))
        if len(data) == 0:
            raise ValueError(f"No data found in {args.input_file}")

    data = merge_seed_sampled_data(data)

    pbar = tqdm(data)
    outputs = []
    sc = 0
    num_pred = 0
    freq_pos = collections.Counter()
    freq = collections.Counter()
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        _annotate = partial(worker, n=args.n)
        for _input in pbar:
            future = executor.submit(_annotate, _input)
            num_pred += len(_input["response"])
            futures.append(future)
            pbar.update()

        for future in tqdm(as_completed(futures), total=len(futures), desc="Collecting results"):
            result = future.result()
            outputs.append(result)

            if result["sc_res"]:
                sc += 1

            if result["sc_res"]:
                freq_pos[result["sc_freq"]] += 1
            freq[result["sc_freq"]] += 1

    ps = list(map(float, args.ps.split(",")))
    reversed_acc_pos = collections.Counter()
    reversed_acc = collections.Counter()
    for p in ps:
        for key, value in freq_pos.items():
            if key / args.n >= p:
                reversed_acc_pos[p] += value
        for key, value in freq.items():
            if key / args.n >= p:
                reversed_acc[p] += value

    p_ratio = {p: reversed_acc_pos[p] / reversed_acc[p] for p in ps}

    print(f"Reversed Acc Pos Ratio: {p_ratio}")

    print(f"SC: {sc}/{len(data)} = {sc / len(data)}")
    print(f"Num Pred: {num_pred}/{len(data)} = {num_pred / len(data)}")

    json.dump(outputs, open(args.output_file, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
