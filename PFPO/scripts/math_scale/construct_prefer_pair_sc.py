import argparse
import json
import sys
import os
from glob import glob
import collections
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.mathscale.util import mathscale_is_equiv


def evaluate_by_sc(item, external_sc: str = None):
    if not item["response"]:
        return []

    if isinstance(item["response"], list):
        preds = item["pred"]
    else:
        preds = [item["pred"]]

    assert len(preds) > 1, f"Self-consistency requires at least 2 predictions, but got {len(preds)}."

    if external_sc:
        sc_pred = external_sc
    else:
        sc_pred = item["sc_pred"]
    res = [mathscale_is_equiv(p, sc_pred)[0] for p in preds]

    return res


def majority_voting_frequency(preds):
    assert isinstance(preds, list)
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


def get_pred_frequency(preds, target_pred):
    assert isinstance(preds, list)
    if isinstance(preds[0], list):
        tmp = []
        for pred in preds:
            tmp.append(str(sorted(pred)))
        tmp = collections.Counter(tmp)
    elif isinstance(preds[0], str):
        tmp = collections.Counter(preds)
    else:
        raise ValueError(f"Unknown type {type(preds[0])}")
    if isinstance(target_pred, list):
        target_pred = str(sorted(target_pred))
    return tmp.get(target_pred, 0) / len(preds)


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


def load_data(file_path):
    if os.path.exists(file_path):
        data = json.load(open(file_path))
    else:
        data = []
        for file in glob(file_path, recursive=True):
            if ".metrics" in file:
                continue
            print(file)
            f = open(file, "r")
            try:
                data += json.load(f)
            except:
                print(f"Error in file {file}")
                new_file = file.replace(".json", ".jsonl")
                lines = open(new_file, "r").readlines()
                for line in lines:
                    try:
                        data.append(json.loads(line))
                    except:
                        print(f"Error in line: {line}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--external_file_for_sc", type=str, default=None)
    parser.add_argument("--top_p", type=float, default=0.0)
    args = parser.parse_args()

    data = load_data(args.input_file)
    data = merge_seed_sampled_data(data)

    filtered = 0
    if args.external_file_for_sc:
        external_data = load_data(args.external_file_for_sc)
        # id2external_sc = {item["id"]: item["sc_pred"] for item in external_data}
        id2sc = {}
        for item in tqdm(external_data):
            sc_pred = item["sc_pred"]

            if item["pred"] == "":
                continue

            if item["pred"] == "failed extracting answer from completion":
                continue

            freq = get_pred_frequency(item["pred"], sc_pred)
            if freq > args.top_p:
                id2sc[item["id"]] = sc_pred
            else:
                filtered += 1
    else:
        id2sc = {}
        for item in tqdm(data):
            sc_pred = item["sc_pred"]

            if item["pred"] == "":
                continue

            if item["pred"] == "failed extracting answer from completion":
                continue

            freq = get_pred_frequency(item["pred"], sc_pred)
            if freq > args.top_p:
                id2sc[item["id"]] = sc_pred
            else:
                filtered += 1

    print(f"Filtered {filtered} samples.")

    outputs = []
    cnt = 0
    pass_at_k = 0
    num_pairs = 0
    pos_missing = 0
    neg_missing = 0
    full_positive_samples = []
    full_negative_samples = []
    for item in data:
        if item["id"] in id2sc:
            res_by_sc = evaluate_by_sc(item, id2sc[item["id"]])
        else:
            continue

        if len(res_by_sc) == 0:
            continue

        if res_by_sc[0]:
            cnt += 1
        if any(res_by_sc):
            pass_at_k += 1

        pos = []
        neg = []
        for resp, r in zip(item["response"], res_by_sc):
            if r:
                pos.append(resp)
            else:
                neg.append(resp)

        if len(pos) == 0:
            full_negative_samples.append(item)
            pos_missing += 1
        if len(neg) == 0:
            neg_missing += 1
            full_positive_samples.append(item)
        if len(pos) == 0 or len(neg) == 0:
            continue

        item["pos"] = pos
        item["neg"] = neg
        num_pairs += len(pos) * len(neg)
        outputs.append(item)

    print(f"Total number of items: {len(data)}")
    print(f"Acc: {cnt / (len(data) - filtered)}")
    print(f"Pass at k: {pass_at_k / len(data)}")
    print(f"No positive solutions: {pos_missing} / {len(data)}")
    print(f"No negative solutions: {neg_missing} / {len(data)}")
    print(f"Num pairs: {num_pairs}")
    json.dump(outputs, open(args.output_file, "w"), indent=2)
    json.dump(full_positive_samples[:100], open(args.output_file.replace(".json", ".pos.sample.json"), "w"), indent=2)
    json.dump(full_negative_samples[:100], open(args.output_file.replace(".json", ".neg.sample.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

"""
########################################## ITERATION 1 ###########################################################

>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.json



>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.by_sc.json
# Total number of items: 491733
# Acc: 0.6936853943095135
# Pass at k: 0.8353761085792493
# No positive solutions: 80951 / 491733
# No negative solutions: 255617 / 491733
# Num pairs: 2550984
Total number of items: 491733
Acc: 0.85128718227168
Pass at k: 1.0
No positive solutions: 0 / 491733
No negative solutions: 275526 / 491733
Num pairs: 3958764

>>> python scripts/math_scale/construct_prefer_pair_sc.py \
  --input_file "${OUTPUT_PATH_PREFIX}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.*-of-8.s0.json" \
  --output_file $OUTPUT_PATH_PREFIX/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.s0.prefer_pair.by_sc.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.0-of-8.s0.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.1-of-8.s0.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.2-of-8.s0.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.3-of-8.s0.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.4-of-8.s0.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.5-of-8.s0.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.6-of-8.s0.json
../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.7-of-8.s0.json
Total number of items: 82642
Acc: 0.706493066479514
Pass at k: 1.0
No positive solutions: 0 / 82642
No negative solutions: 18718 / 82642
Num pairs: 786166


>>> python scripts/math_scale/construct_prefer_pair_sc.py \
  --input_file "${OUTPUT_PATH_PREFIX}/experiments//mathstral.mathscale4o.sc-prm.raft.iter1.H100.dp8.v1.0.s42/checkpoint-800/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.s[01].json" \
  --output_file $OUTPUT_PATH_PREFIX/experiments//mathstral.mathscale4o.sc-prm.raft.iter1.H100.dp8.v1.0.s42/checkpoint-800/mathscale4o/train.500k.de_con.boxed.v1.0.n20.tem1.0.p0.9.v0.1.prefer_pair.by_sc.json

Total number of items: 491733
Acc: 0.8624497440684273
Pass at k: 1.0
No positive solutions: 0 / 491733
No negative solutions: 264522 / 491733
Num pairs: 14647115

>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.by_sc_p0.6.json
    --top_p 0.6
    
Filtered 123792 samples.
Total number of items: 491733
Acc: 0.7133363024242831
Pass at k: 0.7482536254430758
No positive solutions: 0 / 491733
No negative solutions: 272161 / 491733
Num pairs: 1350704

>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n90.tem1.0.p0.9.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n90.tem1.0.p0.9.prefer_pair.by_sc_p0.6.json \
    --top_p 0.6
    
Filtered 0 samples.
Total number of items: 491733
Acc: 0.8171100983663899
Pass at k: 1.0
No positive solutions: 0 / 491733
No negative solutions: 199144 / 491733
Num pairs: 333282827


>>> python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${MODEL_PREFIX_PATH}/mathstral-7B-v0.1/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-64.*json" \
    --output_file ${MODEL_PREFIX_PATH}/mathstral-7B-v0.1/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.by_sc_p0.0.json 

Filtered 0 samples.
Total number of items: 491337
Acc: 0.7944180877890328
Pass at k: 1.0
No positive solutions: 0 / 491337
No negative solutions: 160871 / 491337
Num pairs: 5961085


>>> python scripts/math_scale/construct_prefer_pair_sc.py \
    --input_file "${MODEL_PREFIX_PATH}/mathstral-7B-v0.1/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-64.*json" \
    --output_file ${MODEL_PREFIX_PATH}/mathstral-7B-v0.1/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.by_sc_p0.0.json --top_p 0.5

Filtered 164514 samples.
Total number of items: 491337
Acc: 0.6043285972764111
Pass at k: 0.6651707483865453
No positive solutions: 0 / 491337
No negative solutions: 159681 / 491337
Num pairs: 2672191

"""
