import json
import argparse
from glob import glob
import os
import sys
import collections
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.mathscale.util import mathscale_is_equiv


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


def extract_sc_mul_labels(response_data, id_field: str):
    id2sc_preds = {}
    for item in response_data:
        if not item["response"]:
            continue

        if isinstance(item["response"], list):
            preds = item["pred"]
        else:
            preds = [item["pred"]]

        assert len(preds) > 1, f"Self-consistency requires at least 2 predictions, but got {len(preds)}."

        sorted_preds = majority_voting_frequency(preds)
        item_id = item[id_field] if isinstance(item[id_field], str) else str(item[id_field])
        id2sc_preds[item_id] = sorted_preds

    return id2sc_preds


def counting_partial_response_value(preds, label):
    assert isinstance(preds, list), preds
    v = 0
    for pred in preds:
        res = mathscale_is_equiv(pred, label)[0]
        if res:
            v += 1
    return v


def parse_value(v, binary: bool):
    if binary:
        return 1 if v > 0 else 0
    return v


def _process_response_init(id2sc_preds):
    global _id2sc_preds
    _id2sc_preds = id2sc_preds


def _process_response_worker(item, top_p: float = 0.0):
    item_id, resp_id, prefix_id = item["prefix_id"].split("_")
    prefix = item["prefix"]
    # item_id = int(item_id)
    if item_id not in _id2sc_preds:
        return item_id, {}

    if item["pred"] == "":
        return item_id, {}

    if item["pred"] == "failed extracting answer from completion":
        return item_id, {}

    sc_label, sc_freq = _id2sc_preds[item_id][0]
    tot_num = sum([fre for _, fre in _id2sc_preds[item_id]])
    if sc_freq / tot_num < top_p:
        return item_id, {}

    v = counting_partial_response_value(item["pred"], sc_label)
    return item_id, {"v": v, "prefix": prefix}


def _process_trajectories_worker(item, binary: bool):
    item_id, trajectories = item
    outputs = {
        "idx": item_id
    }

    trajectory_pairs = [(traj["v"], traj["prefix"]) for traj in trajectories]
    prefix_set = set()

    values = []
    prefixes = []
    for v, prefix in trajectory_pairs:
        if prefix in prefix_set:
            continue
        values.append(parse_value(v, binary))
        prefixes.append(prefix)
        prefix_set.add(prefix)
    outputs["value"] = values
    outputs["prefix"] = prefixes
    return outputs


def merge_key(item, value):
    assert isinstance(item, list)
    if isinstance(value, list):
        item = item + value
    else:
        item.append(value)
    return item


def merge_seed_sampled_data(data, id_field="id"):
    id2data = {}
    for item in data:
        if item[id_field] not in id2data:
            id2data[item[id_field]] = item
            continue

        tmp = id2data[item[id_field]]
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
        id2data[item[id_field]] = tmp

    return list(id2data.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--response_file_for_sc", type=str)
    parser.add_argument("--response_id_field", type=str, default="idx")
    parser.add_argument("--binary", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--top_p", type=float, default=0.0)
    args = parser.parse_args()

    print("Collecting data...")
    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in glob(args.input_file):
            if ".metrics" in file:
                continue
            print(file)
            data += json.load(open(file))

    print("Collecting response data...")
    if os.path.exists(args.response_file_for_sc):
        response_data = json.load(open(args.response_file_for_sc))
    else:
        response_data = []
        for file in glob(args.response_file_for_sc):
            if ".metrics" in file:
                continue
            print(file)
            response_data += json.load(open(file))
    response_data = merge_seed_sampled_data(response_data, args.response_id_field)

    last_round_id2sc_preds = extract_sc_mul_labels(response_data, args.response_id_field)
    num_candidate_cnt = collections.Counter()
    for item_id, preds in last_round_id2sc_preds.items():
        num_candidate_cnt[len(preds)] += 1
    print(num_candidate_cnt)
    del response_data

    item_id2partial_trajectories = collections.defaultdict(list)
    missing = 0
    with Pool(args.num_workers, initializer=_process_response_init, initargs=(last_round_id2sc_preds,)) as pool:
        inputs = []
        for item in data:
            inputs.append(item)
        _annotate = partial(_process_response_worker, top_p=args.top_p)
        for item_id, result in tqdm(pool.imap_unordered(_annotate, inputs), total=len(inputs)):
            if len(result) == 0:
                missing += 1
                continue
            item_id2partial_trajectories[item_id].append(result)

    shoot_cnt = collections.Counter()
    for item_id, trajectories in item_id2partial_trajectories.items():
        for traj in trajectories:
            shoot_cnt[traj["v"]] += 1
    print(shoot_cnt)

    print(f"Missing {missing} items in the response data.")

    outputs = []
    cnt = collections.Counter()
    with Pool(args.num_workers) as pool:
        inputs = [(item_id, trajectories) for item_id, trajectories in item_id2partial_trajectories.items()]
        _annotate = partial(_process_trajectories_worker, binary=args.binary)
        for result in tqdm(pool.imap_unordered(_annotate, inputs), total=len(inputs)):
            outputs.append(result)
            cnt.update(result["value"])

    print(cnt)
    parent_dir = os.path.dirname(args.output_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()

"""
>>> python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm_by_sc.azure.json \
    --response_file_for_sc "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.?-of-8.json" \
    --response_id_field id --num_workers 128

Counter({3: 889966, 0: 537841, 2: 459487, 1: 397897})
Missing 0 items in the response data.


>>> python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm.iter0-pdpo-sc.azure.json \
    --response_file_for_sc "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n90.tem1.0.p0.9.json" \
    --response_id_field id --num_workers 128


>>> python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm.iter0-pdpo-sc-p0.6.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n90.tem1.0.p0.9.json" \
    --response_id_field id --num_workers 128 --top_p 0.6
    
Counter({3: 945267, 2: 408413, 0: 308107, 1: 279042})
Missing 1097360 items in the response data.

>>> python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${MODEL_PREFIX_PATH}/mathstral-7B-v0.1/mathscale4o/split-256/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample16.filter_same.*-of-4.prefix_completion.n3.tem1.0.p0.9.*-of-256.json" \
    --output_file ${MODEL_PREFIX_PATH}/mathstral-7B-v0.1/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample16.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm.sc-10-p0.0.azure.json \
    --response_file_for_sc "${MODEL_PREFIX_PATH}/mathstral-7B-v0.1/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-64.*json" --response_id_field id --num_workers 128

Counter({3: 1869170, 0: 1528083, 2: 859527, 1: 787261})
Missing 24189 items in the response data.
Counter({3: 1862945, 0: 1521858, 2: 855962, 1: 783862})


>>> python scripts/math_scale/construct_process_rm_sample_sc.py \
    --input_file "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/completion-split-128/cot.de_con.n8.tem1.0.p1.0.s0.upper0.7.r0.3.sample32.filter_same.{split_id}-of-4.n3.tem1.0.p1.0.*-of-128.s0.json" \
    --output_file ${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/cot.de_con.n8.tem1.0.p1.0.s0.upper0.7.r0.3.sample32.filter_same.process_rm.iter0-pdpo-sc-p0.5.v0.0.{split_id}-of-4.azure.json \
    --response_file_for_sc "${OUTPUT_PREFIX_PATH}/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-*-of-10/cot.de_con.*-of-10.n8.tem1.0.p1.0.*-of-*.s*.json" --response_id_field id --num_workers 128 --top_p 0.5
Counter({3: 803168, 2: 454386, 1: 260676, 0: 159454})
Missing 2783152 items in the response data.
Counter({3: 772040, 2: 436348, 1: 250464, 0: 153477})
"""
