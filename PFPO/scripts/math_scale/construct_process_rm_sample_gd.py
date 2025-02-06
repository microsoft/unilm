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


def _process_response_worker(item):
    item_id, resp_id, prefix_id = item["prefix_id"].split("_")
    prefix = item["prefix"]
    if "mscale-v0.1" in item_id:
        pass
    elif "numina" not in item_id:
        item_id = int(item_id)
    if not item["label"]:
        return item_id, {}

    if item["pred"] == "":
        return item_id, {}

    if item["pred"] == "failed extracting answer from completion":
        return item_id, {}

    v = len([1 for res in item["res"] if res])
    return item_id, {"v": v, "prefix": prefix}


def _process_trajectories_worker(item, binary: bool):
    item_id, trajectories = item
    outputs = {
        "idx": item_id
    }

    trajectory_pairs = [(traj["v"], traj["prefix"]) for traj in trajectories]
    prefix_vis = set()

    values = []
    prefixes = []
    for v, prefix in trajectory_pairs:
        if prefix in prefix_vis:
            continue
        values.append(parse_value(v, binary))
        prefixes.append(prefix)
        prefix_vis.add(prefix)
    outputs[f"value"] = values
    outputs[f"prefix"] = prefixes
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
    parser.add_argument("--binary", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    print("Collecting data...")
    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in sorted(glob(args.input_file)):
            if ".metrics" in file:
                continue
            print(file)
            try:
                sub_data = json.load(open(file))
            except:
                print(f"Warning: {file}l")
                sub_data = [json.loads(line) for line in open(f"{file}l").readlines()]
            data += sub_data

    data = merge_seed_sampled_data(data)

    item_id2partial_trajectories = collections.defaultdict(list)
    missing = 0
    with Pool(args.num_workers) as pool:
        inputs = []
        for item in data:
            inputs.append(item)
        for item_id, result in tqdm(pool.imap_unordered(_process_response_worker, inputs), total=len(inputs)):
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

    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()

"""
>>> python scripts/math_scale/construct_process_rm_sample_gd.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm_gd.binary.local.json \
    --binary --num_workers 24

Counter({0: 1243140, 3: 916473, 2: 442815, 1: 435687})
Missing 74 items in the response data.
Counter({1: 1793986, 0: 1242775})

>>> python scripts/math_scale/construct_process_rm_sample_gd.py \
    --input_file "../msranlpintern/share/models/llama3.1_8b_mathscale4o/model_lr1e-5_batch512_epochs3_gpus8_linearSchedule/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/share/models/llama3.1_8b_mathscale4o/model_lr1e-5_batch512_epochs3_gpus8_linearSchedule/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm_gd.binary.local.json \
    --binary --num_workers 24

Counter({0: 1103040, 3: 908168, 2: 435088, 1: 427143})
Missing 0 items in the response data.
Counter({1: 1769112, 0: 1102585})

>>> python scripts/math_scale/construct_process_rm_sample_gd.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm_gd.local.json \
    --num_workers 24

Counter({0: 1243140, 3: 916473, 2: 442815, 1: 435687})
Missing 74 items in the response data.
Counter({0: 1242779, 3: 915927, 2: 442573, 1: 435482})

>>> python scripts/math_scale/construct_process_rm_sample_gd.py \
    --input_file "../msranlpintern/share/models/llama3.1_8b_mathscale4o/model_lr1e-5_batch512_epochs3_gpus8_linearSchedule/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/share/models/llama3.1_8b_mathscale4o/model_lr1e-5_batch512_epochs3_gpus8_linearSchedule/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm_gd.local.json \
    --num_workers 24

>>> python scripts/math_scale/construct_process_rm_sample_gd.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm_gd.local.json \
    --num_workers 24

Counter({0: 1013210, 3: 660151, 2: 306372, 1: 305458})
Missing 0 items in the response data.
Counter({0: 1012382, 3: 659126, 2: 305949, 1: 305153})

>>> python scripts/math_scale/construct_process_rm_sample_gd.py \
    --input_file "../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/split-512/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.*-of-512.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.prefix_completion.n3.tem1.0.p0.9.process_rm_gd.local.json \
    --num_workers 24
    
Counter({0: 950765, 3: 748413, 2: 353782, 1: 336295})
Missing 0 items in the response data.
Counter({0: 950541, 3: 747952, 2: 353608, 1: 336155})

"""
