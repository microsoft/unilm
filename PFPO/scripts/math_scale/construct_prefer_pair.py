import argparse
import json
import sys
import os
from glob import glob


# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
#
# from data.math_util import is_equiv


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
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in glob(args.input_file, recursive=True):
            print(file)
            # data += json.load(open(file))
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

    data = merge_seed_sampled_data(data)

    outputs = []
    cnt = 0
    pass_at_k = 0
    num_pairs = 0
    pos_missing = 0
    neg_missing = 0
    full_positive_samples = []
    full_negative_samples = []
    for item in data:
        if "res" not in item:
            raise RuntimeError("Use `fix_answer_extract_and_verify.py` to add `res` field to the input file")
            # if isinstance(item["pred"], list):
            #     preds = item["pred"]
            # else:
            #     preds = [item["pred"]]
            #
            # res = [is_equiv(p, str(item["label"])) for p in preds]
            # if isinstance(item["pred"], str):
            #     res = res[0]
            # item["res"] = res

        if not item["res"]:
            continue

        if item["res"][0]:
            cnt += 1
        if any(item["res"]):
            pass_at_k += 1

        pos = []
        neg = []
        for resp, r in zip(item["response"], item["res"]):
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
    print(f"Acc: {cnt / len(data)}")
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
>>> python scripts/math_scale/construct_prefer_pair.py \
    --input_file "../msranlpintern/share/models/mathscale-mistral/mathscale/train.v60.300k.1-of-30.v1.0.0shot.n5.tem1.0.p0.9.0-of-8.json" \
    --output_file ../msranlpintern/share/models/mathscale-mistral/mathscale/train.v60.300k.1-of-30.v1.0.0shot.n5.tem1.0.p0.9.json

Total number of items: 1250
Acc: 0.8544
Pass at k: 0.9696
Missing: 825 / 1250
Num pairs: 2022

>>> python scripts/math_scale/construct_prefer_pair.py \
    --input_file "../msranlpintern/share/models/mathscale-mistral/mathscale/train.v60.300k.1-of-30.v1.0.0shot.n5.tem1.0.p0.9.fix_predict.json" \
    --output_file ../msranlpintern/share/models/mathscale-mistral/mathscale/train.v60.300k.1-of-30.v1.0.0shot.n5.tem1.0.p0.9.fix_predict.dpo.json

Total number of items: 10000
Acc: 0.7141
Pass at k: 0.8515
Missing: 6383 / 10000
Num pairs: 17630

>>> python scripts/math_scale/construct_prefer_pair.py --input_file "../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.*-of-16.json" --output_file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.1-of-30.v1.2.0shot.n10.tem1.0.p0.9.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.0-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.1-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.10-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.11-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.12-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.13-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.14-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.15-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.2-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.3-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.4-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.5-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.6-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.7-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.8-of-16.json
Error in file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/train.v60.300k.2-of-30.v1.2.0shot.n10.tem1.0.p0.9.9-of-16.json
Total number of items: 10000
Acc: 0.6983
Pass at k: 0.8525
Missing: 5670 / 10000
Num pairs: 66254

>>> python scripts/math_scale/construct_prefer_pair.py \
    --input_file "../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/all_splits/train.v60.300k.all.v1.2.0shot.n10.tem1.0.p0.9.*-of-100.json" \
    --output_file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale/all_splits/train.v60.300k.all.v1.2.0shot.n10.tem1.0.p0.9.dpo_v1.0.json

Total number of items: 300000
Acc: 0.4003566666666667
Pass at k: 0.5258733333333333
No positive solutions: 142238 / 300000
No negative solutions: 71338 / 300000
Num pairs: 1361840

>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair.py \
    --input_file "./500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.?-of-8.json" \
    --output_file train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-0-of-20/train.500k.de_con.boxed.v1.0.0-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-1-of-20/train.500k.de_con.boxed.v1.0.1-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-10-of-20/train.500k.de_con.boxed.v1.0.10-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-11-of-20/train.500k.de_con.boxed.v1.0.11-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-12-of-20/train.500k.de_con.boxed.v1.0.12-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-13-of-20/train.500k.de_con.boxed.v1.0.13-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-14-of-20/train.500k.de_con.boxed.v1.0.14-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-15-of-20/train.500k.de_con.boxed.v1.0.15-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-16-of-20/train.500k.de_con.boxed.v1.0.16-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-17-of-20/train.500k.de_con.boxed.v1.0.17-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-18-of-20/train.500k.de_con.boxed.v1.0.18-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-19-of-20/train.500k.de_con.boxed.v1.0.19-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-2-of-20/train.500k.de_con.boxed.v1.0.2-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-3-of-20/train.500k.de_con.boxed.v1.0.3-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-4-of-20/train.500k.de_con.boxed.v1.0.4-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-5-of-20/train.500k.de_con.boxed.v1.0.5-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-6-of-20/train.500k.de_con.boxed.v1.0.6-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-7-of-20/train.500k.de_con.boxed.v1.0.7-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-8-of-20/train.500k.de_con.boxed.v1.0.8-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.0-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.1-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.2-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.3-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.4-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.5-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.6-of-8.json
./500k-split-9-of-20/train.500k.de_con.boxed.v1.0.9-of-20.0shot.n10.tem1.0.p0.9.7-of-8.json
Total number of items: 491733
Acc: 0.6720944089577067
Pass at k: 0.8531032084484873
No positive solutions: 72234 / 491733
No negative solutions: 187738 / 491733
Num pairs: 3873158


>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair.py \
    --input_file "./500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-32.json" \
    --output_file train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.json

Total number of items: 491733
Acc: 0.681286389158344
Pass at k: 0.8782998090427122
No positive solutions: 59844 / 491733
No negative solutions: 211316 / 491733
Num pairs: 3693524


########################################## ITERATION 1 ###########################################################

>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair.py \
    --input_file "../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.json

Total number of items: 491733
Acc: 0.7095415601556129
Pass at k: 0.8631289744637842
No positive solutions: 67304 / 491733
No negative solutions: 250765 / 491733
Num pairs: 2844227

>>> python ~/gpt-chat-examples/scripts/math_scale/construct_prefer_pair.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.prefer_pair.json
    Total number of items: 491733
Acc: 0.6936853943095135
Pass at k: 0.8353761085792493
No positive solutions: 80951 / 491733
No negative solutions: 255617 / 491733
Num pairs: 2550984


"""
