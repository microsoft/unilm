import argparse
import json
import os
from glob import glob

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in glob(args.input_file):
            print(file)
            data.extend(json.load(open(file)))

    print(len(data))

    outputs = []
    for item in tqdm(data):
        item.pop("text")
        if "sc_res" in item:
            item.pop("sc_res")
        if "sc_pred" in item:
            item.pop("sc_pred")
        # item.pop("id")
        # assert "uuid" in item
        if "uuid" in item:
            item.pop("id")
        if "\\boxed{" not in item["response"]:
            continue
        if not item["pred"]:
            continue
        item["label"] = item.pop("pred")
        response = item.pop("response")
        item["box_solution"] = item["solution_wo_suffix"] + response
        outputs.append(item)

    print(len(outputs))
    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()

"""
>>> python ~/gpt-chat-examples/scripts/math_scale/process_raw_4o_labeling.py \
    --input_file "4o.500k.de_con.v1.0.0shot.n1.tem0.0.p1.0.[0-9]*-of-100.json" --output_file ../train.500k.de_con.v1.0.boxed.json

511309
491733
"""

