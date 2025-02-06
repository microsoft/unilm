import json
import argparse
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from data.mathscale.util import mathscale_extract_answer_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--split", type=int, default=0)
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.input_file).readlines()]

    outputs = []
    for i, item in tqdm(enumerate(data)):
        item["id"] = f"numina-{i}"
        if "\\boxed{" not in item["completion"]:
            continue
        label = mathscale_extract_answer_v2(item["completion"])
        if label != "":
            item["label"] = label
            outputs.append(item)

    print(len(outputs))
    if args.split <= 0:
        json.dump(outputs, open(args.output_file, "w"), indent=2)
    else:
        split_size = (len(outputs) + args.split - 1) // args.split
        for i in range(args.split):
            json.dump(outputs[i * split_size:(i + 1) * split_size],
                      open(args.output_file.replace(".json", f".{i}-of-{args.split}.json"), "w"), indent=2)


if __name__ == '__main__':
    main()
