import json
import argparse
from glob import glob
import os
from tqdm import tqdm


def remove_suffix(solution: str):
    return solution.split("The answer is")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = [json.loads(line) for line in open(args.input_file).readlines()]
    else:
        data = []
        for file in glob(args.input_file):
            data.extend([json.loads(line) for line in open(file).readlines()])

    q_set = set()
    outputs = []
    for i, item in tqdm(enumerate(data)):
        q_set.add(item["question"] + item["completion"])
        item["solution"] = item["completion"]
        item["solution_wo_suffix"] = remove_suffix(item["completion"])
        item["id"] = i
        outputs.append(item)

    print(len(q_set))
    print(len(outputs))
    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
