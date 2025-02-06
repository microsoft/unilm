import json
import argparse
from glob import glob
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response_file_format", type=str)
    parser.add_argument("--seed_list", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    seed_list = eval(args.seed_list)
    rewards = []
    for seed in seed_list:
        # _file = args.response_file_format.format(seed)
        _file = args.response_file_format.replace("[[seed]]", str(seed))
        if os.path.exists(_file):
            print(f"Loading: {_file}")
            data = json.load(open(_file))
            for item in data:
                item["index"] = f"{item['index']}_{seed}"
            rewards.extend(data)
        else:
            print(f"File not found: {_file}")

    json.dump(rewards, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
