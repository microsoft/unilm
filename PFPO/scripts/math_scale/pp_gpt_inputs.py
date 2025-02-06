import json
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    data = [json.loads(line) for line in open(args.input_file).readlines()]

    for item in data:
        item["prompt"] = f"{item['question']}\n\nPlease put your final answer within " + "\\boxed{}."

    with open(args.output_file, "w") as f:
        for item in tqdm(data):
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
