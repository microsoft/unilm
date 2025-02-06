import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Process some inputs.")
    parser.add_argument("--input_file", type=str, help="input file path")
    parser.add_argument("--split", type=int, help="split size")
    args = parser.parse_args()

    data = json.load(open(args.input_file, encoding='utf-8'))

    split_size = args.split
    bsz = (len(data) + split_size - 1) // split_size
    for i in range(split_size):
        with open(args.input_file.replace(".json", f".{i}-of-{split_size}.json"), "w", encoding="utf-8") as f:
            json.dump(data[i * bsz:(i + 1) * bsz], f)

    print(f"Split data into {split_size} parts.")


if __name__ == "__main__":
    main()
