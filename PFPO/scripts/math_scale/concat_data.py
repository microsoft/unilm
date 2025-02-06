import json
import argparse
from glob import glob
import sys

sys.set_int_max_str_digits(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    data = []
    for file in glob(args.input_file):
        print(file)
        data.extend(json.load(open(file)))

    print(len(data))
    print(data[0].keys())
    json.dump(data, open(args.output_file, "w"), indent=2)


if __name__ == "__main__":
    main()
