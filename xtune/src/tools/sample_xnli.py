import argparse
import json
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_path",
        default=None,
        type=str,
        required=True,
        help="input xnli file",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=True,
        help="output xnli file",
    )
    parser.add_argument(
        "--sample_ratio",
        default=None,
        type=float,
        required=True,
        help="sample ratio",
    )

    args = parser.parse_args()
    lines = open(args.input_path, "r").readlines()
    head = lines[0]
    lines = lines[1:]
    random.seed(0)
    random.shuffle(lines)

    n_lines = int(len(lines) * args.sample_ratio)

    fout = open(args.output_path, "w")
    fout.write(head)
    for i, line in enumerate(lines[:n_lines]):
        fout.write(line)