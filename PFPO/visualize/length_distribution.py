import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import torch
import argparse
import random
from transformers import AutoTokenizer
from tqdm import tqdm
import sys

sys.set_int_max_str_digits(0)


def plot_histogram(data, bins=10, x_label="Value", y_label="Frequency", title="Histogram", output_file="histogram.png"):
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    # plt.show()
    plt.savefig(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--output_file", type=str, default="histogram.png")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    data = json.load(open(args.input_file))
    if args.sample > 0:
        data = random.sample(data, args.sample)
    pos_data = []
    neg_data = []
    for item in data:
        # if not item["pos_code"] or not item["neg_code"]:
        #     continue
        # pos_data.append(item["pos_code"][0])
        # neg_data.append(item["neg_code"][0])
        if not item["pos"] or not item["neg"]:
            continue
        pos_data.append(item["pos"][0])
        neg_data.append(item["neg"][0])

    res = tokenizer(pos_data + neg_data, padding=False)
    half = len(pos_data)

    pos_lengths = [len(res["input_ids"][i]) for i in range(half)]
    neg_lengths = [len(res["input_ids"][i]) for i in range(half, len(res["input_ids"]))]

    diffs = [pos_lengths[i] - neg_lengths[i] for i in range(half)]

    # plot_histogram(pos_lengths, bins=20, x_label="Length", y_label="Frequency", title="Positive Length Distribution", output_file="pos_histogram.png")
    # plot_histogram(neg_lengths, bins=20, x_label="Length", y_label="Frequency", title="Negative Length Distribution", output_file="neg_histogram.png")
    plot_histogram(diffs, bins=20, x_label="Difference", y_label="Frequency", title="Difference Length Distribution", output_file="diff_histogram.png")


if __name__ == "__main__":
    main()
