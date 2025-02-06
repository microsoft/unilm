import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import json
import torch
import argparse


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
    parser.add_argument("--output_file", type=str, default="histogram.png")
    args = parser.parse_args()

    rewards = json.load(open(args.input_file))
    if isinstance(rewards[0]["reward"], list):
        rewards = [item["reward"][0] for item in rewards]
    else:
        rewards = [item["reward"] for item in rewards]
    plot_histogram(rewards, bins=20, x_label="Reward", y_label="Frequency", title="Reward Histogram")


if __name__ == "__main__":
    main()
