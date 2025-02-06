import argparse
import collections
import os.path

import matplotlib.pyplot as plt
from glob import glob
import json


def majority_voting_predict(preds):
    if isinstance(preds, str):
        return preds

    preds = [pred for pred in preds if pred]
    if len(preds) == 0:
        return ""

    assert isinstance(preds, list)
    if isinstance(preds[0], list):
        tmp = []
        for pred in preds:
            tmp.append(str(sorted(pred)))
        pred, freq = collections.Counter(tmp).most_common(1)[0]
        pred = eval(pred)
    elif isinstance(preds[0], str):
        pred, freq = collections.Counter(preds).most_common(1)[0]
    else:
        print(f"Unknown type {type(preds[0])}")
        pred = ""
        freq = 0
    return pred, freq


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
    parser.add_argument("--input_file")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        responses = json.load(open(args.input_file))
    else:
        responses = []
        for file in glob(args.input_file):
            responses += json.load(open(file))
    freq = {'correct': 0, 'incorrect': 0}
    num = {'correct': 0, 'incorrect': 0}
    correct_freqs = []
    incorrect_freqs = []
    for item in responses:
        sc_pred, f = majority_voting_predict(item["pred"])
        if item["sc_res"]:
            num['correct'] += 1
            freq['correct'] += f
            correct_freqs.append(f)
        else:
            num['incorrect'] += 1
            freq['incorrect'] += f
            incorrect_freqs.append(f)

    print("Correct: ", num['correct'])
    print("Incorrect: ", num['incorrect'])
    print("Correct freq: ", freq['correct'])
    print("Incorrect freq: ", freq['incorrect'])
    print(f"Correct sc avg freq: {freq['correct'] / num['correct']}")
    print(f"Incorrect sc avg freq: {freq['incorrect'] / num['incorrect']}")

    plot_histogram(correct_freqs, bins=50, title="Correct SC Prediction Frequency", output_file="correct_sc_freq.png")
    plot_histogram(incorrect_freqs, bins=50, title="Incorrect SC Prediction Frequency", output_file="incorrect_sc_freq.png")


if __name__ == '__main__':
    main()