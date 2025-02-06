import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_bar_chart(data, file_path):
    indices = list(range(len(data)))

    # Create the bar chart
    plt.figure(figsize=(4, 4))
    bars = plt.bar(indices, data)

    # Customize the chart
    plt.title('Bar Chart of Array Values')
    plt.xlabel('Frequency', fontsize=16)
    plt.ylabel('Amount Difference between Correct and Incorrect Prediction', fontsize=12)

    # Add value labels on top of each bar
    # for i, bar in enumerate(bars):
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2., height,
    #              f'{data[i]}',
    #              ha='center', va='bottom',
    #              fontsize=12)

    # Color positive and negative bars differently
    for i, value in enumerate(data):
        if value >= 0:
            bars[i].set_color('blue')
        else:
            bars[i].set_color('red')

    plt.tick_params(axis='x', labelsize=15)  # Change the font size of the x-axis scale
    plt.tick_params(axis='y', labelsize=15)  # Change the font size of the y-axis scale

    plt.savefig(file_path, bbox_inches='tight', dpi=800, format='png')


def draw_line_plot(data, file_path):
    # Compute histogram data
    x = np.arange(0.1, 1.1, 0.1)

    # Create the line plot
    plt.figure(figsize=(4, 4))
    plt.plot(x, data, color='#658873', marker='o', linestyle='-')

    # Change the size of the font in the legent
    # plt.legend(prop={"size": 15})
    plt.tick_params(axis='x', labelsize=15)  # Change the font size of the x-axis scale
    plt.tick_params(axis='y', labelsize=15)  # Change the font size of the y-axis scale
    plt.grid(True)

    # Add labels and title
    # plt.xlabel('Top-1 Averaged Frequency', fontsize=16)
    plt.ylabel('Ratio of Correct Predictions', fontsize=16)
    # plt.legend(loc='upper left')

    plt.savefig(file_path, bbox_inches='tight', dpi=800, format='png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--n", type=int, default=128)
    args = parser.parse_args()

    data = json.load(open(args.input_file))

    freq_pos = collections.Counter()
    freq_neg = collections.Counter()
    freq = collections.Counter()
    for item in data:
        if item["sc_res"]:
            freq_pos[item["sc_freq"]] += 1
        else:
            freq_neg[item["sc_freq"]] += 1
        freq[item["sc_freq"]] += 1

    diffs = []
    for i in range(args.n + 1):
        tmp = 0
        if i in freq_pos:
            tmp += freq_pos[i]
        if i in freq_neg:
            tmp -= freq_neg[i]
        diffs.append(tmp)

    plot_bar_chart(diffs, args.output_file)

    ps =  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reversed_acc_pos = collections.Counter()
    reversed_acc = collections.Counter()
    for p in ps:
        for key, value in freq_pos.items():
            if key / args.n >= p:
                reversed_acc_pos[p] += value
        for key, value in freq.items():
            if key / args.n >= p:
                reversed_acc[p] += value

    p_ratio = {p: reversed_acc_pos[p] / reversed_acc[p] for p in ps}

    draw_line_plot(list(p_ratio.values()), args.output_file.replace(".png", "_line.png"))


if __name__ == "__main__":
    main()
