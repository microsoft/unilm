import json
import argparse

import matplotlib.pyplot as plt
import numpy as np

colors = ['#FFEBAD', '#DCD7C1', '#BFB1D0', '#A7C0DE', '#6C91C2', '#F46F43']


def draw_double_histogram(correct_data, incorrect_data, file_path):
    # Create histogram
    plt.figure(figsize=(5, 4))
    plt.hist(correct_data, bins=10, alpha=0.7, label='Correct', color='blue')
    plt.hist(incorrect_data, bins=10, alpha=0.7, label='Incorrect', color='red')

    # Add labels and title
    plt.xlabel('Top-1 Frequency Averaged by Number of Test Cases')
    plt.ylabel('No. of Data Points')
    # plt.title('Frequency Distribution of Correct and Incorrect Data')
    plt.legend(loc='upper right')

    # Show the plot
    # plt.show()
    plt.savefig(file_path)


def draw_histogram(data, labels, file_path):
    # Create bar chart for the single group of data
    # Create a bar chart for the three metrics
    plt.figure(figsize=(4, 4))
    # bars = plt.bar(labels, data, color='#5ea69c', width=0.4)
    # bars = plt.bar(labels, data, color=colors[5], width=0.4)
    bars = plt.bar(labels, data, color="#1E90FF", width=0.4)
    # Add text labels above each bar to indicate the values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, round(yval, 2), ha='center', va='bottom', fontsize=16)

    plt.tick_params(axis='x', labelsize=15)  # Change the font size of the x-axis scale
    plt.tick_params(axis='y', labelsize=15)  # Change the font size of the y-axis scale

    # Add labels and title
    # plt.xlabel('Category')
    plt.ylabel('Pass@1', fontsize=16)
    # plt.title('Accuracy Comparison Across Different Metrics')

    plt.ylim(bottom=15)

    plt.savefig(file_path, bbox_inches='tight', dpi=800, format='png')


def draw_line_plot(correct_data, incorrect_data, file_path):
    # Compute histogram data
    correct_hist, correct_bins = np.histogram(correct_data, bins=10)
    incorrect_hist, incorrect_bins = np.histogram(incorrect_data, bins=10)

    # Get the center of each bin for plotting
    correct_bin_centers = 0.5 * (correct_bins[1:] + correct_bins[:-1])
    incorrect_bin_centers = 0.5 * (incorrect_bins[1:] + incorrect_bins[:-1])

    # Create the line plot
    plt.figure(figsize=(4, 4))
    # plt.plot(correct_bin_centers, correct_hist, label='Correct', color='#658873', marker='.', linestyle='-')
    # plt.plot(incorrect_bin_centers, incorrect_hist, label='Incorrect', color='#d2bfa5', marker='.', linestyle='-')
    plt.plot(correct_bin_centers, correct_hist, label='Correct', color=colors[2], marker='.', linestyle='-')
    plt.plot(incorrect_bin_centers, incorrect_hist, label='Incorrect', color=colors[4], marker='.', linestyle='-')

    # Change the size of the font in the legent
    plt.legend(prop={"size": 15})
    plt.tick_params(axis='x', labelsize=15)  # Change the font size of the x-axis scale
    plt.tick_params(axis='y', labelsize=15)  # Change the font size of the y-axis scale

    # Add labels and title
    # plt.xlabel('Top-1 Averaged Frequency', fontsize=16)
    plt.ylabel('No. of Data Points', fontsize=16)
    plt.legend(loc='upper left')

    plt.savefig(file_path, bbox_inches='tight', dpi=800, format='png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    data = json.load(open(args.freq_file))

    pos_freqs = []
    neg_freqs = []
    difficulties = set([item["difficulty"] for item in data])
    diff_freqs = {
        diff: {"pos": [], "neg": []} for diff in difficulties
    }
    for item in data:
        if item["prog_sc_res"]:
            pos_freqs.append(item["tot_freq"])
            diff_freqs[item["difficulty"]]["pos"].append(item["tot_freq"])
        else:
            neg_freqs.append(item["tot_freq"])
            diff_freqs[item["difficulty"]]["neg"].append(item["tot_freq"])

    # paint(pos_freqs, neg_freqs, args.output_file.replace(".png", "_all.png"))
    draw_line_plot(pos_freqs, neg_freqs, args.output_file.replace(".png", "_all.png"))
    for diff, freqs in diff_freqs.items():
        # paint(freqs["pos"], freqs["neg"], args.output_file.replace(".png", f"_{diff}.png"))
        draw_line_plot(freqs["pos"], freqs["neg"], args.output_file.replace(".png", f"_{diff}.png"))

    sc = 0
    prog_sc = 0
    first_res = 0
    diff_res = {
        diff: {"sc": 0, "prog_sc": 0, "first_res": 0} for diff in difficulties
    }
    for item in data:
        if item["sc_res"]:
            sc += 1
            diff_res[item["difficulty"]]["sc"] += 1
        if item["prog_sc_res"]:
            prog_sc += 1
            diff_res[item["difficulty"]]["prog_sc"] += 1
        if item["res"]:
            first_res += 1
            diff_res[item["difficulty"]]["first_res"] += 1

    print(f"Self-consistency: {sc}/{len(data)} = {sc / len(data)}")
    print(f"Program self-consistency: {prog_sc}/{len(data)} = {prog_sc / len(data)}")
    print(f"First res: {first_res}/{len(data)} = {first_res / len(data)}")
    for diff, res in diff_res.items():
        print(f"Difficulty {diff}:")
        print(f"Self-consistency: {res['sc']}/{len(data)} = {res['sc'] / len(data)}")
        print(f"Program self-consistency: {res['prog_sc']}/{len(data)} = {res['prog_sc'] / len(data)}")
        print(f"First res: {res['first_res']}/{len(data)} = {res['first_res'] / len(data)}")

    draw_histogram([19.24, prog_sc / len(data) * 100, sc / len(data) * 100],
                   ["G.D.", "S.C. - P", "S.C. - T"], args.output_file.replace(".png", "_res.png"))


if __name__ == "__main__":
    main()
