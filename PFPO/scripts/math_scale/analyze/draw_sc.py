import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Dict, List


def draw_line_plot(data, file_path, baseline, baseline_label):
    all_colors = ["#AFEEEE", "#4682B4", "#6A5ACD", "#BA55D3", "#3CB371"]

    # Compute histogram data
    x_real = np.array([4, 8, 16, 32, 64, 128])
    x = np.arange(6)

    # Create the line plot
    plt.figure(figsize=(4, 4))
    # plt.plot(x, data, color='#658873', marker='o', linestyle='-')
    for i, (label, values) in enumerate(data.items()):
        plt.plot(x, values, color=all_colors[i], marker='.', linestyle='-', label=label)

    baseline = np.ones_like(list(data.values())[0]) * baseline
    plt.plot(x, baseline, linestyle='--', color=all_colors[-1], label=baseline_label)

    # Change the size of the font in the legent
    plt.legend(prop={"size": 15})
    plt.xticks(x, x_real)
    plt.tick_params(axis='x', labelsize=15)  # Change the font size of the x-axis scale
    plt.tick_params(axis='y', labelsize=15)  # Change the font size of the y-axis scale
    plt.grid(True)

    # Add labels and title
    plt.xlabel('Sample@K', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='center right')

    plt.savefig(file_path, bbox_inches='tight', dpi=800, format='png')


def main():
    # data = {
    #     "self-consistency": [65.96, 69.30, 71.14, 72.30, 72.28, 72.52],
    #     "BoN weighted": [66.98, 69.50, 71.32, 72.42, 72.54, 72.66],
    # }
    # greedy_decoding = 65.50
    #
    # draw_line_plot(data, "mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42.png", greedy_decoding, "greedy decoding")

    # data = {
    #     "self-consistency": [67.86, 70.26, 71.54, 72.38, 72.58, 72.88],
    #     "BoN weighted": [68.84, 70.84, 71.66, 72.24, 72.70, 72.96],
    # }
    # greedy_decoding = 66.72
    #
    # draw_line_plot(data, "mathstral.mscale-v0.1-300k.process-dpo-sc.p0.5.iter1.V100.tp4dp32.v3.1.s42.png", greedy_decoding, "greedy decoding")

    # data = {
    #     "self-consistency": [68.76, 70.62, 71.72, 72.40, 72.50, 72.58],
    #     "BoN weighted": [69.22, 70.90, 72.00, 72.36, 72.56, 72.66],
    # }
    # greedy_decoding = 67.50

    data = {
        "self-consistency": [60.34,	65.92,	68.40,	69.66,	70.52,	71.02],
        "BoN weighted": [62.32,	66.70,	68.48,	69.68,	70.36,	71.20],
    }
    greedy_decoding = 61.42

    draw_line_plot(data, "mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42.png", greedy_decoding, "greedy decoding")


if __name__ == "__main__":
    main()
