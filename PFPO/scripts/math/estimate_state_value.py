import json
import argparse
import os.path
from glob import glob
import collections
from tqdm import tqdm

"""
We sampled some intermediate steps from single response, and each intermediate state will be calculated with a value by counting the reached outcomes.
We can rank the intermediate steps according to their distance to the origin, i.e., the prompt.
The problem is how to adjust the value based on the values of its preceding states:


Remember that if the outcome label is accurate, we can directly use the expected value as the reward since it can well indicate its importance.
However, under self-consistency setting, we should always assume that, if the outcome is incorrect, then we should find the most distant state from the origin,
and remains the largest probability that it is still possible to reach the correct answer.

TO maintain the prefixes with the least confidence over the pseudo label.


TODO: Deepseek's extraction utils always return a list for MATH questions. Think about how to process this (for self-consistency).
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--top_k", type=int)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = json.load(open(args.input_file))
    else:
        data = []
        for file in glob(args.input_file):
            print(file)
            data += json.load(open(file))

    item_id2prefixes = collections.defaultdict(list)
    for i, prefix in tqdm(enumerate(data)):
        tmp = prefix["prefix_id"].split("_")
        # resp_id = int(resp_id)  # -2
        # prefix_id = int(prefix_id)  # -1
        item_id = "_".join(tmp[:-2])

        last_iter_pseudo_label = prefix["sc_label_0"]

        cnt = collections.Counter()
        for pred in prefix["pred"]:
            if isinstance(pred, list):
                cnt.update(pred)
            else:
                cnt[pred] += 1

        if last_iter_pseudo_label not in cnt:
            v = 0
        else:
            v = cnt[last_iter_pseudo_label]

        item_id2prefixes[item_id].append((i, v))

    print(len(item_id2prefixes))

    outputs = []
    for item_id, prefixes in item_id2prefixes.items():
        prefixes.sort(key=lambda x: x[1])  # Ascending order
        for i, v in prefixes[:args.top_k]:
            outputs.append(data[i])

    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
