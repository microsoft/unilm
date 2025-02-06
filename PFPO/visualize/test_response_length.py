import json
from transformers import AutoTokenizer, PreTrainedTokenizer
import argparse
from glob import glob
from tqdm import tqdm
import os
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt

_tokenizer: PreTrainedTokenizer


def _init_(tokenizer):
    global _tokenizer
    _tokenizer = tokenizer


def plot_histogram(data, bins=10, x_label="Value", y_label="Frequency", title="Histogram", output_file="histogram.png"):
    # clear previous data
    plt.clf()
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    # plt.show()
    plt.savefig(output_file)


def merge_key(item, value):
    assert isinstance(item, list)
    if isinstance(value, list):
        item = item + value
    else:
        item.append(value)
    return item


def merge_seed_sampled_data(data, key_field="response"):
    id2data = {}
    for item in data:
        if item["id"] not in id2data:
            id2data[item["id"]] = item
            continue

        tmp = id2data[item["id"]]
        if isinstance(tmp[key_field], str):
            tmp[key_field] = [tmp[key_field]]

        tmp[key_field] = merge_key(tmp[key_field], item[key_field])
        id2data[item["id"]] = tmp

    return list(id2data.values())


def worker(item):
    text = item["text"]

    tokens = _tokenizer.tokenize(text)

    item["length"] = len(tokens)
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--tokenizer", "-t", type=str)
    parser.add_argument("--key_field", type=str, default="response")
    parser.add_argument("--topic_field", type=str, default=None)
    parser.add_argument("--ks", type=str, default="1,4,8,16")
    parser.add_argument("--num_workers", type=int, default=16)
    # parser.add_argument("--output_file", type=str, default="response_length.png")
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        print("Reading from file")
        print(args.input_file)
        with open(args.input_file, "r") as f:
            data = json.load(f)
    else:
        data = []
        for file in glob(args.input_file):
            print(file)
            with open(file, "r") as f:
                data.extend(json.load(f))

    data = merge_seed_sampled_data(data, key_field=args.key_field)
    ks = sorted([int(k) for k in args.ks.split(",")])
    ks = [0] + ks
    mp_inputs = []
    for item in data:
        if isinstance(item[args.key_field], str):
            item[args.key_field] = [item[args.key_field]]

        _inputs = [{"text": x} for x in item[args.key_field]]
        if args.topic_field:
            for x in _inputs:
                x["topic"] = item[args.topic_field]

        for i, k in enumerate(ks):
            if i == 0:
                continue
            for x in _inputs[ks[i - 1]:k]:
                x["id"] = item["id"]
                x["k"] = k
                mp_inputs.append(x)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    with Pool(args.num_workers, initializer=_init_, initargs=(tokenizer,)) as p:
        results = list(tqdm(p.imap(worker, mp_inputs), total=len(mp_inputs)))

    k2data = {k: [] for k in ks}
    for item in results:
        k2data[item["k"]].append(item["length"])

    acc = 0
    acc_n = 0
    for k, data in k2data.items():
        acc += sum(data)
        acc_n += len(data)
        if acc_n:
            print(f"k={k}, len={acc_n}, average={acc / acc_n}")
        else:
            print(f"k={k}, len={acc_n}, average=0")

    if args.topic_field:
        topic2data = {}
        for item in results:
            topic = item["topic"]
            if topic not in topic2data:
                topic2data[topic] = []
            topic2data[topic].append(item["length"])

        for topic, data in topic2data.items():
            if len(data):
                print(f"topic={topic}, len={len(data)}, average={sum(data) / len(data)}")
            else:
                print(f"topic={topic}, len={len(data)}, average=0")

            plot_histogram(data, bins=10, x_label="Length", y_label="Frequency", title=f"{topic} Histogram", output_file=f"{topic}_histogram.png")


if __name__ == '__main__':
    main()
