import json
import random
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--id_field", type=str, default="id")
    args = parser.parse_args()

    random.seed(42)

    data = json.load(open(args.input_file))

    id2data = {}
    for item in data:
        idx = item[args.id_field]
        if idx not in id2data:
            id2data[idx] = []
        id2data[idx].append(item)

    print(f"Number of unique ids: {len(id2data)}")
    train_num = int(input("Number of ids for training: "))

    train_ids = random.sample(list(id2data.keys()), train_num)
    train_data = []
    test_data = []
    for idx, items in id2data.items():
        if idx in train_ids:
            train_data.extend(items)
        else:
            test_data.extend(items)

    json.dump(train_data, open(args.input_file.replace(".json", f".train_{len(train_data)}.json"), "w"), indent=2)
    json.dump(test_data, open(args.input_file.replace(".json", f".test_{len(test_data)}.json"), "w"), indent=2)

