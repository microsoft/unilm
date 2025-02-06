import argparse
import json
import os.path
import sys
from glob import glob
from collections import Counter
from datasets import load_dataset

sys.set_int_max_str_digits(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    if not os.path.exists(f'apps_difficulty_{args.split}.json'):
        _dataset = load_dataset("codeparrot/apps", split=args.split).to_list()
        problem_id2difficulty = {item["problem_id"]: item["difficulty"] for item in _dataset}
        all_difficulties = Counter(problem_id2difficulty.values())
        json.dump(problem_id2difficulty, open(f'apps_difficulty_{args.split}.json', "w"), ensure_ascii=False)
        json.dump(all_difficulties, open(f'apps_difficulty_{args.split}_all.json', "w"), ensure_ascii=False)
    else:
        problem_id2difficulty = json.load(open(f'apps_difficulty_{args.split}.json'))
        all_difficulties = json.load(open(f'apps_difficulty_{args.split}_all.json'))
        problem_id2difficulty = {int(k): v for k, v in problem_id2difficulty.items()}

    if os.path.exists(args.input_file):
        if args.input_file.endswith(".json"):
            data = json.load(open(args.input_file))
        else:
            data = [json.loads(line) for line in open(args.input_file).readlines()]
    else:
        data = []
        for file in glob(args.input_file):
            if file.endswith(".json"):
                # data += json.load(open(file))
                f = open(file, "r")
                data += json.load(f)
            else:
                # data += [json.loads(line) for line in open(file).readlines()]
                lines = list(open(file).readlines())
                for line in lines:
                    data.append(json.loads(line))

    success = 0
    success_at_k = 0
    successes_at_difficulty = {difficulty: 0 for difficulty in all_difficulties}
    successes_at_k_at_difficulty = {difficulty: 0 for difficulty in all_difficulties}
    for item in data:
        p_id = item["id"]
        if isinstance(item["res"], bool):
            if item["res"]:
                success += 1
                success_at_k += 1
                successes_at_difficulty[problem_id2difficulty[p_id]] += 1
                successes_at_k_at_difficulty[problem_id2difficulty[p_id]] += 1
        else:
            if len(item["res"]):
                if any(item["res"]):
                    success_at_k += 1
                    successes_at_k_at_difficulty[problem_id2difficulty[p_id]] += 1
                if item["res"][0]:
                    success += 1
                    successes_at_difficulty[problem_id2difficulty[p_id]] += 1

    metrics = {"acc": success / len(data), "pass@k": success_at_k / len(data), "correct": success, "total": len(data)}
    for difficulty in all_difficulties:
        metrics[f"acc_{difficulty}"] = successes_at_difficulty[difficulty] / all_difficulties[difficulty]
        metrics[f"pass@k_{difficulty}"] = successes_at_k_at_difficulty[difficulty] / all_difficulties[difficulty]
        metrics[f"correct_{difficulty}"] = successes_at_difficulty[difficulty]
        metrics[f"total_{difficulty}"] = all_difficulties[difficulty]
    print(json.dumps(metrics, indent=2))

    if args.output_file:
        json.dump(data, open(args.output_file, "w"), ensure_ascii=False)
        json.dump(metrics, open(args.output_file.replace(".json", ".metrics"), "w"), ensure_ascii=False)


if __name__ == '__main__':
    main()
