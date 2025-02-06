import sys
import json
import os
import argparse

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.human_eval import HumanEvalReader
from post_processors.code.evaluator import HumanEvaluator
from post_processors.code.clean import standard_cleaner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    args = parser.parse_args()

    reader = HumanEvalReader()
    data = reader()

    id2output = {}
    with open(args.input_file, "r", encoding="utf-8")as f:
        for line in f:
            item = json.loads(line)
            pred = standard_cleaner(item["completion"])
            id2output[item["task_id"]] = {
                "pred": pred,
                "response": item["completion"],
            }

    for item in data:
        item["pred"] = id2output[item["task_id"]]["pred"]
        item["test_cases"] = item["test"]
        item["id"] = item["task_id"]

    evaluator = HumanEvaluator()
    predictions, metrics = evaluator(data, num_workers=24)

    print(metrics)


if __name__ == "__main__":
    main()
