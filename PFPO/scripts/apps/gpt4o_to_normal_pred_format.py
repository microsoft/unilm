import sys
import json
import os
import argparse

sys.set_int_max_str_digits(0)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from post_processors.code.clean import standard_cleaner_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    outputs = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = standard_cleaner_default(item["completion"])
            item["pred"] = pred
            # item["test_cases"] = item.pop("input_output")
            item["response"] = item.pop("completion")
            item["id"] = item["problem_id"]
            outputs.append(item)

    json.dump(outputs, open(args.output_file, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
