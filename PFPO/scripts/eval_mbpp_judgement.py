import argparse
import copy
import json
import re

from datasets import load_dataset


def mbpp_prediction_eval_judge(prediction):
    if "### The program is" not in prediction:
        return None

    pattern = r"correct|incorrect"
    regrex = re.compile(pattern)

    preds = re.findall(regrex, prediction.split("### The program is")[1])
    if len(preds) == 0 or len(preds) > 1:
        return None

    if preds[0] == "correct":
        return True
    return False


def mbpp_judge(file_path, prompt2passed):
    data = [json.loads(line) for line in open(file_path, "r", encoding="utf8").readlines()]

    missing = 0
    correct = 0
    true_samples = 0
    false_samples = 0
    for item in data:
        passed = prompt2passed[item["orig_prompt"]]

        res = mbpp_prediction_eval_judge(item["completion"])
        if res is None:
            missing += 1
            continue
        if res == passed:
            correct += 1

    print(f"Correct: {correct}, Missing: {missing}, Total: {len(data)}")
    print(f"Accuracy: {correct / len(data)}")
    print(f"Missing rate: {missing / len(data)}")
    print(f"True samples: {true_samples}, False samples: {false_samples}")


def main():
    parser = argparse.ArgumentParser(description='Completion Judge')
    parser.add_argument("--judgement_file", type=str, required=True)
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()

    results = json.load(open(args.result_file))
    predictions = json.load(open(args.prediction_file))
    assert len(results) == len(predictions)
    prompt2passed = {}
    for res, pred in zip(results.values(), predictions):
        prompt2passed[pred["prompt"]] = res[0][1]["passed"]

    mbpp_judge(args.judgement_file, prompt2passed)


if __name__ == '__main__':
    main()
