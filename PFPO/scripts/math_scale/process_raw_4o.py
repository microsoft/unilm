import json
import argparse
from glob import glob
import os
from tqdm import tqdm


def extract_qa(completion: str, remove_suffix: bool = False):
    q_s = completion.find("Created Question:")
    if q_s == -1:
        return "", ""

    s_s = completion.find("Solution to the Created Question:")
    if s_s == -1:
        return "", ""

    question = completion[q_s + len("Created Question:"):s_s].strip()
    solution = completion[s_s + len("Solution to the Created Question:"):].strip()

    if remove_suffix:
        solution = solution.split("The answer is")[0].strip()

    return question, solution


def remove_suffix(solution: str):
    return solution.split("The answer is")[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        data = [json.loads(line) for line in open(args.input_file).readlines()]
    else:
        data = []
        for file in glob(args.input_file):
            data.extend([json.loads(line) for line in open(file).readlines()])

    q_set = set()
    cnt = 0
    outputs = []
    for item in tqdm(data):
        question, solution = extract_qa(item["completion"])
        if question == "" or solution == "":
            continue
        if question + solution in q_set:
            cnt += 1
            continue
        q_set.add(question + solution)
        item["question"] = question
        item["solution"] = solution
        item["solution_wo_suffix"] = remove_suffix(solution)
        outputs.append(item)

    print(len(outputs))
    json.dump(outputs, open(args.output_file, "w"), indent=2)


if __name__ == '__main__':
    main()
