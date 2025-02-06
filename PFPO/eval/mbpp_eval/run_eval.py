import argparse
import json
import os
import random

from datasets import load_dataset

from eval.mbpp_eval.utils import compute_code_eval


def get_fewshot():
    return """
You are an expert Python programmer, and here is your task: Write a function to find the shared elements from the given two lists. Your code should pass these tests:

assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))

[BEGIN]
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res) 
[DONE]

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:

assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False

[BEGIN]
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
[DONE]

You are an expert Python programmer, and here is your task: Write a function to find the n largest integers from a given list of numbers, returned in descending order. Your code should pass these tests:

assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]

[BEGIN]
import heapq as hq
def heap_queue_largest(nums,n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
[DONE]

"""


EXAMPLE_TEMPLATE = '''You are an expert Python programmer, and here is your task: {text}\nYour code should pass these tests:\n\n{tests}\n\n'''
EXAMPLE_TEMPLATE_493 = '''You are an expert Python programmer, and here is your task: {text}\n\ncalculate_polygons(startx, starty, endx, endy, radius)\n\n'''


def remove_extra_symbols(code):
    lines = code.split("\n")

    # se_lines = [line.startswith("```") for line in lines]
    outputs = []
    if "```" in lines[0]:
        lines = lines[1:]

    for line in lines:
        if not line.startswith("```"):
            outputs.append(line)
        else:
            break

    return "\n".join(outputs)


def extract_code(raw_completions):
    missing = 0
    for item in raw_completions:
        if "[BEGIN]" not in item["completion"] or "[END]" not in item["completion"]:
            if "```python" in item["completion"] or "```" in item["completion"]:
                s1 = item["completion"].find("```python")
                s2 = item["completion"].find("```")
                if s1 == -1:
                    s = s2 + 3
                else:
                    s = s1 + len("```python")
                e = item["completion"].find("```", s)
                if e == -1:
                    missing += 1
                    print(f"Warning: {item['completion']}")
                    continue
                code = item["completion"][s:e].strip()
            else:
                missing += 1
                continue
        else:
            s = item["completion"].index("[BEGIN]") + len("[BEGIN]")
            e = item["completion"].index("[END]")
            code = item["completion"][s:e].strip()
            code = remove_extra_symbols(code)

        item["completion"] = code
    print(f"Missing {missing} segments of code.")
    return raw_completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str)
    parser.add_argument("--sanitized", default=False, action="store_true")
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    outputs = [json.loads(line) for line in open(args.prediction_file).readlines()]

    random.seed(42)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.sanitized:
        test_data = load_dataset("mbpp", "sanitized", split="test").to_list()
    else:
        test_data = load_dataset("mbpp", split="test").to_list()

    print("Number of examples:", len(test_data))

    assert len(test_data) == len(outputs)
    # predictions = [{"task_id": example["task_id"], "prompt": example[prompt_key], "completion": output} for
    #                example, output in zip(duplicate_test_data, outputs)]
    predictions = extract_code(outputs)

    predictions_code_only = [[] for _ in range(len(test_data))]
    for i in range(len(predictions)):
        # predictions_code_only[i // args.unbiased_sampling_size_n].append(predictions[i]["completion"])
        predictions_code_only[i].append(predictions[i]["completion"])
    reference_test_list = ["\n".join(example["test_list"]) for example in test_data]
    assert len(predictions_code_only) == len(reference_test_list)

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pass_at_k_results, eval_results = compute_code_eval(
        references=reference_test_list,
        predictions=predictions_code_only,
        num_workers=1,
    )

    for item, result in zip(predictions, eval_results.values()):
        result.sort()
        item["passed"] = result[0][1]["passed"]

    prediction_save_path = os.path.join(args.save_dir, "mbpp_eval_predictions.json")
    with open(prediction_save_path, "w") as fout:
        json.dump(predictions, fout)

    print(pass_at_k_results)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(pass_at_k_results, fout)


if __name__ == "__main__":
    main()
