import argparse
import copy
import json

from datasets import load_dataset

_print_inter_template = """I have a program with the instruction or comments to indicate the usage of it. I want to know the intermediate running process of the program, so please help me:
(1) Add several `print` clauses after each line of code to print the values of the involved variables. Please follow the format: `print(f"x: ", x), where `x` is one of the intermediate values. Each print clause is used for checking exactly one variable. Return the completed program using <NEW PROGRAM></NEW PROGRAM>.
(2) Give me 5 test cases according to the docstring or instruction of the program. Use <TEST CASE></TEST CASE> to return them to me. Each pair of <TEST CASE></TEST CASE> should include only one case. The cases should also in python code format so that I can use `eval(test_case)` to obtain the results directly.

I will first give you an example:

Instruction: Write a function to sort a given matrix in ascending order according to the sum of its rows.
Program:
def sort_matrix(matrix):
    matrix.sort(key=sum)
    return matrix
    
Here's the program with added `print` statements to track the intermediate values:

<NEW PROGRAM>
def sort_matrix(matrix):
    # Print the original matrix
    print("Original matrix:", matrix)

    # Print the sum of each row before sorting
    for row in matrix:
        print(f"Sum of {row}: ", sum(row))

    matrix.sort(key=sum)

    # Print the matrix after sorting
    print("Sorted matrix:", matrix)

    return matrix
</NEW PROGRAM>

Next, I'll provide you with 5 test cases based on the instruction:

<TEST CASE>
sort_matrix([[3, 2, 1], [1, 1, 1], [2, 2, 2]])
</TEST CASE>

<TEST CASE>
sort_matrix([[10, 5, 3], [0, 0, 0], [2, 2, 6]])
</TEST CASE>

<TEST CASE>
sort_matrix([[1, 2, 3], [3, 2, 1], [2, 2, 2]])
</TEST CASE>

<TEST CASE>
sort_matrix([[7], [3, 5], [1, 1, 1, 1]])
</TEST CASE>

<TEST CASE>
sort_matrix([[-1, -1, -1], [1, 0, -1], [0, 0, 0]])
</TEST CASE>

Now, I will give a new program and let's get started to process it:

Instruction: {}
Program:
{}

"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_file", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    results = json.load(open(args.result_file))
    print(len(results))

    predictions = json.load(open(args.prediction_file))
    print(len(predictions))
    outputs = []
    for res, pred in zip(results.values(), predictions):
        orig_prompt = pred["prompt"]
        orig_completion = pred["completion"]
        new_prompt = _print_inter_template.format(orig_prompt, pred["completion"], row="row")
        pred["orig_prompt"] = orig_prompt
        pred["orig_completion"] = orig_completion
        pred["prompt"] = new_prompt
        pred["passed"] = res[0][1]["passed"]
        outputs.append(pred)

    with open(args.output_file, "w", encoding="utf8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    main()
