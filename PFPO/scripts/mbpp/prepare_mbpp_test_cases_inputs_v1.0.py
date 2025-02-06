from datasets import load_dataset
import argparse
import json
from tqdm import tqdm


_prompt_w_test_cases = """You are an expert Python programmer, and here is your task: 

{}

And here are some test cases in assertion format:

<EXAMPLE TEST CASE INPUT>
{}
</EXAMPLE TEST CASE INPUT>

where only the function name and test cases inputs are included, and the expected outputs are omitted to avoid any confusion.

Now, please **STRICTLY** following the above assertion format to generate **50** more test case inputs for me. Organize your results between <TEST CASE INPUTS> and </TEST CASE INPUTS> tags:

Here is the return format:

<TEST CASE INPUTS>
*assertion inputs 1*
*assertion inputs 2*
...
*assertion inputs 50*
</TEST CASE INPUTS>

Remember my requirements and now let's get started:

"""


PROMPTS = {
    "w_test_cases": _prompt_w_test_cases,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanitized", default=False, action="store_true")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--prompt_type", type=str, default="vanilla")
    args = parser.parse_args()

    if args.sanitized:
        dataset = load_dataset("mbpp", "sanitized", split="test").to_list()
    else:
        dataset = load_dataset("mbpp", split="test").to_list()
    prompt_key = "prompt" if args.sanitized else "text"

    with open(args.output_file, "w") as f:
        for item in tqdm(dataset):
            query = item[prompt_key]
            test_cases = "\n".join([case.split(" == ")[0] for case in item["test_list"]])
            assert len(item["test_list"]), item
            prompt = PROMPTS[args.prompt_type].format(query, test_cases)

            item["prompt"] = prompt
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
