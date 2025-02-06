import json
import sys
from argparse import ArgumentParser

from datasets import load_dataset
from tqdm import tqdm

sys.set_int_max_str_digits(0)


def main():
    parser = ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_file", type=str, default="prompts/apps/test_input_gen_2shot_v2.0.txt")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--function_only", default=False, action="store_true")
    args = parser.parse_args()

    data = load_dataset("codeparrot/apps", split="train").to_list()
    print(len(data))

    prompt_template = open(args.prompt_file).read()

    with open(args.output_file, "w") as f:
        for item in tqdm(data, desc="Writing prompts"):
            question = item["question"]
            if item["input_output"]:
                input_output = json.loads(item["input_output"])
                if "fn_name" in input_output:
                    question += f"\n\n{item['starter_code']}"
            else:
                input_output = {}

            prompt = prompt_template.replace("[[Question]]", question)
            item["prompt"] = prompt
            if args.function_only:
                if "fn_name" in input_output:
                    f.write(json.dumps(item) + "\n")
            else:
                f.write(json.dumps(item) + "\n")


if __name__ == '__main__':
    main()

"""
>>> python scripts/apps/pp_test_case_gen_inputs_v2.0.py \
    --split train --prompt_file prompts/apps/test_input_gen_2shot_v2.0.txt \
    --output outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.0.inputs.jsonl
    
>>> python scripts/apps/pp_test_case_gen_inputs_v2.0.py \
    --split train --prompt_file prompts/apps/test_input_gen_2shot_v2.1.txt \
        --output outputs/apps/test_case_inputs_gen/apps.train.test_case_inputs.gen.v2.1.func_only.inputs.jsonl --function_only
"""
