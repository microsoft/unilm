import argparse
import copy
import json
import os
import sys

from tqdm import tqdm

sys.set_int_max_str_digits(0)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from post_processors.code.clean import tag_cleaner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)  # Use the dpo results file
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--difficulty", type=str, default="introductory|interview")
    parser.add_argument("--prompt_file", type=str, )
    args = parser.parse_args()

    data = json.load(open(args.input_file))
    difficulty = args.difficulty.split("|")

    prompt_template = open(args.prompt_file).read()

    parent_dir = os.path.dirname(args.output_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    outputs = []
    for item in tqdm(data):
        if item["difficulty"] not in difficulty:
            continue

        for i, neg in enumerate(item["neg"]):
            question = item["question"]
            if item["starter_code"]:
                question = question + "\n\n" + item["starter_code"]

            neg_code = tag_cleaner(neg)
            if not neg_code:
                continue

            assert neg_code, (neg, item["question"])

            prompt = prompt_template.format(question=question, code=neg_code)

            new_item = copy.deepcopy(item)
            new_item["id"] = f"{item['problem_id']}_neg{i}"
            new_item["prompt"] = prompt
            new_item["neg_code"] = neg_code
            outputs.append(new_item)

    with open(args.output_file, "w") as f:
        for item in outputs:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()

"""
>>> python scripts/apps/pp_critique_difficulty.py \
    --input_file outputs/apps/apps.train.r2c.vanilla.gpt-4o.tem1.0.n11.exec.dpo_v1.0.json \
    --output_file outputs/apps/critique/apps.train.gpt4o.tem1.0.n11.neg.inputs.intro.inter.jsonl \
    --prompt_file prompts/apps/critique_0shot_v1.0.txt 

>>> python azure/gpt_crawler_mp.py \
    --prompt_file ../msranlpintern/reward_modeling/experiments/deepseek-coder-v1.5-ins.7b.apps.r2c.gpt4o.distil.A100.w8.v3.0.s42/apps/checkpoint-400/train.0shot.tem1.0.n10.v1.1.worsen_4o_critic.s42.f100k.jsonl \
    --outfile ../gpt-chat-examples/outputs/apps/critique/r2c.sft.train.0shot.tem1.0.n10.v1.1.worsen_4o_critic.s42.f100k.gpt4o.tem1.0.s42.n1.json_obj.jsonl \
    --model gpt-4o --max_gen_tokens 4096 --temperature 1.0 --num_processes 24 --seed 42 --n 1 --response_format json_object
"""
