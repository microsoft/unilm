import collections
import json
import os.path
from argparse import ArgumentParser
from glob import glob
import random
from tqdm import tqdm


def sample_step(response: str, upper_step_ratio: float, sample_ratio: float):
    orig_lines = response.split("\n")
    lines = [(i, line) for i, line in enumerate(orig_lines)]
    lines = [(i, line) for i, line in lines if line.strip()]

    if len(lines) < 5:
        return []

    upper_step = int(len(lines) * upper_step_ratio)
    if upper_step == 0:
        return []

    sample_step_num = int(upper_step * sample_ratio)
    if sample_step_num == 0:
        return []

    sample_steps = random.sample(lines[:upper_step], sample_step_num)
    if len(sample_steps) == 0:
        return []

    step_prefixes = []
    for i, line in sample_steps:
        step_prefixes.append("\n".join(orig_lines[:(i + 1)]))

    return step_prefixes


def get_pred_set(preds):
    if isinstance(preds[0], list):
        tmp = []
        for pred in preds:
            tmp.append(str(sorted(pred)))
        tmp = set(tmp)
    elif isinstance(preds[0], str):
        tmp = set(preds)
    else:
        raise ValueError(f"Unknown type {type(preds[0])}")
    return tmp


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--upper_step_ratio", type=float, default=0.6)
    parser.add_argument("--sample_ratio", type=float, default=0.3)
    parser.add_argument("--filter_all_correct", action="store_true", default=False)
    parser.add_argument("--filter_all_same", action="store_true", default=False)
    parser.add_argument("--sample_over_p", default=-1, type=int)
    args = parser.parse_args()

    if os.path.exists(args.input_file):
        if args.input_file.endswith(".json"):
            data = json.load(open(args.input_file))
        else:
            data = [json.loads(line) for line in open(args.input_file).readlines()]
    else:
        data = []
        for file in sorted(glob(args.input_file)):
            if ".metrics" in file:
                continue
            print(file)
            if file.endswith(".json"):
                data += json.load(open(file))
            else:
                data += [json.loads(line) for line in open(file).readlines()]

    # # First use self-consistency to construct pseudo labels
    # for item in data:
    #     all_preds = collections.Counter()
    #     for pred in item["pred"]:
    #         if isinstance(pred, list):
    #             all_preds.update(pred)
    #         else:
    #             all_preds[pred] += 1
    #
    #     pseudo_label = all_preds.most_common(1)[0][0]
    #     item["sc_label_0"] = pseudo_label

    outputs = []
    num = 0
    for item in tqdm(data):
        if args.filter_all_correct and all(item["res"]):
            continue
        if args.filter_all_same and len(set(item["pred"])) == 1:
            continue

        prefixes = []
        prefix_ids = []
        if not item["response"]:
            item["prefix"] = []
            continue

        for resp_id, resp in enumerate(item["response"]):
            response_prefixes = sample_step(resp, args.upper_step_ratio, args.sample_ratio)
            prefixes.extend(response_prefixes)
            prefix_ids.extend([f"{item['id']}_{resp_id}_{i}" for i in range(len(response_prefixes))])

        if args.sample_over_p > 0:
            if len(prefixes) > args.sample_over_p:
                prefixes = random.sample(prefixes, args.sample_over_p)
                prefix_ids = random.sample(prefix_ids, args.sample_over_p)
        item["prefix"] = prefixes
        item["prefix_id"] = prefix_ids

        item.pop("response")
        item.pop("pred")
        item.pop("res")
        item.pop("sc_pred")
        item.pop("sc_res")

        outputs.append(item)
        num += len(prefixes)

    json.dump(outputs, open(args.output_file, "w"), indent=2)
    print(f"Number of prefixes: {num}")
    print(len(outputs))


if __name__ == "__main__":
    main()

"""
>>> python ~/gpt-chat-examples/scripts/math/deepseek_math_sample_steps.py \
    --input_file "math.test.v1.1.0shot.n10.tem1.0.p0.9.8-of-?.json" \
    --output_file math.test.v1.1.0shot.n10.tem1.0.p0.9.prefix.upper0.6.r0.3.json --upper_step_ratio 0.6 --sample_ratio 0.3
    
>>> python scripts/math/deepseek_math_sample_steps.py --input_file "../msranlpintern/share/models/deepseek-math-7b-instruct/meta_math/sub_math.cot.train.0shot.n10.tem1.0.p0.9.v1.0.?-of-24.json" --output_file "../msranlpintern/share/models/deepseek-math-7b-instruct/meta_math/sub_math.cot.train.0shot.n10.tem1.0.p0.9.v1.0.upper0.7.r0.3.inter_step.filter_all_true.json" --upper_step_ra
tio 0.7 --sample_ratio 0.3 --filter_all_correct

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/share/models/mathstral-7B-v0.1/mathscale4o/split-0-of-11/train.330k.boxed.v1.0.0-of-11.0shot.n20.tem1.0.p0.9.[0-9]*-of-64.json" \
    --output_file "../msranlpintern/share/models/mathstral-7B-v0.1/mathscale4o/split-0-of-11/train.330k.boxed.v1.0.0-of-11.0shot.n20.tem1.0.p0.9.upper0.7.r0.3.inter_step.json" \
    --upper_step_ratio 0.7 --sample_ratio 0.3

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/share/models/mathstral-7B-v0.1/mathscale4o/split-0-of-11/train.330k.boxed.v1.0.0-of-11.0shot.n20.tem1.0.p0.9.[0-9]*-of-64.json" \
    --output_file "../msranlpintern/share/models/mathstral-7B-v0.1/mathscale4o/split-0-of-11/train.330k.boxed.v1.0.0-of-11.0shot.n20.tem1.0.p0.9.upper0.7.r0.3.filter_same.json" \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.?-of-8.json" \
    --output_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.filter_same.json" \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same

Number of prefixes: 19521786
309876

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.?-of-8.json" \
    --output_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.sft.V100.tp2dp8.v2.0.s42/checkpoint-800/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.json" \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10

Number of prefixes: 3098719
309876

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/share/models/llama3.1_8b_mathscale4o/model_lr1e-5_batch512_epochs3_gpus8_linearSchedule/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-32.json" \
    --output_file "../msranlpintern/share/models/llama3.1_8b_mathscale4o/model_lr1e-5_batch512_epochs3_gpus8_linearSchedule/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.json" \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10

Number of prefixes: 2879063
287913

############################################################ ITERATION 1 #######################################################

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10

Number of prefixes: 2389255
238928

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample10.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 10

Number of prefixes: 2389255
238928


>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/llama3.1.8b.mathscale4o.process-dpo.iter0.A100.dp8.v2.2.s42/checkpoint-1200/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample32.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 32

Number of prefixes: 7512389
238928

>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-8.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample32.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 32


>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-[0-9]-of-10/cot.de_con.[0-9]-of-10.n8.tem1.0.p1.0.*-of-16.s0.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/cot.de_con.n8.tem1.0.p1.0.s0.upper0.7.r0.3.sample16.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 16


>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/share/models/mathstral-7B-v0.1/mathscale4o/500k-split-*-of-20/train.500k.de_con.boxed.v1.0.*-of-20.0shot.n10.tem1.0.p0.9.*-of-64.*json" \
    --output_file ../msranlpintern/share/models/mathstral-7B-v0.1/mathscale4o/train.500k.de_con.boxed.v1.0.n10.tem1.0.p0.9.upper0.7.r0.3.sample16.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 16

Number of prefixes: 5053462
345626


>>> python scripts/math/deepseek_math_sample_steps.py \
    --input_file "../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/830k-split-0-of-10/cot.de_con.0-of-10.n8.tem1.0.p1.0.*-of-8.s0.json" \
    --output_file ../msranlpintern/reward_modeling/experiments/mathstral.mathscale4o.process-dpo.iter0.V100.tp8dp48.v2.2.fix.s42/checkpoint-600/numina/cot.de_con.n8.tem1.0.p1.0.orig_0-of-8.s0.upper0.7.r0.3.sample32.filter_same.json \
    --upper_step_ratio 0.7 --sample_ratio 0.3 --filter_all_same --sample_over_p 32

"""

