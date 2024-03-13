import argparse
import json
import os
import re
import sys

import eval_vllm.util as util

from vllm import LLM, SamplingParams
from tqdm import tqdm

MAX_INT = sys.maxsize

TEMPLATE_DICT = {
    "none": (
        "{instruction}"
    ), 
    "alpaca": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ), 
    "alpaca_force_ans": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\nTry to conclude your response with 'The answer is ...'.\n### Response:"
    ), 
    "alpaca_cot": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
}

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def evaluate_one_task(args, model, sampling_params, prompt_template, task_name, sample):
    math_ins = []
    math_answers = []
    for item in sample:
        question = item["question"]
        answer = item["answer"]
        temp_instr = prompt_template.format(instruction=question)
        math_ins.append(temp_instr)
        math_answers.append(answer)

    batch_math_ins = batch_data(math_ins, batch_size=args.batch_size)
    res_completions = []
    for batch_prompt in batch_math_ins:
        completions = model.generate(batch_prompt, sampling_params)
        for output in completions:
            prompt_temp = output.prompt
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    fw = open(os.path.join(args.save_dir, task_name.strip(".") + ".prediction.json"), "w")
    results = []
    for idx, (example, completion, answer) in enumerate(zip(sample, res_completions, math_answers)):
        res, clean_prediction_ans, clean_reference_ans = util.is_correct(completion, answer, verbose=args.verbose)
        results.append(res)
        dump = {
            "question": example["question"], 
            "answer": answer, 
            "completion": completion, 
            'clean_reference_ans': clean_reference_ans, 
            'clean_prediction_ans': clean_prediction_ans, 
            "judge": res
        }
        dump = json.dumps(dump, ensure_ascii=False)
        fw.write(dump + "\n")
    fw.close()
    acc = sum(results) / len(results)

    fw = open(os.path.join(args.save_dir, task_name.strip(".") + ".metric.json"), "w")
    metric = {
        "task_name": task_name, 
        "test_size": len(results), 
        "accuracy": acc, 
    }
    print(metric)
    print(f"evaluate task done.")
    metric = json.dump(metric, fw, ensure_ascii=False)
    fw.close()
    return acc

def main(args):
    if args.save_dir is None:
        args.save_dir = os.path.join("results", args.model_name_or_path.replace("/", ".").strip(".") + f".{args.prompt_template}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    task2sample = {}
    with open(args.data_file) as fd:
        for line in tqdm(fd, desc="load data..."):
            example = json.loads(line)
            task = example["data_topic"]
            if args.target_tasks is not None:
                if task not in args.target_tasks:
                    continue
            if task not in task2sample:
                task2sample[task] = []
            task2sample[task].append(example)
    if args.max_num_examples_per_task != -1:
        task2sample_t = {}
        for task_name, sample in task2sample.items():
            task2sample_t[task_name] = sample[:args.max_num_examples_per_task]
        task2sample = task2sample_t
    print("load data done.")
    for task_name, sample in task2sample.items():
        print(f"evaluating task name: {task_name}; sample size: {len(sample)}")

    prompt_template = TEMPLATE_DICT[args.prompt_template]
    print(f"using prompt template: {args.prompt_template}\n{prompt_template}")

    # Init model
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)
    print("init model done.")
    stop_tokens = ["Question:", "Question", "USER:", "USER", "ASSISTANT:", "ASSISTANT", "Instruction:", "Instruction", "Response:", "Response", "</s>"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048, stop=stop_tokens)
    print(f"init sampling params done: {sampling_params}")

    # evaluate tasks
    layer_MATH_task2acc = {}
    layer_college_math_task2acc = {}
    layer_top_task2acc = {}
    full_MATH_size = 0
    full_college_math_size = 0
    full_size = 0
    for task_name, sample in task2sample.items():
        try:
            acc = evaluate_one_task(args, model, sampling_params, prompt_template, task_name, sample)
            test_size = len(sample)
            full_size += test_size
            if task_name.startswith("MATH."):
                layer_MATH_task2acc[task_name] = {"accuracy": acc, "test_size": test_size}
                full_MATH_size += test_size
            elif task_name.startswith("college_math."):
                layer_college_math_task2acc[task_name] = {"accuracy": acc, "test_size": test_size}
                full_college_math_size += test_size
            else:
                layer_top_task2acc[task_name] = {"accuracy": acc, "test_size": test_size}
        except Exception as e:
            print(e)
            continue

    # compute MATH acc
    MATH_acc = 0
    for task_name, task_metric in layer_MATH_task2acc.items():
        acc = task_metric["accuracy"]
        test_size = task_metric["test_size"]
        weight = test_size / full_MATH_size
        MATH_acc += weight * acc
    layer_top_task2acc["MATH"] = {"accuracy": MATH_acc, "test_size": full_MATH_size, "subset_metric": layer_MATH_task2acc}

    # compute college_math acc
    college_math_acc = 0
    for task_name, task_metric in layer_college_math_task2acc.items():
        acc = task_metric["accuracy"]
        test_size = task_metric["test_size"]
        weight = test_size / full_college_math_size
        college_math_acc += weight * acc
    layer_top_task2acc["college_math"] = {"accuracy": college_math_acc, "test_size": full_college_math_size, "subset_metric": layer_college_math_task2acc}

    # compute micro & macro avg
    micro_acc = 0
    macro_acc = 0
    for task_name, task_metric in layer_top_task2acc.items():
        acc = task_metric["accuracy"]
        test_size = task_metric["test_size"]
        weight = test_size / full_size
        micro_acc += weight * acc
        macro_acc += acc
    macro_acc /= len(layer_top_task2acc)
    layer_top_task2acc["micro_average_accuracy"] = micro_acc
    layer_top_task2acc["macro_average_accuracy"] = macro_acc

    print("evaluate all done.")
    print(json.dumps(layer_top_task2acc, indent=4))
    fw = open(os.path.join(args.save_dir, "all.metric.json"), "w")
    layer_top_task2acc = json.dump(layer_top_task2acc, fw, ensure_ascii=False)
    fw.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)  # model path
    parser.add_argument("--data_file", type=str, default='data/full_test.json')  # data path
    parser.add_argument("--target_tasks", type=str, default=None)  # choose from gsm8k,MATH.Algebra,MATH.Counting_&_Probability,MATH.Geometry,MATH.Intermediate_Algebra,MATH.Number_Theory,MATH.Prealgebra,MATH.Precalculus,college_math.algebra,college_math.precalculus,college_math.calculus,college_math.vector_calculus,college_math.probability,college_math.linear_algebra,college_math.differential_equation,tal,gaokao_bench_math_en,math23k_en,ape210k_en,agieval.gaokao-math-en,agieval.math,agieval.sat-math
    parser.add_argument("--save_dir", type=str, default=None)  # data path
    parser.add_argument("--max_num_examples_per_task", type=int, default=2000) # max_num_examples_per_task, set -1 to disable it
    parser.add_argument("--batch_size", type=int, default=60)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=4)  # num_gpus
    parser.add_argument("--prompt_template", type=str, default="alpaca") # choose from [none, alpaca, alpaca_force_ans, alpaca_cot]
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)