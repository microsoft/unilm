import argparse
import json
import os
import re
import sys
import time
import openai

import eval_vllm.util as util

from tqdm import tqdm
from multiprocessing import Pool

openai.api_key = os.environ["OPENAI_API_KEY"]
if os.environ.get("OPENAI_ORGANIZATION") is not None:
    openai.organization = os.environ["OPENAI_ORGANIZATION"]

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

def request_one_example(input_t):
    example = input_t[0]
    args = input_t[1]
    prompt_template = input_t[2]
    engine = input_t[3]
    completion_kwargs = input_t[4]

    question = example["question"]
    answer = example["answer"]
    temp_instr = prompt_template.format(instruction=question)
    messages = [{"role": "user", "content": temp_instr}]
    retry_count = 0
    while retry_count < args.retry_limit:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages, 
                **completion_kwargs
            )
            return question, answer, temp_instr, response["choices"][0]["message"]["content"], retry_count
        except Exception as e:
            print(e)
            retry_count += 1
            time.sleep(args.failure_sleep_time)
    return question, answer, temp_instr, "", retry_count

def evaluate_one_task(args, engine, completion_kwargs, prompt_template, task_name, sample):
    res_completions = []
    math_answers = []
    pbar = []
    for example in sample:
        pbar.append([example, args, prompt_template, engine, completion_kwargs])
    pbar = tqdm(pbar, desc=f"{task_name}: requesting openai...")
    with Pool(args.num_threads) as p:
        for output in p.imap(request_one_example, pbar):
            question = output[0]
            answer = output[1]
            prompt = output[2]
            completion = output[3]
            retry_count = output[4]
            res_completions.append(completion)
            math_answers.append(answer)

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
        args.save_dir = os.path.join("results", args.openai_model + f".{args.prompt_template}")
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
    engine=args.openai_model
    completion_kwargs = {
        "temperature": 0., 
        "top_p": 1., 
        "n": 1, 
        "stop": [], 
        "max_tokens": 2048
    }
    print(f"engine: {engine}")
    print(f"completion_kwargs: {completion_kwargs}")
    num_threads = args.num_threads
    failure_sleep_time=args.failure_sleep_time
    retry_limit=args.retry_limit
    print(f"num_threads: {num_threads}")
    print(f"failure_sleep_time: {failure_sleep_time}")
    print(f"retry_limit: {retry_limit}")

    # evaluate tasks
    layer_MATH_task2acc = {}
    layer_college_math_task2acc = {}
    layer_top_task2acc = {}
    full_MATH_size = 0
    full_college_math_size = 0
    full_size = 0
    for task_name, sample in task2sample.items():
        try:
            acc = evaluate_one_task(args, engine, completion_kwargs, prompt_template, task_name, sample)
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
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo-0613")  # model path
    parser.add_argument("--num_threads", type=int, default=10)  # num_threads requesting openai
    parser.add_argument("--failure_sleep_time", type=int, default=10)  # sleep time (in seconds) of openai request failure
    parser.add_argument("--retry_limit", type=int, default=200)  # retry limit for openai request failure
    parser.add_argument("--data_file", type=str, default='data/full_test.json')  # data path
    parser.add_argument("--target_tasks", type=str, default=None)  # # choose from gsm8k,MATH.Algebra,MATH.Counting_&_Probability,MATH.Geometry,MATH.Intermediate_Algebra,MATH.Number_Theory,MATH.Prealgebra,MATH.Precalculus,college_math.algebra,college_math.precalculus,college_math.calculus,college_math.vector_calculus,college_math.probability,college_math.linear_algebra,college_math.differential_equation,tal,gaokao_bench_math_en,math23k_en,ape210k_en,agieval.gaokao-math-en,agieval.math,agieval.sat-math
    parser.add_argument("--save_dir", type=str, default=None)  # data path
    parser.add_argument("--max_num_examples_per_task", type=int, default=2000) # max_num_examples_per_task, set -1 to disable it
    parser.add_argument("--prompt_template", type=str, default="alpaca") # choose from [none, alpaca, alpaca_force_ans, alpaca_cot]
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)