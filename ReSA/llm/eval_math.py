import os
import json
import time

from arch.model import create_kv_cache
from utils.math_utils import *

import random
random.seed(42)

import torch.distributed as dist

class MathArgs:
    prompt_type: str = 'direct'
    adapt_few_shot: bool = False
    num_shots: int = 0
    generate_length: int = 1024
    save_freq: int = 128


def get_rank():
    return int(os.environ['RANK'])


def first_print(*args):
    if get_rank() == 0:
        print(*args)


def load_data(data_name, split, data_dir):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    assert os.path.exists(data_file), f"File not found: {data_file}"
    examples = list(load_jsonl(data_file))
    
    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples


def prepare_data(data_name, args, limit):
    examples = load_data(data_name, 'test', '../math_data')

    examples = examples[:limit]
    if len(examples) < limit:
        repeat_times = (limit + len(examples) - 1) // len(examples)
        print(f"Warning: {data_name} has only {len(examples)} examples, repeat {repeat_times} times")
        examples = examples * repeat_times

    out_folder = os.path.join(args.output_folder, data_name)
    out_file = os.path.join(args.output_folder, data_name, os.path.basename(args.checkpoint_dir) + "_" + args.save_feature + ".jsonl")
    os.makedirs(f"{out_folder}", exist_ok=True)

    return examples, out_file


def model_generation(model, prompts, max_length, math_args):
    outputs = []
    for i in range(0, len(prompts), model._batch_size):
        prompt = prompts[i:i + model._batch_size]
        net_input = model.tok_batch_encode(prompt)[0].cuda()
        
        generate_length = math_args.generate_length
        if net_input.size(1) + generate_length > max_length:
            print("Warning: input too long, reduce generation length to", max_length - net_input.size(1))
            generate_length = max_length - net_input.size(1)
        if generate_length <= 0:
            print("Warning: input too long, skip sample")
            continue
        
        kv_cache = create_kv_cache(model.model.args, model._batch_size)
        output = model._model_generate(net_input, net_input.size(1) + generate_length, kv_cache=kv_cache)
        prediction = [model.tok_decode(o) for o in output.cpu().tolist()]
        prediction = [pred[len(prpt):] for prpt, pred in zip(prompt, prediction)]

        outputs.extend(prediction)
    return outputs


def evaluate(args, model, limit):
    math_args = MathArgs()
    if 'DeepSeek-R1-Distill' in args.checkpoint_dir:
        math_args.prompt_type = 'r1'
        math_args.generate_length = 16384
        math_args.save_freq = 128
    else:
        raise ValueError("Please use math fine-tuned model for evaluation.")
        
    # 1024 for prompt
    max_length = 1024 + math_args.generate_length
    # here we set max_seq_len to max_length, to create less kv cache for an original 64k model
    model._model.args.max_seq_len = 32768
    # model._model.args.max_seq_len = max_length
    
    data_names = "minerva_math,gaokao2023en,olympiadbench,aime24,amc23"
    # data_names = "gsm8k,math,svamp,asdiv,mawps,carp_en,tabmwp,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23"
    
    data_list = data_names.split(",")
    for data_name in data_list:
        first_print("Start eval on", data_name)
        examples, out_file = prepare_data(data_name, args, limit)
        first_print("Total samples:", len(examples))
        finished_examples = []
        finished_examples_num = 0
        if os.path.exists(out_file):
            finished_examples = list(load_jsonl(out_file))
            finished_examples_num = len(finished_examples)
        examples = examples[finished_examples_num:]
        first_print("Left samples:", len(examples))

        new_samples = []
        for i in range(0, len(examples), math_args.save_freq):
            first_print("Current samples:", i)
            out_example = eval_math_save_part(model, data_name, 
                                              examples[i:i+math_args.save_freq], math_args,
                                              max_length)
            new_samples.extend(out_example)
            if get_rank() == 0:
                save_jsonl(finished_examples + new_samples, out_file)
            first_print("saved to", out_file)


def eval_math_save_part(model, data_name, examples, math_args, max_length):
    idx_span = (len(examples) + dist.get_world_size() - 1) // dist.get_world_size()
    idx_start, idx_end = idx_span * get_rank(), idx_span * (get_rank() + 1)
    examples = examples[idx_start:idx_end]

    executor = PythonExecutor(get_answer_from_stdout=True)
    samples = []
    for example in examples:
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, math_args)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [sample["prompt"] for sample in samples]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if math_args.prompt_type in ["cot", "pal"] else 4

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        # self.first_print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        
        outputs = model_generation(model, prompts, max_length, math_args)
        
        # mean_generate_length /= len(prompts)
        # first_print("Mean generate length:", mean_generate_length)
        
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))
        
        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)
    
    # unsolved samples
    # first_print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in ["</s>", "<|im_end|>", "<|endoftext|>"]:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, math_args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i : (i + 1)]
        result = results[i : (i + 1) ]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)
    
    all_samples_gather = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_samples_gather, all_samples)
    all_samples_gather = sum(all_samples_gather, [])
    
    return all_samples_gather