import ast
import json
from tqdm import tqdm, trange
from collections import defaultdict
import re
import os, sys
import pdb
import sys, json

def clean_special_tokens(input_string):
    pattern = "<.*?>"
    result = re.sub(pattern, "", input_string)
    result = ' '.join(result.split())  # Remove extra spaces
    return result

def find_consecutive_int_indices(numbers):
    for index in range(len(numbers) - 1):
        if numbers[index] == 20032 and numbers[index + 1] == 55:
            return index+2

    return None

def eval(answer_file, json_file, result_file, split_str='Answer:'):
    question_type_dict = json.load(open(json_file, 'rb'))['question_type']
    question_type_r_dict = {}
    for k,v in question_type_dict.items():
        question_type_r_dict[v] = k

    split_str = 'Answer:' # ['Answer:', 'A:'][idx]

    all_answers = defaultdict(dict)
    all_index = []
    with open(answer_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            # pdb.set_trace()
            cols = line.strip().split("\t")
            all_index.append(cols[0])
            all_answers[cols[0]]["answer"] = cols[3]
            all_answers[cols[0]]["question"] = cols[1]
            all_answers[cols[0]]["question_type_id"] = cols[-1]

    all_predictions = defaultdict(list)
    all_prediction_probs = defaultdict(list)
    answer_length = None
    answer_index = None
    with open(result_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.startswith('ST-'):
                src_tokens = ast.literal_eval(line.split('\t')[-1])
                answer_index = find_consecutive_int_indices(src_tokens)
                answer_length = len(src_tokens[answer_index:])
            elif line.startswith('H-'):
                idx = line.split('\t')[0][2:]
                line = line.split('</image>')[-1]
                line = line.split('<image>')[0]
                answer = line.split(split_str)[1].strip()
                answer = clean_special_tokens(answer)

                all_predictions[all_index[int(idx)]].append(answer)
            elif line.startswith('P-'):
                idx = line.split('\t')[0][2:]
                scores_list = list(map(float, line.split('\t')[1].split(" ")))
                answer_scores_list = scores_list[(answer_index-1):]
                mean_score = sum(answer_scores_list) / len(answer_scores_list)
                all_prediction_probs[all_index[int(idx)]].append(mean_score)
    
    correct = 0
    total = 0
    answer_map_dict = {0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F"}

    question_type_correct = {}
    question_type_total = {}
    for k,v in question_type_r_dict.items():
        question_type_correct[k] = 0
        question_type_total[k] = 0

    for qid in all_answers:
        
        hit = True
        prediction = all_prediction_probs[qid].index(max(all_prediction_probs[qid]))
        if answer_map_dict[prediction] != all_answers[qid]["answer"]:
            hit = False

        if hit:
            correct += 1

        question_type_id = int(all_answers[qid]["question_type_id"])
        question_type_total[question_type_id] += 1
        if hit:
            question_type_correct[question_type_id] += 1

        total += 1

    final_scores = {}
    final_scores["acc"] = correct / total * 100.0
    print("{}\t{}\t{}".format(correct, total, final_scores))
    for k,v in question_type_r_dict.items(): 
        print(k, v, question_type_correct[k] / max(question_type_total[k], 1))



if __name__ == "__main__":
    save_dir = '/path/to/data'
    json_file = f'{save_dir}/SEED-Bench/SEED-Bench.json'
    result_file = sys.argv[1]
    if 'task12' in result_file:
        answer_file = f'{save_dir}/SEED-Bench/seed_bench_task12_pplformat.answer'
    elif 'task10' in result_file:
        answer_file = f'{save_dir}/SEED-Bench/seed_bench_task10_pplformat.answer'
    elif 'task11' in result_file:
        answer_file = f'{save_dir}/SEED-Bench/seed_bench_task11_pplformat.answer'
    else:
        answer_file = f'{save_dir}/SEED-Bench/seed_bench_pplformat.answer'
    eval(answer_file, json_file, result_file)