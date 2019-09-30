"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import json
import argparse
import math
import string
from multiprocessing import Pool, cpu_count
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
# pip install py-rouge
import rouge
import time
import tempfile
import shutil

from pytorch_pretrained_bert.tokenization import BertTokenizer
# pip install pyrouge
from cnndm.bs_pyrouge import Rouge155


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

# Required parameters
parser.add_argument("--gold", type=str, help="Gold output file.")
parser.add_argument("--pred", type=str, help="Input prediction file.")
parser.add_argument("--split", type=str, default="",
                    help="Data split (train/dev/test).")
parser.add_argument("--save_best", action='store_true',
                    help="Save best epoch.")
parser.add_argument("--only_eval_best", action='store_true',
                    help="Only evaluate best epoch.")
parser.add_argument("--trunc_len", type=int, default=60,
                    help="Truncate line by the maximum length.")
parser.add_argument("--duplicate_rate", type=float, default=0.7,
                    help="If the duplicat rate (compared with history) is large, we can discard the current sentence.")
default_process_count = max(1, cpu_count() - 1)
parser.add_argument("--processes", type=int, default=default_process_count,
                    help="Number of processes to use (default %(default)s)")
parser.add_argument("--perl", action='store_true',
                    help="Using the perl script.")
parser.add_argument('--lazy_eval', action='store_true',
                    help="Skip evaluation if the .rouge file exists.")
args = parser.parse_args()

SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]
evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2,
                        limit_length=False, apply_avg=True, weight_factor=1.2)


def test_rouge(cand, ref):
    temp_dir = tempfile.mkdtemp()
    candidates = cand
    references = ref
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:
        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155(temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100
    )


def count_tokens(tokens):
    counter = {}
    for t in tokens:
        if t in counter.keys():
            counter[t] += 1
        else:
            counter[t] = 1
    return counter


def get_f1(text_a, text_b):
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()
    if len(tokens_a) == 0 or len(tokens_b) == 0:
        return 1 if len(tokens_a) == len(tokens_b) else 0
    set_a = count_tokens(tokens_a)
    set_b = count_tokens(tokens_b)
    match = 0
    for token in set_a.keys():
        if token in set_b.keys():
            match += min(set_a[token], set_b[token])
    p = match / len(tokens_a)
    r = match / len(tokens_b)
    return 2.0 * p * r / (p + r + 1e-5)


_tok_dict = {"(": "-LRB-", ")": "-RRB-",
             "[": "-LSB-", "]": "-RSB-",
             "{": "-LCB-", "}": "-RCB-"}


def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True


def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)


def remove_duplicate(l_list, duplicate_rate):
    tk_list = [l.lower().split() for l in l_list]
    r_list = []
    history_set = set()
    for i, w_list in enumerate(tk_list):
        w_set = set(w_list)
        if len(w_set & history_set)/len(w_set) <= duplicate_rate:
            r_list.append(l_list[i])
        history_set |= w_set
    return r_list


def process_eval(eval_fn):
    gold_list = []
    with open(args.gold, "r", encoding="utf-8") as f_in:
        for l in f_in:
            line = l.strip().replace(" <S_SEP> ", '\n')
            gold_list.append(line)

    pred_list = []
    with open(eval_fn, "r", encoding="utf-8") as f_in:
        for l in f_in:
            buf = []
            for sentence in l.strip().split("[X_SEP]"):
                sentence = fix_tokenization(sentence)
                if any(get_f1(sentence, s) > 1.0 for s in buf):
                    continue
                s_len = len(sentence.split())
                if s_len <= 4:
                    continue
                buf.append(sentence)
            if args.duplicate_rate and args.duplicate_rate < 1:
                buf = remove_duplicate(buf, args.duplicate_rate)
            if args.trunc_len:
                num_left = args.trunc_len
                trunc_list = []
                for bit in buf:
                    tk_list = bit.split()
                    n = min(len(tk_list), num_left)
                    trunc_list.append(' '.join(tk_list[:n]))
                    num_left -= n
                    if num_left <= 0:
                        break
            else:
                trunc_list = buf
            line = "\n".join(trunc_list)
            pred_list.append(line)
    with open(eval_fn+'.post', 'w', encoding='utf-8') as f_out:
        for l in pred_list:
            f_out.write(l.replace('\n', ' [X_SEP] ').strip())
            f_out.write('\n')
    # rouge scores
    if len(pred_list) < len(gold_list):
        # evaluate subset
        gold_list = gold_list[:len(pred_list)]
    assert len(pred_list) == len(gold_list)
    if args.perl:
        scores = test_rouge(pred_list, gold_list)
    else:
        scores = evaluator.get_scores(pred_list, [[it] for it in gold_list])
    return eval_fn, scores


def main():
    if args.perl:
        eval_fn_list = list(glob.glob(args.pred))
    else:
        eval_fn_list = [eval_fn for eval_fn in glob.glob(args.pred) if not(
            args.lazy_eval and Path(eval_fn+".rouge").exists())]
    eval_fn_list = list(filter(lambda fn: not(fn.endswith(
        '.post') or fn.endswith('.rouge')), eval_fn_list))

    if args.only_eval_best:
        best_epoch_dict = {}
        for dir_path in set(Path(fn).parent for fn in eval_fn_list):
            fn_save = os.path.join(dir_path, 'save_best.dev')
            if Path(fn_save).exists():
                with open(fn_save, 'r') as f_in:
                    __, o_name, __ = f_in.read().strip().split('\n')
                    epoch = o_name.split('.')[1]
                    best_epoch_dict[dir_path] = epoch
        new_eval_fn_list = []
        for fn in eval_fn_list:
            dir_path = Path(fn).parent
            if dir_path in best_epoch_dict:
                if Path(fn).name.split('.')[1] == best_epoch_dict[dir_path]:
                    new_eval_fn_list.append(fn)
        eval_fn_list = new_eval_fn_list

    logger.info("***** Evaluation: %s *****", ','.join(eval_fn_list))
    num_pool = min(args.processes, len(eval_fn_list))
    p = Pool(num_pool)
    r_list = p.imap_unordered(process_eval, eval_fn_list)
    r_list = sorted([(fn, scores)
                     for fn, scores in r_list], key=lambda x: x[0])
    rg2_dict = {}
    for fn, scores in r_list:
        print(fn)
        if args.perl:
            print(rouge_results_to_str(scores))
        else:
            rg2_dict[fn] = scores['rouge-2']['f']
            print(
                "ROUGE-1: {}\tROUGE-2: {}\n".format(scores['rouge-1']['f'], scores['rouge-2']['f']))
            with open(fn+".rouge", 'w') as f_out:
                f_out.write(json.dumps(
                    {'rg1': scores['rouge-1']['f'], 'rg2': scores['rouge-2']['f']}))
    p.close()
    p.join()

    if args.save_best:
        # find best results
        group_dict = {}
        for k, v in rg2_dict.items():
            d_name, o_name = Path(k).parent, Path(k).name
            if (d_name not in group_dict) or (v > group_dict[d_name][1]):
                group_dict[d_name] = (o_name, v)
        # compare and save the best result
        for k, v in group_dict.items():
            fn = os.path.join(k, 'save_best.'+args.split)
            o_name_s, rst_s = v
            should_save = True
            if Path(fn).exists():
                with open(fn, 'r') as f_in:
                    rst_f = float(f_in.read().strip().split('\n')[-1])
                if rst_s <= rst_f:
                    should_save = False
            if should_save:
                with open(fn, 'w') as f_out:
                    f_out.write('{0}\n{1}\n{2}\n'.format(k, o_name_s, rst_s))


if __name__ == "__main__":
    main()
