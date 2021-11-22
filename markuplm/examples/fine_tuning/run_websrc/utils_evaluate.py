import argparse
import collections
import json
import os
import re
import string
import sys
from copy import deepcopy

from bs4 import BeautifulSoup


class EvalOpts:
    r"""
    The options which the matrix evaluation process needs.
    
    Arguments:
        data_file (str): the SQuAD-style json file of the dataset in evaluation.
        root_dir (str): the root directory of the raw WebSRC dataset, which contains the HTML files.
        pred_file (str): the prediction file which contain the best predicted answer text of each question from the
                         model.
        tag_pred_file (str): the prediction file which contain the best predicted answer tag id of each question from
                             the model.
        result_file (str): the file to write down the matrix evaluation results of each question.
        out_file (str): the file to write down the final matrix evaluation results of the whole dataset.
    """
    def __init__(self, data_file, root_dir, pred_file, tag_pred_file, result_file='', out_file=""):
        self.data_file = data_file
        self.root_dir = root_dir
        self.pred_file = pred_file
        self.tag_pred_file = tag_pred_file
        self.result_file = result_file
        self.out_file = out_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', metavar='data.json', help='Input data JSON file.')
    parser.add_argument('root_dir', metavar='./data', help='The root directory of the raw WebSRC dataset')
    parser.add_argument('pred_file', metavar='pred.json', help='Model predictions.')
    parser.add_argument('tag_pred_file', metavar='tag_pred.json', help='Model predictions.')
    parser.add_argument('--result-file', '-r', metavar='qas_eval.json')
    parser.add_argument('--out-file', '-o', metavar='eval.json',
                        help='Write accuracy metrics to file (default is stdout).')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def make_pages_list(dataset):
    r"""
    Record all the pages which appears in the dataset and return the list.
    """
    pages_list = []
    last_page = None
    for domain in dataset:
        for w in domain['websites']:
            for qa in w['qas']:
                if last_page != qa['id'][:4]:
                    last_page = qa['id'][:4]
                    pages_list.append(last_page)
    return pages_list


def make_qid_to_has_ans(dataset):
    r"""
    Pick all the questions which has answer in the dataset and return the list.
    """
    qid_to_has_ans = {}
    for domain in dataset:
        for w in domain['websites']:
            for qa in w['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    r"""
    Get the word list in the input.
    """
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    r"""
    Calculate the exact match.
    """
    if normalize_answer(a_gold) == normalize_answer(a_pred):
        return 1
    return 0


def compute_f1(a_gold, a_pred):
    r"""
    Calculate the f1 score.
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_pos(f, t_gold, addition, t_pred):
    r"""
    Calculate the POS score.

    Arguments:
        f (str): the html file on which the question is based.
        t_gold (int): the gold answer tag id provided by the dataset (the value correspond to the key element_id).
        addition (int): the addition information used for yes/no question provided by the dataset (the value
                        corresponding to the key answer_start).
        t_pred (list[int]): the tag ids of the tags corresponding the each word in the predicted answer.
    Returns:
        float: the POS score.
    """
    h = BeautifulSoup(open(f), "lxml")
    p_gold, e_gold = set(), h.find(tid=t_gold)
    if e_gold is None:
        if len(t_pred) != 1:
            return 0
        else:
            t = t_pred[0]
            e_pred, e_prev = h.find(tid=t), h.find(tid=t-1)
            if (e_pred is not None) or (addition == 1 and e_prev is not None) or\
                    (addition == 0 and e_prev is None):
                return 0
            else:
                return 1
    else:
        p_gold.add(e_gold['tid'])
        for e in e_gold.parents:
            if int(e['tid']) < 2:
                break
            p_gold.add(e['tid'])
        p = None
        for t in t_pred:
            p_pred, e_pred = set(), h.find(tid=t)
            if e_pred is not None:
                p_pred.add(e_pred['tid'])
                if e_pred.name != 'html':
                    for e in e_pred.parents:
                        if int(e['tid']) < 2:
                            break
                        p_pred.add(e['tid'])
            else:
                p_pred.add(str(t))
            if p is None:
                p = p_pred
            else:
                p = p & p_pred # 预测值的公共祖先序列，except html&body
        return len(p_gold & p) / len(p_gold | p)


def get_raw_scores(dataset, preds, tag_preds, root_dir):
    r"""
    Calculate all the three matrix (exact match, f1, POS) for each question.

    Arguments:
        dataset (dict): the dataset in use.
        preds (dict): the answer text prediction for each question in the dataset.
        tag_preds (dict): the answer tags prediction for each question in the dataset.
        root_dir (str): the base directory for the html files.

    Returns:
        tuple(dict, dict, dict): exact match, f1, pos scores for each question.
    """
    exact_scores = {}
    f1_scores = {}
    pos_scores = {}
    for websites in dataset:
        for w in websites['websites']:
            f = os.path.join(root_dir, websites['domain'], w['page_id'][0:2], 'processed_data',
                             w['page_id'] + '.html')
            for qa in w['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                gold_tag_answers = [a['element_id'] for a in qa['answers']]
                additional_tag_information = [a['answer_start'] for a in qa['answers']]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred, t_pred = preds[qid], tag_preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
                pos_scores[qid] = max(compute_pos(f, t, a, t_pred)
                                      for t, a in zip(gold_tag_answers, additional_tag_information))
    return exact_scores, f1_scores, pos_scores


def make_eval_dict(exact_scores, f1_scores, pos_scores, qid_list=None):
    r"""
    Make the dictionary to show the evaluation results.
    """
    if qid_list is None:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('pos', 100.0 * sum(pos_scores.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        if total == 0:
            return collections.OrderedDict([
                ('exact', 0),
                ('f1', 0),
                ('pos', 0),
                ('total', 0),
            ])
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('pos', 100.0 * sum(pos_scores[k] for k in qid_list) / total),
            ('total', total),
        ])


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval['%s_%s' % (prefix, k)] = new_eval[k]


def main(opts):
    with open(opts.data_file) as f:
        dataset_json = json.load(f)
        dataset = dataset_json['data']
    if isinstance(opts.pred_file, str):
        with open(opts.pred_file) as f:
            preds = json.load(f)
    else:
        preds = opts.pred_file
    if isinstance(opts.tag_pred_file, str):
        with open(opts.tag_pred_file) as f:
            tag_preds = json.load(f)
    else:
        tag_preds = opts.tag_pred_file
    qid_to_has_ans = make_qid_to_has_ans(dataset)
    has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
    no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
    exact, f1, pos = get_raw_scores(dataset, preds, tag_preds, opts.root_dir)
    out_eval = make_eval_dict(exact, f1, pos)
    if has_ans_qids:
        has_ans_eval = make_eval_dict(exact, f1, pos, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
    if no_ans_qids:
        no_ans_eval = make_eval_dict(exact, f1, pos, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
    print(json.dumps(out_eval, indent=2))
    pages_list, write_eval = make_pages_list(dataset), deepcopy(out_eval)
    for p in pages_list:
        pages_ans_qids = [k for k, _ in qid_to_has_ans.items() if p in k]
        page_eval = make_eval_dict(exact, f1, pos, qid_list=pages_ans_qids)
        merge_eval(write_eval, page_eval, p)
    if opts.result_file:
        with open(opts.result_file, 'w') as f:
            w = {}
            for k, v in qid_to_has_ans.items():
                w[k] = {'exact': exact[k], 'f1': f1[k], 'pos': pos[k]}
            json.dump(w, f)
    if opts.out_file:
        with open(opts.out_file, 'w') as f:
            json.dump(write_eval, f)
    return out_eval


if __name__ == '__main__':
    a="$4.99"
    b="$4.99"
    print(compute_exact(a,b))
