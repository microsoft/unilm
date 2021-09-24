import os
from data import SROIETask2
from tqdm import tqdm
import shutil
import zipfile
import torch
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def one_gram(ref, sentence):
    smoothf = SmoothingFunction()
    ref = ref.split()
    sentence = sentence.split()
    return sentence_bleu([ref], sentence, weights=[1, 0, 0, 0], smoothing_function=smoothf.method2)

def mbr(y, candidates, probabilitys):
    assert len(candidates) == len(probabilitys)
    total = 0
    for c, p in zip(candidates, probabilitys):
        w = 1 - one_gram(y, c)
        total += w * p
    return total


class SROIEScorer:
    def __init__(self):
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0
    
    def add_string(self, ref, pred):        
        pred_words = list(pred.split())
        ref_words = list(ref.split())
        self.n_gt_words += len(ref_words)
        self.n_detected_words += len(pred_words)
        for pred_w in pred_words:
            if pred_w in ref_words:
                self.n_match_words += 1
                ref_words.remove(pred_w)

    def score(self):
        prec = self.n_match_words / float(self.n_detected_words) * 100
        recall = self.n_match_words / float(self.n_gt_words) * 100
        f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"


# def main():
#     cand_num = 5
#     ori_scorer = SROIEScorer()
#     new_scorer = SROIEScorer()
#
#     test_dir = '/home/minghaoli/minghaoli/data/SROIE_Task2_Original/test'
#     output_dir = 'temp'
#     os.makedirs(output_dir, exist_ok=True)
#     generate_txt_path = 'checkpoints/DeiTTR/FT_DA2_FT_DA2_LargeR_CDIP_ALL_BSZ16_LR5E_5_UpdatePer16_WUS2_DA_205K/generate-test-nbest5.txt'
#     output_file = None
#     output_fp = None
#
#
#     with open(generate_txt_path, 'r', encoding='utf8') as fp:
#         lines = list(fp.readlines())
#     while not lines[0].startswith('T-0'):
#         lines = lines[1:]
#
#     _, data = SROIETask2(test_dir, None, None)
#     bar = tqdm(data, desc='Caculating:')
#     for t in bar:
#         file_name = t['file_name']
#         image_id = int(t['image_id'])
#
#         gt_line_id = image_id * (1 + cand_num * 3)
#         gt_line = lines[gt_line_id]
#         assert gt_line.startswith('T-{:d}'.format(image_id))
#         gt_str = gt_line[gt_line.find('\t') + 1:].rstrip()
#
#         preds = {}
#         preds_list = []
#         ori_pred = None
#         for cand_i in range(cand_num):
#             pred_line_id = gt_line_id + 2 + cand_i * 3
#             pred_line = lines[pred_line_id]
#             assert pred_line.startswith('D-{:d}'.format(image_id))
#             pred_line = pred_line[pred_line.find('\t') + 1:]
#             pred_str = pred_line[pred_line.find('\t') + 1:].rstrip()
#             pred_score = float(pred_line[:pred_line.find('\t')])
#             pred_score = pow(2, pred_score)
#             preds_list.append((pred_str, pred_score))
#             if pred_str in preds:
#                 preds[pred_str] += pred_score / (cand_i + 2)
#             else:
#                 preds[pred_str] = pred_score / (cand_i + 2)
#
#             if cand_i == 0:
#                 ori_scorer.add_string(gt_str, pred_str)
#                 ori_pred = pred_str
#
#         candidates = preds_list
#         risks = {}
#         min_risk = None
#         min_risk_str = ''
#         for c in candidates:
#             other_candidates = candidates.copy()
#             other_candidates.remove(c)
#             risks[c[0]] = mbr(c[0], [t[0] for t in other_candidates], [t[1] for t in other_candidates])
#             if not min_risk or risks[c[0]] < min_risk:
#                 min_risk_str = c[0]
#                 min_risk = risks[c[0]]
#
#         pred_str = min_risk_str
#
#         new_scorer.add_string(gt_str, pred_str)
#
#         bar.set_description(new_scorer.result_string())
#
#
#     print('Original:', ori_scorer.result_string())
#     print('New:', new_scorer.result_string())
#
#
# if __name__ == '__main__':
#     main()