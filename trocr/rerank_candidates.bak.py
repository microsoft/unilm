import os
from data import SROIETask2
from tqdm import tqdm
import shutil
import zipfile
import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaForMaskedLM
# Load pre-trained model (weights)
with torch.no_grad():
        model = RobertaForMaskedLM.from_pretrained('checkpoints/MLM/SROIE_Test_MLM_100Epochs_BS32_DDP')
        model.to('cuda')
        model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

NumCandidate = 1

def score(sentence):
    tokenize_input = tokenizer.encode(sentence)[:512]
    tensor_input = torch.tensor([tokenize_input], device=model.device)
    loss=model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.cpu().detach().numpy())

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

class SROIEScorerIgnoreSpace:
    def __init__(self):
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0
    
    def add_string(self, ref, pred):        
        pred_words = list(pred.split())
        self.n_detected_words += len(pred_words)
        
        for pred_w in pred_words:
            if pred_w in ref:
                self.n_match_words += 1
                self.n_gt_words += 1
                ref = ref.replace(pred_w, '', 1)
        self.n_gt_words += len(ref.split())

    def score(self):
        prec = self.n_match_words / float(self.n_detected_words) * 100
        recall = self.n_match_words / float(self.n_gt_words) * 100
        f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"

class SROIEScorerColonSpace:
    def __init__(self):
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0

        self.should_split = 0
        self.total = 0
        self.should_not_split = 0
    
    def add_string(self, ref, pred):        
        pred_words = list(pred.split())          
        ref_words = list(ref.split())        
        for word in pred_words:
            if word.endswith(':') and word != ':':
                self.total += 1
                if word in ref_words:
                    self.should_not_split += 1
                elif word[:-1] in ref_words and ':' in ref_words:
                    self.should_split += 1
                else:
                    print(word, ref_words)
                
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
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f} ShouldSplit: {self.should_split:d} ShouldNotSplit: {self.should_not_split:d} Total: {self.total:d}"

def extract_predictions(generate_txt_path, sroie_data):
    with open(generate_txt_path, 'r', encoding='utf8') as fp:
        lines = list(fp.readlines())
    while not lines[0].startswith('T-0'):
        lines = lines[1:]

    res = {}
    bar = tqdm(sroie_data, desc='Caculating:')
    for t in bar:        
        image_id = int(t['image_id'])
        file_name = t['file_name']
        if not file_name in res:
            res[file_name] = []

        gt_line_id = image_id * (1 + NumCandidate * 3)
        gt_line = lines[gt_line_id]
        assert gt_line.startswith('T-{:d}'.format(image_id))
        gt_str = gt_line[gt_line.find('\t') + 1:].rstrip()
        
        preds = []        
        for cand_i in range(NumCandidate):
            pred_line_id = gt_line_id + 2 + cand_i * 3
            pred_line = lines[pred_line_id]
            assert pred_line.startswith('D-{:d}'.format(image_id))
            pred_line = pred_line[pred_line.find('\t') + 1:]
            pred_str = pred_line[pred_line.find('\t') + 1:].rstrip()
            pred_score = float(pred_line[:pred_line.find('\t')])
            pred_score = pow(2, pred_score)

            preds.append((pred_str, pred_score))
        res[file_name].append((gt_str, preds))

    return res


def main():

    test_dir = '/home/minghaoli/minghaoli/data/SROIE_Task2_Original/test'
    generate_txt_path = 'checkpoints/DeiTTR/FT_DA_Receipt53K_BS2048_LR5E_5_DeiT_TR_Large_FT_CDIP5000_BSZ24_LR5E_5_WestUS2_1200K/generate-test.txt'
    
    _, data = SROIETask2(test_dir, None, None)
    predictions = extract_predictions(generate_txt_path, data)

    scorer = SROIEScorerIgnoreSpace()

    for file_name in predictions:
        for gt_str, preds in predictions[file_name]:
            scorer.add_string(gt_str, preds[0][0])
    
    print(scorer.result_string())

if __name__ == '__main__':
    main()
    
