import os
from data import SROIETask2
from tqdm import tqdm
import shutil
import zipfile
import torch
import numpy as np

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# # Load pre-trained model (weights)
# with torch.no_grad():
#         model = GPT2LMHeadModel.from_pretrained('gpt2')
#         model.to('cuda')
#         model.eval()
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# def score(sentence):
#     tokenize_input = tokenizer.encode(sentence)
#     tensor_input = torch.tensor([tokenize_input], device=model.device)
#     loss=model(tensor_input, labels=tensor_input)[0]
#     return np.exp(loss.cpu().detach().numpy())

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

class NewSROIEScorer:
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

def main():
    cand_num = 5
    ori_scorer = SROIEScorer()
    new_scorer = SROIEScorer()

    test_dir = '/home/minghaoli/minghaoli/data/SROIE_Task2_Original/test'
    output_dir = 'temp'
    os.makedirs(output_dir, exist_ok=True)
    generate_txt_path = 'checkpoints/DeiTTR/FT_DA2_SROIE_FT_Receipt_SyntheticV2_DA2_LargeR_2MCDIP_BSZ16_LR5E_5_UP4_ITP_DeiT_LargeR_DA2_205K/generate-test.txt'

    with open(generate_txt_path, 'r', encoding='utf8') as fp:
        lines = list(fp.readlines())
    while not lines[0].startswith('T-0'):
        lines = lines[1:]

    log_fp = open('compare_original_with_upperbound.txt', 'w')

    total_mismatch = 0
    match_without_space = 0
    more_than_two_equal_without_space = 0
    
    _, data = SROIETask2(test_dir, None, None)
    bar = tqdm(data, desc='Caculating:')
    for t in bar:
        image_id = int(t['image_id'])

        gt_line_id = image_id * (1 + cand_num * 3)
        gt_line = lines[gt_line_id]
        assert gt_line.startswith('T-{:d}'.format(image_id))
        gt_str = gt_line[gt_line.find('\t') + 1:].rstrip()
        
        preds = {}
        ori_pred = None
        for cand_i in range(cand_num):
            pred_line_id = gt_line_id + 2 + cand_i * 3
            pred_line = lines[pred_line_id]
            assert pred_line.startswith('D-{:d}'.format(image_id))
            pred_line = pred_line[pred_line.find('\t') + 1:]
            pred_str = pred_line[pred_line.find('\t') + 1:].rstrip()
            pred_score = float(pred_line[:pred_line.find('\t')])
            pred_score = pow(2, pred_score)
            if pred_str in preds:
                preds[pred_str] += pred_score / (cand_i + 2)
            else:
                preds[pred_str] = pred_score / (cand_i + 2)

            if cand_i == 0:
                ori_scorer.add_string(gt_str, pred_str)
                ori_pred = pred_str

        best_f1 = 0
        best_pred = ori_pred

        repeated_list = {}
        for pred_str in preds:
            if pred_str.replace(' ', '') in repeated_list:
                repeated_list[pred_str.replace(' ', '')] += 1
            else:
                repeated_list[pred_str.replace(' ', '')] = 1
       
            temp_scorer = SROIEScorer()
            temp_scorer.add_string(gt_str, pred_str)
            try:
                _, _, f1 = temp_scorer.score()
            except ZeroDivisionError:
                continue
            if f1 > best_f1:
                best_f1 = f1
                best_pred = pred_str

        if repeated_list[ori_pred.replace(' ', '')] > 1:
            more_than_two_equal_without_space += 1

        pred_str = best_pred

        if ori_pred != pred_str:
            log_fp.write('{} {} {}\n'.format('ImageID:', image_id, '[SPACE Caused]' if ori_pred.replace(' ', '') == pred_str.replace(' ', '') else ''))
            log_fp.write('{} {}\n'.format('GT:', gt_str))
            log_fp.write('{} {} {}\n'.format('Original:', ori_pred, str(ori_pred == gt_str)))
            log_fp.write('{} {} {}\n\n'.format('New:', pred_str, str(pred_str == gt_str)))
            total_mismatch += 1
            match_without_space += int(ori_pred.replace(' ', '') == pred_str.replace(' ', ''))
        new_scorer.add_string(gt_str, pred_str)        

        # if pred_str != gt_str:
            # log_fp.write('GT: {}\tPred: {}\n'.format(gt_str, pred_str))

        bar.set_description(new_scorer.result_string())
        # for word in pred_str.split():
        #     output_fp.write(word + '\n')
    
    # if output_fp:
    #     output_fp.close()
    
    # if os.path.exists('predictions.zip'):
    #     print('Remove exist predictions.zip.')
    #     os.remove('predictions.zip')
    # zip_fp = zipfile.ZipFile('predictions.zip', 'w')
    # for txt_file in os.listdir(output_dir):
    #     zip_fp.write(os.path.join(output_dir, txt_file), txt_file)
    # zip_fp.close()
    # shutil.rmtree(output_dir)

    print('Original:', ori_scorer.result_string())
    print('New:', new_scorer.result_string())
    log_fp.write('{} {}\n'.format('Original:', ori_scorer.result_string()))
    log_fp.write('{} {}\n'.format('New:', new_scorer.result_string()))
    log_fp.write('{} {}\n'.format('Total Mismatch:', total_mismatch))
    log_fp.write('{} {}\n'.format('Match without space:', match_without_space))
    log_fp.write('{} {}\n'.format('More than two equal without space:', more_than_two_equal_without_space))
    log_fp.close()

if __name__ == '__main__':
    main()