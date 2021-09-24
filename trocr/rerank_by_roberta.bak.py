import os
from data import SROIETask2
from tqdm import tqdm
import shutil
import zipfile
import torch
import numpy as np
import pickle 

from transformers import AutoTokenizer, RobertaForMaskedLM
import torch
from scipy.special import softmax

max_input_len = 512
device = 'cuda'
model = RobertaForMaskedLM.from_pretrained('roberta-base').to(device)

def score(sentence_ids, mask, cands_ids):
    sentence_ids = sentence_ids.unsqueeze(0)
    with torch.no_grad():
        outputs = model(sentence_ids)
    logits = outputs.logits[0][mask].detach().cpu().numpy()
    probs = softmax(logits, axis=1)
    cand_scores = []
    for cand_ids in cands_ids:
        cand_score = 1
        for i, token_id in enumerate(cand_ids):
            cand_score *= probs[i][token_id]
        cand_scores.append(cand_score)

    ref_ids = np.argmax(probs, axis=1)

    return cand_scores, ref_ids

def mask_sentence(preds, sample_id, tokenizer):    

    pre_context = [tokenizer.bos_token_id]
    for i in range(0, sample_id):
        pred_str = preds[i]['preds'][0][0]
        pred_ids = tokenizer.encode(pred_str, add_special_tokens=False) + [tokenizer.sep_token_id]
        pre_context.extend(pred_ids)

    suf_context = []
    for i in range(sample_id + 1, len(preds)):
        pred_str = preds[i]['preds'][0][0]
        pred_ids = tokenizer.encode(pred_str, add_special_tokens=False) + [tokenizer.sep_token_id]
        suf_context.extend(pred_ids)

    max_len = 0
    cands_ids = []
    for candidate in preds[sample_id]['preds']:
        candidate_str = candidate[0]
        candidate_ids = tokenizer.encode(candidate_str, add_special_tokens=False)
        cands_ids.append(candidate_ids)
        if len(candidate_ids) > max_len:
            max_len = len(candidate_ids)
    
    sentence_ids = pre_context + [tokenizer.mask_token_id] * max_len + [tokenizer.sep_token_id] + suf_context
    sentence_ids = np.array(sentence_ids)
    mask = sentence_ids == tokenizer.mask_token_id
    return sentence_ids, mask, cands_ids

# def main():
#     cand_num = 5
#
#     test_dir = '/home/minghaoli/minghaoli/data/SROIE_Task2_Original/test'
#     generate_txt_path = 'checkpoints/DeiTTR/SROIE_300Epochs_Large_DeiT_DIY_FT_FT_Receipt53K_on_CDIP5000/generate-test-best5.txt'
#
#     if os.path.exists('temp.pkl'):
#         filename_preds = pickle.load(open('temp.pkl', 'rb'))
#     else:
#         with open(generate_txt_path, 'r', encoding='utf8') as fp:
#             lines = list(fp.readlines())
#         while not lines[0].startswith('T-0'):
#             lines = lines[1:]
#
#         _, data = SROIETask2(test_dir, None, None)
#         filename_preds = {}
#         bar = tqdm(data, desc='Processing:')
#         for t in bar:
#             file_name = t['file_name']
#             image_id = int(t['image_id'])
#
#             gt_line_id = image_id * (1 + cand_num * 3)
#             gt_line = lines[gt_line_id]
#             assert gt_line.startswith('T-{:d}'.format(image_id))
#             gt_str = gt_line[gt_line.find('\t') + 1:].rstrip()
#
#             preds = {'gt':gt_str, 'preds':[]}
#             for cand_i in range(cand_num):
#                 pred_line_id = gt_line_id + 2 + cand_i * 3
#                 pred_line = lines[pred_line_id]
#                 assert pred_line.startswith('D-{:d}'.format(image_id))
#                 pred_line = pred_line[pred_line.find('\t') + 1:]
#                 pred_str = pred_line[pred_line.find('\t') + 1:].rstrip()
#                 pred_score = float(pred_line[:pred_line.find('\t')])
#                 pred_score = pow(2, pred_score)
#                 preds['preds'].append((pred_str, pred_score))
#
#             if file_name in filename_preds:
#                 filename_preds[file_name].append(preds)
#             else:
#                 filename_preds[file_name] = [preds]
#         pickle.dump(filename_preds, open('temp.pkl', 'wb'))
#
#     tokenizer = AutoTokenizer.from_pretrained('roberta-base')
#     for filename in filename_preds:
#         for sample_id in range(len(filename_preds[filename])):
#             sentence_ids, mask, cands_ids = mask_sentence(filename_preds[filename], sample_id, tokenizer)
#             sentence_ids = torch.tensor(sentence_ids).to(device)
#             mask = torch.tensor(mask).to(device)
#             roberta_scores, ref_ids = score(sentence_ids, mask, cands_ids)
#             print(tokenizer.decode(ref_ids))
#             print(filename_preds[filename][sample_id]['gt'])
#             print()
#
#         break
#
# if __name__ == '__main__':
#     main()