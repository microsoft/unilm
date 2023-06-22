# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import os
import re
import copy

gts = {
    'ChineseDrawText': [],
    'DrawBenchText': [],
    'DrawTextCreative': [],
    'LAIONEval4000': [],
    'OpenLibraryEval500': [],
    'TMDBEval500': [],
}

results = {
    'stablediffusion': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    'textdiffuser': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    'controlnet': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
    'deepfloyd': {'cnt':0, 'p':0, 'r':0, 'f':0, 'acc':0},
}

def get_key_words(text: str):
    words = []
    text = text
    matches = re.findall(r"'(.*?)'", text) # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())
   
    return words


# load gt
files = os.listdir('/path/to/MARIOEval')
for file in files:
    lines = open(os.path.join('/path/to/MARIOEval', file, f'{file}.txt')).readlines()
    for line in lines:
        line = line.strip().lower()
        gts[file].append(get_key_words(line))
print(gts['ChineseDrawText'][:10])


def get_p_r_acc(method, pred, gt):

    pred = [p.strip().lower() for p in pred] 
    gt = [g.strip().lower() for g in gt]

    pred_orig = copy.deepcopy(pred)
    gt_orig = copy.deepcopy(gt)

    pred_length = len(pred)
    gt_length = len(gt)

    for p in pred:
        if p in gt_orig:
            pred_orig.remove(p) 
            gt_orig.remove(p)

    p = (pred_length - len(pred_orig)) / (pred_length + 1e-8)
    r = (gt_length - len(gt_orig)) / (gt_length + 1e-8)
   
    pred_sorted = sorted(pred)
    gt_sorted = sorted(gt)
    if ''.join(pred_sorted) == ''.join(gt_sorted):
        acc = 1
    else:
        acc = 0

    return p, r, acc


files = os.listdir('/path/to/MaskTextSpotterV3/tools/ocr_result')
print(len(files))

for file in files:
    method, dataset, prompt_index, image_index = file.strip().split('_')
    ocrs = open(os.path.join('/path/to/MaskTextSpotterV3/tools/ocr_result', file)).readlines()
    p, r, acc = get_p_r_acc(method, ocrs, gts[dataset][int(prompt_index)])
    results[method]['cnt'] += 1
    results[method]['p'] += p
    results[method]['r'] += r
    results[method]['acc'] += acc

for method in results.keys():
    results[method]['p'] /= results[method]['cnt']
    results[method]['r'] /= results[method]['cnt']
    results[method]['f'] = 2 * results[method]['p'] * results[method]['r'] / (results[method]['p'] + results[method]['r'] + 1e-8)
    results[method]['acc'] /= results[method]['cnt']
    
print(results)

