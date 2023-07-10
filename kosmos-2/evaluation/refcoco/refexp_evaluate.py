import copy
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.data
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from prettytable import PrettyTable

import re
import json

from box_ops import generalized_box_iou, box_iou
from decode_string import decode_bbox_from_caption

import pdb

class RefExpEvaluatorFromTxt(object):
    def __init__(self, refexp_gt_path, k=(1, -1), thresh_iou=0.5):
        assert isinstance(k, (list, tuple))
        with open(refexp_gt_path, 'r') as f:
            self.refexp_gt = json.load(f)
        self.img_ids = [item['id'] for item in self.refexp_gt['images']]
        print(f"Load {len(self.img_ids)} images")
        print(f"Load {len(self.refexp_gt['annotations'])} annotations")
        self.k = k
        self.thresh_iou = thresh_iou

    def summarize(self,
                  prediction_file: str,
                  quantized_size: int = 32,
                  verbose: bool = False,):
        
        # get the predictions
        with open(prediction_file, 'r', encoding='utf-8') as f:
            predict_all_lines = f.readlines()
        # filter the invaild lines for predict_all_lines
        filter_prediction_lines = []
        for line in predict_all_lines:
            line_pieces = line.strip('\n').split('\t')
            if 'H-' in line_pieces[0]:
                if line_pieces[0].split('-')[-1].isdigit():
                    filter_prediction_lines.append(line)
        
        predict_all_lines = filter_prediction_lines
        predict_index = 0
        
        dataset2score = {
            "refcoco": {k: 0.0 for k in self.k},
            "refcoco+": {k: 0.0 for k in self.k},
            "refcocog": {k: 0.0 for k in self.k},
        }
        dataset2count = {"refcoco": 0.0, "refcoco+": 0.0, "refcocog": 0.0}
        for item_img, item_ann in tqdm(zip(self.refexp_gt['images'], self.refexp_gt['annotations'])):
            # quit when evaluating all predictions
            if predict_index == len(predict_all_lines):
                    break
                
            if item_img['id'] != item_ann['image_id']:
                raise ValueError(f"Ann\n{item_ann} \nis not matched\n {item_img}")
            
            dataset_name = item_img['dataset_name']
            img_height = item_img['height']
            img_width = item_img['width']
            caption = item_img['caption']
            target_bbox = item_ann["bbox"]
            converted_bbox = [
                target_bbox[0],
                target_bbox[1],
                target_bbox[2] + target_bbox[0],
                target_bbox[3] + target_bbox[1],
            ]
            target_bbox = torch.as_tensor(converted_bbox).view(-1, 4)
            
            
            prediction_line = predict_all_lines[predict_index].split('</image>')[-1]
            predict_index += 1
            
            collect_entity_location = decode_bbox_from_caption(prediction_line, quantized_size=quantized_size, verbose=verbose)
            
            predict_boxes = []
            for (p_pred, p_x1, p_y1, p_x2, p_y2) in collect_entity_location:
                if p_pred.strip() != caption.strip():
                    continue
                else:
                    pred_box = [p_x1 * img_width, p_y1 * img_height, p_x2 * img_width, p_y2 * img_height]
                    predict_boxes.append(pred_box)
                    
            if len(predict_boxes) == 0:
                print(f"Can't find valid bbox for the given phrase {caption}, \n{collect_entity_location}")
                print(f"We set a 0-area box to calculate result")
                predict_boxes = [[0., 0., 0., 0.]]
                
            predict_boxes = torch.as_tensor(predict_boxes).view(-1, 4)
            
            iou, _ = box_iou(predict_boxes, target_bbox)
            mean_iou, _ = box_iou(predict_boxes.mean(0).view(-1, 4), target_bbox)
            for k in self.k:
                if k == 'upper bound':
                    if max(iou) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0
                elif k == 'mean':
                    if max(mean_iou) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0
                else:
                    if max(iou[0, :k]) >= self.thresh_iou:
                        dataset2score[dataset_name][k] += 1.0

            dataset2count[dataset_name] += 1.0

        for key, value in dataset2score.items():
            for k in self.k:
                try:
                    value[k] /= dataset2count[key]
                except:
                    pass
                
        results = {}
        for key, value in dataset2score.items():
            results[key] = sorted([v for k, v in value.items()])
            print(f" Dataset: {key} - Precision @ 1, mean, all: {results[key]} \n")

        return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_file', help='prediction_file')
    parser.add_argument('annotation_file', default='/path/to/mdetr_processed_json_annotations', help='annotation_file')
    parser.add_argument('--quantized_size', default=32, type=int)
    
    args = parser.parse_args()
    
    evaluator = RefExpEvaluatorFromTxt(
        refexp_gt_path=args.annotation_file, 
        k=(1, 'mean', 'upper bound'), 
        thresh_iou=0.5,
    )
    
    evaluator.summarize(args.prediction_file, args.quantized_size, verbose=False)
