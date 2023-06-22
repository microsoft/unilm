# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file provides the inference script.
# ------------------------------------------

import json
import os
import numpy as np
import argparse
from clipscore import cal_clipscore
from fid_score import calculate_fid_given_paths


def eval_clipscore(root_eval, root_res, dataset, device="cuda:0", num_images_per_prompt=4):
    with open(os.path.join(root_eval, dataset, dataset + '.txt'), 'r') as fr:
        text_list = fr.readlines()
        text_list = [_.strip() for _ in text_list]
    clip_scores = []
    scores = []
    for seed in range(num_images_per_prompt):
        if 'stablediffusion' in root_res:
            format = '.png'
        else:
            format = '.jpg'
        image_list = [os.path.join(root_res, dataset, 'images_' + str(seed),
                                   str(idx) + '_' +  str(seed) + format) for idx in range(len(text_list))]
        image_ids = [str(idx) + '_' +  str(seed) + format for idx in range(len(text_list))]
        score = cal_clipscore(image_ids=image_ids, image_paths=image_list, text_list=text_list, device=device)
        clip_score = np.mean([s['CLIPScore'] for s in score.values()])
        clip_scores.append(clip_score)
        scores.append(score)
    print("clip_score:", np.mean(clip_scores), clip_scores)
    return np.mean(clip_scores), scores


def MARIOEval_evaluate_results(root, datasets_with_images, datasets, methods, gpu,
                               eval_clipscore_flag=True, eval_fid_flag=True, num_images_per_prompt=4):
    root_eval = os.path.join(root, "MARIOEval")
    method_res = {}
    device = "cuda:" + str(gpu)
    for method_idx, method in enumerate(methods):
        if method_idx != gpu:  # running in different gpus simultaneously to save time
            continue
        print("\nmethod:", method)
        dataset_res = {}
        root_res = os.path.join(root, 'generation', method)
        for dataset in datasets:
            print("dataset:", dataset)
            dataset_res[dataset] = {}
            if eval_clipscore_flag:
                dataset_res[dataset]['clipscore'], dataset_res[dataset]['scores'] =\
                    eval_clipscore(root_eval, root_res, dataset, device, num_images_per_prompt)
            if eval_fid_flag and dataset in datasets_with_images:
                gt_path = os.path.join(root_eval, dataset, 'images')
                fids = []
                for idx in range(num_images_per_prompt):
                    gen_path = os.path.join(root_res, dataset, 'images_' + str(idx))
                    fids.append(calculate_fid_given_paths(paths=[gt_path, gen_path]))
                print("fid:", np.mean(fids), fids)
                dataset_res[dataset]['fid'] = np.mean(fids)

        if eval_clipscore_flag:
            method_clipscores = []
            for seed in range(num_images_per_prompt):
                clipscore_list = []
                for dataset in dataset_res.keys():
                    clipscore_list += [_['CLIPScore'] for _ in dataset_res[dataset]['scores'][seed].values()]
                method_clipscores.append(np.mean(clipscore_list))
            method_clipscore = np.mean(method_clipscores)
            dataset_res['clipscore'] = method_clipscore
        if eval_fid_flag:
            method_fids = []
            for idx in range(num_images_per_prompt):
                gt_paths = []
                gen_paths = []
                for dataset in dataset_res.keys():
                    if dataset in datasets_with_images:
                        gt_paths.append(os.path.join(root_eval, dataset, 'images'))
                        gen_paths.append(os.path.join(root_res, dataset, 'images_' + str(idx)))
                if len(gt_paths):
                    method_fids.append(calculate_fid_given_paths(paths=[gt_paths, gen_paths]))
            print("fid:", np.mean(method_fids), method_fids)
            method_fid = np.mean(method_fids)
            dataset_res['fid'] = method_fid

        method_res[method] = dataset_res
        with open(os.path.join(root_res, 'eval.json'), 'w') as fw:
            json.dump(dataset_res, fw)

    print(method_res)
    with open(os.path.join(root, 'generation', 'eval.json'), 'w') as fw:
        json.dump(method_res, fw)


def merge_eval_results(root, methods):
    method_res = {}
    for method_idx, method in enumerate(methods):
        root_res = os.path.join(root, 'generation', method)
        with open(os.path.join(root_res, 'eval.json'), 'r') as fr:
            dataset_res = json.load(fr)
            for k, v in dataset_res.items():
                if type(v) is dict:
                    del v['scores']  # too long
            method_res[method] = dataset_res

    with open(os.path.join(root, 'generation', 'eval.json'), 'w') as fw:
        json.dump(method_res, fw)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--dataset",
        type=str,
        default='TMDBEval500',
        required=False,
        choices=['TMDBEval500', 'OpenLibraryEval500', 'LAIONEval4000',
                 'ChineseDrawText', 'DrawBenchText', 'DrawTextCreative']
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/path/to/data/TextDiffuser/evaluation/",
        required=True,
    )
    parser.add_argument(
        "--method",
        type=str,
        default='controlnet',
        required=False,
        choices=['controlnet', 'deepfloyd', 'stablediffusion', 'textdiffuser']
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        required=False,
    )
    parser.add_argument(
        "--total_split",
        type=int,
        default=1,
        required=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    datasets_with_images = ['TMDBEval500', 'OpenLibraryEval500', 'LAIONEval4000']
    datasets = datasets_with_images + ['ChineseDrawText', 'DrawBenchText', 'DrawTextCreative']
    methods = ['textdiffuser', 'controlnet', 'deepfloyd', 'stablediffusion'] 

    MARIOEval_evaluate_results(args.root, datasets_with_images, datasets, methods, args.gpu,
                               eval_clipscore_flag=True, eval_fid_flag=True, num_images_per_prompt=4)
    merge_eval_results(args.root, methods)
