# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import os
import sys
import argparse

import torch
from torch import nn
from torchvision import transforms as pth_transforms
from timm.models import create_model

from PIL import Image

import utils
import modeling_vqkd 

def get_code(args):
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # Normalize in pre-process of vqkd
    ])
    print(f"Image transforms: {transform}")

    images = transform(Image.open(args.img_path)).unsqueeze(0)

    # ============ building network ... ============
    model = create_model(
            args.model,
            pretrained=True,
            pretrained_weight=args.pretrained_weights,
            as_tokenzer=True,
        ).eval()

    input_ids = model.get_codebook_indices(images)
    print(input_ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get code for VQ-KD')
    parser.add_argument('--model', default='vqkd_encoder_base_decoder_1x768x12_clip', type=str, help="model")
    parser.add_argument('--pretrained_weights', 
                        default='https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_1x768x12_clip-d93179da.pth', 
                        type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--img_path', default='demo/ILSVRC2012_val_00031649.JPEG', type=str, help="image path.")
    args = parser.parse_args()

    get_code(args)
    
    
# tensor([[3812, 7466, 1913, 1913, 1903, 1913, 1903, 1913, 3812, 7820, 6337, 2189,
#          7466, 7466, 2492, 3743, 5268, 3481, 5268, 4987,  445, 8009, 3501, 5268,
#          7820, 7831, 4816, 2189, 7549, 7549, 5548, 4987,  445, 4198,  445, 5216,
#          4987, 5268, 3278, 5203, 6337, 1799,  847, 6454, 4527, 5302, 8009, 3743,
#          5216, 4678, 3743, 4858, 5203, 4816, 7831, 2189, 7549, 5386, 6628, 5004,
#          2779, 7131, 7131, 7131, 4928, 3743,  119,  445, 1903, 7466, 4527, 5386,
#          5398, 5704, 2104, 5398, 2779, 7258, 7989,  624, 7131, 1186, 5216, 7466,
#          8015, 5004,  452, 7243, 3145, 6690, 7017, 2104, 5398, 4198, 7989, 7131,
#          3717, 7466,  580, 5004, 5004, 6202, 6202, 6202, 1826, 7521, 1473, 5722,
#          2486, 5663, 4928, 3941,  580, 5548, 7983, 7983, 7983, 2104, 5004, 2063,
#          2637, 1822, 3100, 3100, 1405, 1637, 8187, 5433, 2779, 5398, 5004, 5004,
#          1107, 3469, 3469, 5302, 2590, 6381, 3100, 4194, 3717,  356, 7131, 7688,
#          5104, 3081, 3812, 3950, 1186, 7131, 7131, 3717, 4399, 1186, 2221, 6501,
#          7131, 5433, 3014, 3950, 3278, 2812, 7131, 1186, 7036, 6947, 7036, 4648,
#          2812, 7131, 3014, 5295, 7266, 5180, 4123, 3792, 4648, 8009, 4648, 4816,
#          1511, 7036,  375, 2221, 5813, 5698,  168, 7131, 3792, 5698, 5698, 2667,
#          5698, 4648, 4171, 6501]], device='cuda:0')