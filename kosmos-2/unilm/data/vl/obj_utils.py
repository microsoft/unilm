import glob
import os
import numpy as np
import time
import json
import random
import itertools
import hydra
import copy
import ast

from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

import base64
import io


def process_grounding_data(self, 
                           item, 
                           ori_img_size, 
                           mode='phrase', 
                           mode_switch_prob=0.5, 
                           drop_crop_thr=0.,
                           perform_centercrop=True, # default is Ture as in pretraining
                           ):
    
    caption = item[1]
    obj_lists = ast.literal_eval(item[5])
    ori_img_w, ori_img_h = ori_img_size
    cluster_obj_dict = {}
    
    if mode == 'phrase':
        if isinstance(obj_lists, dict):
            obj_lists = obj_lists['phrase']
        for obj_list in obj_lists:
            phrase_start, phrase_end, x1_norm, y1_norm, x2_norm, y2_norm, score = obj_list
            if score < self.box_score_threshold:
                continue
            # phrase = caption[phrase_start:phrase_end]
            
            if perform_centercrop:
                # center-crop the box to align with the croped image after image transform
                croped_box = centercrop_norm_bbox((ori_img_w, ori_img_h), self.input_resolution, 
                                                    self.input_resolution, (x1_norm, y1_norm, x2_norm, y2_norm), 
                                                    drop_crop_thr=drop_crop_thr)
                if croped_box is None:
                    # the box is outside current image area
                    continue
                
                x1_norm, y1_norm, x2_norm, y2_norm = croped_box
            else:
                croped_box = [x1_norm, y1_norm, x2_norm, y2_norm]
            ul_idx, lr_idx = get_box_coords_index(self.quantized_size, np.array([x1_norm, y1_norm, x2_norm, y2_norm]))
            
            # if ul_idx == lr_idx:
            #     continue # filter the too small boxes
            
            if (phrase_start, phrase_end) in cluster_obj_dict.keys():
                cluster_obj_dict[(phrase_start, phrase_end)][0].append((ul_idx, lr_idx))
                cluster_obj_dict[(phrase_start, phrase_end)][1].append(score)
                cluster_obj_dict[(phrase_start, phrase_end)][2].append(croped_box)
            else:
                cluster_obj_dict[(phrase_start, phrase_end)] = [[(ul_idx, lr_idx)], [score], [croped_box]]
        return cluster_obj_dict
    
    elif mode == 'expression':
        obj_lists = obj_lists['expression_v1']
        
        filter_obj_lists = []
        # filter some boxes that do not meet the conditions
        for obj_list in obj_lists:
            phrase_start, phrase_end, x1_norm, y1_norm, x2_norm, y2_norm, score = obj_list
            if score < self.box_score_threshold:
                continue
            
            if perform_centercrop:
                # center-crop the box to align with the croped image after image transform
                croped_box = centercrop_norm_bbox((ori_img_w, ori_img_h), self.input_resolution, 
                                                    self.input_resolution, (x1_norm, y1_norm, x2_norm, y2_norm),
                                                    drop_crop_thr=drop_crop_thr)
                if croped_box is None:
                    # the box is outside current image area
                    continue
                
                x1_norm, y1_norm, x2_norm, y2_norm = croped_box
            else:
                croped_box = [x1_norm, y1_norm, x2_norm, y2_norm]
            filter_obj_lists.append([phrase_start, phrase_end, x1_norm, y1_norm, x2_norm, y2_norm, score])
        
        obj_lists = filter_obj_lists
        obj_child_lists = []
        for i, obj_list in enumerate(obj_lists):
            child_dict = {}
            phrase_start, phrase_end, x1_norm, y1_norm, x2_norm, y2_norm, score = obj_list
            child_dict['item'] = obj_list
            # child_dict['range'] = [phrase_start, phrase_end]
            child_dict['childs'] = []
            child_dict['roots'] = []
            for j, obj_list2 in enumerate(obj_lists): 
                phrase_start2, phrase_end2, _, _, _, _, _ = obj_list2
                
                # itself or multiple box for one phrase
                if phrase_start == phrase_start2 and phrase_end == phrase_end2:
                    continue
                # phrase1 contains phrase2
                elif phrase_start <= phrase_start2 and phrase_end >= phrase_end2:
                    child_dict['childs'].append(caption[phrase_start2:phrase_end2])
                # phrase2 contains phrase1
                elif phrase_start2 <= phrase_start and phrase_end2 >= phrase_end:
                    child_dict['roots'].append(caption[phrase_start2:phrase_end2])
                else:
                    continue
            obj_child_lists.append(child_dict)
        
        # we have multiple choices here because some phrase have both childs and roots
        
        # mode 1
        # we choice the largest range phrase that have childs but not be child of others
        filter_obj_lists = []
        for child_dict in obj_child_lists:
            if len(child_dict['roots']) == 0:
                filter_obj_lists.append(child_dict['item'])
        
        # turn it to patch-index format and return 
        for i, obj_list in enumerate(filter_obj_lists):
            phrase_start, phrase_end, x1_norm, y1_norm, x2_norm, y2_norm, score = obj_list
            ul_idx, lr_idx = get_box_coords_index(self.quantized_size, np.array([x1_norm, y1_norm, x2_norm, y2_norm]))
                
            # if ul_idx == lr_idx:
            #     continue # filter the too small boxes
            
            if (phrase_start, phrase_end) in cluster_obj_dict.keys():
                cluster_obj_dict[(phrase_start, phrase_end)][0].append((ul_idx, lr_idx))
                cluster_obj_dict[(phrase_start, phrase_end)][1].append(score)
                cluster_obj_dict[(phrase_start, phrase_end)][2].append([x1_norm, y1_norm, x2_norm, y2_norm])
            else:
                cluster_obj_dict[(phrase_start, phrase_end)] = [[(ul_idx, lr_idx),], [score,], [[x1_norm, y1_norm, x2_norm, y2_norm],]]
                
        return cluster_obj_dict
                    

def centercrop_norm_bbox(original_size, resize_size, crop_size, normalized_box, drop_crop_thr=0):
    ori_img_w, ori_img_h = original_size
    nx1, ny1, nx2, ny2 = normalized_box
    ox1, oy1, ox2, oy2 = nx1 * ori_img_w, ny1 * ori_img_h, nx2 * ori_img_w, ny2 * ori_img_h
    
    # process resize
    # calculate the resize image
    if (ori_img_w <= ori_img_h and ori_img_w == resize_size) or (ori_img_h <= ori_img_w and ori_img_h == resize_size):
        resize_w, resize_h = ori_img_w, ori_img_h
    elif ori_img_w < ori_img_h:
        resize_w = resize_size
        resize_h = int(resize_size * ori_img_h / ori_img_w)
    else:
        resize_h = resize_size
        resize_w = int(resize_size * ori_img_w / ori_img_h)
    rx1, ry1, rx2, ry2 = nx1 * resize_w, ny1 * resize_h, nx2 * resize_w, ny2 * resize_h
    
    crop_top = (resize_h - crop_size) / 2.
    crop_left = (resize_w - crop_size) / 2.
    
    # re-normalized using original size and crop it
    max_size = torch.as_tensor([crop_size, crop_size], dtype=torch.float32)
    boxes = torch.as_tensor([rx1, ry1, rx2, ry2], dtype=torch.float32)
    cropped_boxes = boxes - torch.as_tensor([crop_left, crop_top, crop_left, crop_top])
    cropped_boxes = torch.min(cropped_boxes.reshape(2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    
    # calculate original box area  
    original_area = (rx2 - rx1) * (ry2 - ry1)
    cropped_boxes_tmp = cropped_boxes.reshape(4).tolist()
    cropped_boxes_area = (cropped_boxes_tmp[3] - cropped_boxes_tmp[1]) * (cropped_boxes_tmp[2] - cropped_boxes_tmp[0])
    if cropped_boxes_area / original_area < drop_crop_thr:
        return None

    # normalized using current size and calculate the area size
    cropped_boxes /= max_size
    if torch.all(cropped_boxes[1, :] > cropped_boxes[0, :]):
        return cropped_boxes.reshape(4).tolist()
    else:
        return None     

def visualize_normed_img_with_bbox(image_np, cluster_obj_dict, bin_size, caption, name):
    """
    Args:
        image_np (_type_): np.array
        cluster_obj_dict (_type_): {(1, 10): [[(0, 1024)], [0.6991432]]}
        bin_size (_type_): 32
        caption
    """
    # (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
    norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
    to_pil = T.ToPILImage()
    
    # pdb.set_trace()
    image_tensor = torch.from_numpy(image_np)
    image_tensor = image_tensor * norm_std + norm_mean
    # image_tensor = image_tensor[[2, 1, 0], ...] * 255
    image = to_pil(image_tensor)
    image.save(os.path.join('output', 'debug', f"{name}.jpg"))
    
    # visualize original box after crop
    tgt = {}
    W, H = image.size
    tgt['size'] = (W, H)
    tgt['boxes'] = []
    tgt['labels'] = []
    for k, v in cluster_obj_dict.items():
        label = caption[k[0]:k[1]]
        for _v in v[2]: # original box
            tgt['labels'].append(label)            
            tgt['boxes'].append(torch.as_tensor(_v))
    print(f"\n draw {tgt} on image {name}")
    draw_image = plot_boxes_to_image(image.copy(), tgt)
    draw_image.save(os.path.join('output', 'debug', f"{name}_with_orig_box.jpg"))
    
    # visualize quantized box
    tgt = {}
    W, H = image.size
    tgt['size'] = (W, H)
    tgt['boxes'] = []
    tgt['labels'] = []
    for k, v in cluster_obj_dict.items():
        label = caption[k[0]:k[1]]
        for _v in v[0]:
            tgt['labels'].append(label)            
            ul_idx, lr_idx = _v
            box = get_box_coords_from_index(bin_size, ul_idx, lr_idx)
            tgt['boxes'].append(torch.from_numpy(box))
    print(f"\n draw {tgt} on image {name}")
    draw_image = plot_boxes_to_image(image.copy(), tgt)
    draw_image.save(os.path.join('output', 'debug', f"{name}_with_quan_box.jpg"))
    
    
def plot_boxes_to_image(image_pil, tgt):
    
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)
        print(f"Add {label}-{box.tolist()}")
    return image_pil
   
def get_box_coords_index(P, box_coords):
    """
    
    Given a grid of length P and the coordinates of a bounding box, returns the indices of the grid cells that
    correspond to the upper-left and lower-right corners of the bounding box.
    
    Args:
    - P (int): the length of the grid
    - box_coords (np.array of shape (4,)): the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2]
    
    Returns:
    - ul_idx (int): the index of the grid cell that corresponds to the upper-left corner of the bounding box
    - lr_idx (int): the index of the grid cell that corresponds to the lower-right corner of the bounding box
    """
    # pdb.set_trace()
    # Compute the size of each cell in the grid
    cell_size = 1.0 / P
    
    # Compute the indices of the grid cells that correspond to the upper-left and lower-right corners of the bounding box
    ul_x = int(np.floor(max(box_coords[0], 0) / cell_size))
    ul_y = int(np.floor(max(box_coords[1], 0) / cell_size))
    ul_idx = ul_x + ul_y * P
    
    lr_x = int(np.floor(min(box_coords[2], 0.99999) / cell_size))
    lr_y = int(np.floor(min(box_coords[3], 0.99999) / cell_size))
    lr_idx = lr_x + lr_y * P
    
    return ul_idx, lr_idx

def get_box_coords_from_index(P, ul_idx, lr_idx):  
    """  
    Given a grid of length P and the indices of the upper-left and lower-right corners of a bounding box,  
    returns the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2].  
      
    Args:  
    - P (int): the length of the grid  
    - ul_idx (int): the index of the grid cell that corresponds to the upper-left corner of the bounding box  
    - lr_idx (int): the index of the grid cell that corresponds to the lower-right corner of the bounding box  
      
    Returns:  
    - box_coords (np.array of shape (4,)): the normalized coordinates of the bounding box, in the form [x1, y1, x2, y2]  
    """  
    # Compute the size of each cell in the grid  
    cell_size = 1.0 / P  
      
    # Compute the x and y indices of the upper-left and lower-right corners of the bounding box  
    ul_x = ul_idx % P  
    ul_y = ul_idx // P  
      
    lr_x = lr_idx % P  
    lr_y = lr_idx // P  
      
    # Compute the normalized coordinates of the bounding box  
    if ul_idx == lr_idx:  
        x1 = ul_x * cell_size  
        y1 = ul_y * cell_size  
        x2 = lr_x * cell_size + cell_size  
        y2 = lr_y * cell_size + cell_size  
    elif ul_x == lr_x or ul_y == lr_y:  
        x1 = ul_x * cell_size  
        y1 = ul_y * cell_size  
        x2 = lr_x * cell_size + cell_size  
        y2 = lr_y * cell_size + cell_size  
    else:  
        x1 = ul_x * cell_size + cell_size / 2  
        y1 = ul_y * cell_size + cell_size / 2  
        x2 = lr_x * cell_size + cell_size / 2  
        y2 = lr_y * cell_size + cell_size / 2  
      
    return np.array([x1, y1, x2, y2])
