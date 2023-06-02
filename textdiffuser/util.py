# ------------------------------------------
# TextDiffuser: Diffusion Models as Text Painters
# Paper Link: https://arxiv.org/abs/2305.10855
# Code Link: https://github.com/microsoft/unilm/tree/master/textdiffuser
# Copyright (c) Microsoft Corporation.
# This file defines a set of commonly used utility functions.
# ------------------------------------------

import os
import re
import cv2
import math
import shutil
import string
import textwrap
import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps

from typing import *

# define alphabet and alphabet_dic
alphabet = string.digits + string.ascii_lowercase + string.ascii_uppercase + string.punctuation + ' ' # len(aphabet) = 95
alphabet_dic = {}
for index, c in enumerate(alphabet):
    alphabet_dic[c] = index + 1 # the index 0 stands for non-character
    


def transform_mask_pil(mask_root):
    """
    This function extracts the mask area and text area from the images.
    
    Args:
        mask_root (str): The path of mask image.
            * The white area is the unmasked area
            * The gray area is the masked area
            * The white area is the text area
    """
    img = np.array(mask_root)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY) # pixel value is set to 0 or 255 according to the threshold
    return 1 - (binary.astype(np.float32) / 255) 
    

def transform_mask(mask_root: str):
    """
    This function extracts the mask area and text area from the images.
    
    Args:
        mask_root (str): The path of mask image.
            * The white area is the unmasked area
            * The gray area is the masked area
            * The white area is the text area
    """
    img = cv2.imread(mask_root)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY) # pixel value is set to 0 or 255 according to the threshold
    return 1 - (binary.astype(np.float32) / 255) 


def segmentation_mask_visualization(font_path: str, segmentation_mask: np.array):
    """
    This function visualizes the segmentaiton masks with characters.
    
    Args:
        font_path (str): The path of font. We recommand to use Arial.ttf
        segmentation_mask (np.array): The character-level segmentation mask.
    """
    segmentation_mask = cv2.resize(segmentation_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
    font = ImageFont.truetype(font_path, 8)
    blank = Image.new('RGB', (512,512), (0,0,0))
    d = ImageDraw.Draw(blank)
    for i in range(64):
        for j in range(64):
            if int(segmentation_mask[i][j]) == 0 or int(segmentation_mask[i][j])-1 >= len(alphabet): 
                continue
            else:
                d.text((j*8, i*8), alphabet[int(segmentation_mask[i][j])-1], font=font, fill=(0, 255, 0))
    return blank


def make_caption_pil(font_path: str, captions: List[str]):
    """
    This function converts captions into pil images.
    
    Args:
        font_path (str): The path of font. We recommand to use Arial.ttf
        captions (List[str]): List of captions.
    """
    caption_pil_list = []
    font = ImageFont.truetype(font_path, 18)

    for caption in captions:
        border_size = 2
        img = Image.new('RGB', (512-4,48-4), (255,255,255)) 
        img = ImageOps.expand(img, border=(border_size, border_size, border_size, border_size), fill=(127, 127, 127))
        draw = ImageDraw.Draw(img)
        border_size = 2
        text = caption
        lines = textwrap.wrap(text, width=40)
        x, y = 4, 4
        line_height = font.getsize('A')[1] + 4 

        start = 0
        for line in lines:
            draw.text((x, y+start), line, font=font, fill=(200, 127, 0))
            y += line_height

        caption_pil_list.append(img)
    return caption_pil_list


def filter_segmentation_mask(segmentation_mask: np.array):
    """
    This function removes some noisy predictions of segmentation masks.
    
    Args:
        segmentation_mask (np.array): The character-level segmentation mask.
    """
    segmentation_mask[segmentation_mask==alphabet_dic['-']] = 0
    segmentation_mask[segmentation_mask==alphabet_dic[' ']] = 0
    return segmentation_mask
    
    

def combine_image(args, sub_output_dir: str, pred_image_list: List, image_pil: Image, character_mask_pil: Image, character_mask_highlight_pil: Image, caption_pil_list: List):
    """
    This function combines all the outputs and useful inputs together.
    
    Args:
        args (argparse.ArgumentParser): The arguments.
        pred_image_list (List): List of predicted images.
        image_pil (Image): The original image.
        character_mask_pil (Image): The character-level segmentation mask.
        character_mask_highlight_pil (Image): The character-level segmentation mask highlighting character regions with green color.
        caption_pil_list (List): List of captions.
    """
    
    # # create a "latest" folder to store the results 
    # if os.path.exists(f'{args.output_dir}/latest'):
    #     shutil.rmtree(f'{args.output_dir}/latest')
    # os.mkdir(f'{args.output_dir}/latest')
    
    # save each predicted image
    # os.makedirs(f'{args.output_dir}/{sub_output_dir}', exist_ok=True)
    for index, img in enumerate(pred_image_list):
        img.save(f'{args.output_dir}/{sub_output_dir}/{index}.jpg')
        # img.save(f'{args.output_dir}/latest/{index}.jpg')
        
    length = len(pred_image_list)
    lines = math.ceil(length / 3)
    
    blank = Image.new('RGB', (512*3, 512*(lines+1)+48*lines), (0,0,0)) 
    blank.paste(image_pil,(0,0))
    blank.paste(character_mask_pil,(512,0))
    blank.paste(character_mask_highlight_pil,(512*2,0))
    
    for i in range(length):
        row, col = i // 3, i % 3
        blank.paste(pred_image_list[i],(512*col,512*(row+1)+48*row))
        blank.paste(caption_pil_list[i],(512*col,512*(row+1)+48*row+512))
    
    blank.save(f'{args.output_dir}/{sub_output_dir}/combine.jpg')
    # blank.save(f'{args.output_dir}/latest/combine.jpg')
    
    return blank.convert('RGB')
    
    
def combine_image_gradio(args, sub_output_dir: str, pred_image_list: List, image_pil: Image, character_mask_pil: Image, character_mask_highlight_pil: Image, caption_pil_list: List):
    """
    This function combines all the outputs and useful inputs together.
    
    Args:
        args (argparse.ArgumentParser): The arguments.
        pred_image_list (List): List of predicted images.
        image_pil (Image): The original image.
        character_mask_pil (Image): The character-level segmentation mask.
        character_mask_highlight_pil (Image): The character-level segmentation mask highlighting character regions with green color.
        caption_pil_list (List): List of captions.
    """
    
    size = len(pred_image_list)
    
    if size == 1:
        return pred_image_list[0]
    elif size == 2:
        blank = Image.new('RGB', (512*2, 512), (0,0,0))
        blank.paste(pred_image_list[0],(0,0))
        blank.paste(pred_image_list[1],(512,0))
    elif size == 3:
        blank = Image.new('RGB', (512*3, 512), (0,0,0))
        blank.paste(pred_image_list[0],(0,0))
        blank.paste(pred_image_list[1],(512,0))
        blank.paste(pred_image_list[2],(1024,0))
    elif size == 4:
        blank = Image.new('RGB', (512*2, 512*2), (0,0,0))
        blank.paste(pred_image_list[0],(0,0))
        blank.paste(pred_image_list[1],(512,0))
        blank.paste(pred_image_list[2],(0,512))
        blank.paste(pred_image_list[3],(512,512))

    
    return blank
    
def get_width(font_path, text):
    """
    This function calculates the width of the text.
    
    Args:
        font_path (str): user prompt.
        text (str): user prompt.
    """
    font = ImageFont.truetype(font_path, 24)
    width, _ = font.getsize(text)
    return width



def get_key_words(text: str):
    """
    This function detect keywords (enclosed by quotes) from user prompts. The keywords are used to guide the layout generation.
    
    Args:
        text (str): user prompt.
    """

    words = []
    text = text
    matches = re.findall(r"'(.*?)'", text) # find the keywords enclosed by ''
    if matches:
        for match in matches:
            words.extend(match.split())
            
    if len(words) >= 8:
        return []
   
    return words


def adjust_overlap_box(box_output, current_index):
    """
    This function adjust the overlapping boxes.
    
    Args:
        box_output (List): List of predicted boxes.
        current_index (int): the index of current box.
    """
    
    if current_index == 0:
        return box_output
    else:
        # judge whether it contains overlap with the last output
        last_box = box_output[0, current_index-1, :]
        xmin_last, ymin_last, xmax_last, ymax_last = last_box
        
        current_box = box_output[0, current_index, :]
        xmin, ymin, xmax, ymax = current_box
        
        if xmin_last <= xmin <= xmax_last and ymin_last <= ymin <= ymax_last:
            print('adjust overlapping')
            distance_x = xmax_last - xmin
            distance_y = ymax_last - ymin
            if distance_x <= distance_y:
                # avoid overlap
                new_x_min = xmax_last + 0.025
                new_x_max = xmax - xmin + xmax_last + 0.025
                box_output[0,current_index,0] = new_x_min
                box_output[0,current_index,2] = new_x_max
            else:
                new_y_min = ymax_last + 0.025
                new_y_max = ymax - ymin + ymax_last + 0.025
                box_output[0,current_index,1] = new_y_min
                box_output[0,current_index,3] = new_y_max  
                
        elif xmin_last <= xmin <= xmax_last and ymin_last <= ymax <= ymax_last:
            print('adjust overlapping')
            new_x_min = xmax_last + 0.05
            new_x_max = xmax - xmin + xmax_last + 0.05
            box_output[0,current_index,0] = new_x_min
            box_output[0,current_index,2] = new_x_max
                    
        return box_output
    
    
def shrink_box(box, scale_factor = 0.9):
    """
    This function shrinks the box.
    
    Args:
        box (List): List of predicted boxes.
        scale_factor (float): The scale factor of shrinking.
    """
    
    x1, y1, x2, y2 = box
    x1_new = x1 + (x2 - x1) * (1 - scale_factor) / 2
    y1_new = y1 + (y2 - y1) * (1 - scale_factor) / 2
    x2_new = x2 - (x2 - x1) * (1 - scale_factor) / 2
    y2_new = y2 - (y2 - y1) * (1 - scale_factor) / 2
    return (x1_new, y1_new, x2_new, y2_new)


def adjust_font_size(args, width, height, draw, text):
    """
    This function adjusts the font size.
    
    Args:
        args (argparse.ArgumentParser): The arguments.
        width (int): The width of the text.
        height (int): The height of the text.
        draw (ImageDraw): The ImageDraw object.
        text (str): The text.
    """
    
    size_start = height
    while True:
        font = ImageFont.truetype(args.font_path, size_start)
        text_width, _ = draw.textsize(text, font=font)
        if text_width >= width:
            size_start = size_start - 1
        else:
            return size_start
    
    
def inpainting_merge_image(original_image, mask_image, inpainting_image):
    """
    This function merges the original image, mask image and inpainting image.
        
    Args:
        original_image (PIL.Image): The original image.
        mask_image (PIL.Image): The mask images.
        inpainting_image (PIL.Image): The inpainting images.
    """
    
    original_image = original_image.resize((512, 512))
    mask_image = mask_image.resize((512, 512))
    inpainting_image = inpainting_image.resize((512, 512))
    mask_image.convert('L')
    threshold = 250 
    table = []
    for i in range(256):
        if i < threshold:
            table.append(1)
        else:
            table.append(0)
    mask_image = mask_image.point(table, "1")
    merged_image = Image.composite(inpainting_image, original_image, mask_image)
    return merged_image
