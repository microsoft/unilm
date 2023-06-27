import os
import sys
from pathlib import Path
import textwrap
import re

import ast
import os
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 20, 12

import cv2
import base64
import io

from decode_string import decode_bbox_from_caption

EOD_SYMBOL = "</doc>"
BOI_SYMBOL = "<image>"
EOI_SYMBOL = "</image>"
EOC_SYMBOL = "</chunk>"
EOL_SYMBOL = "</line>"

BOP_SYMBOL="<phrase>"
EOP_SYMBOL="</phrase>"
BOO_SYMBOL="<object>"
EOO_SYMBOL="</object>"
DOM_SYMBOL="</delimiter_of_multi_objects/>"

SPECIAL_SYMBOLS = [EOD_SYMBOL, BOI_SYMBOL, EOI_SYMBOL, EOC_SYMBOL, EOL_SYMBOL]

def add_location_symbols(quantized_size):
    custom_sp_symbols = []
    for symbol in SPECIAL_SYMBOLS:
        custom_sp_symbols.append(symbol)
    for symbol in [BOP_SYMBOL, EOP_SYMBOL, BOO_SYMBOL, EOO_SYMBOL, DOM_SYMBOL]:
        custom_sp_symbols.append(symbol)
    for i in range(quantized_size ** 2):
        token_name = f"<patch_index_{str(i).zfill(4)}>"
        custom_sp_symbols.append(token_name)
    return custom_sp_symbols

def imshow(img, file_name = "tmp.jpg", caption='test'):
    # Create figure and axis objects
    fig, ax = plt.subplots()
    # Show image on axis
    ax.imshow(img[:, :, [2, 1, 0]])
    ax.set_axis_off()
    # Set caption text
    # Add caption below image
    # ax.text(0.5, -0.1, caption, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.1, '\n'.join(textwrap.wrap(caption, 120)), ha='center', transform=ax.transAxes, fontsize=18)
    plt.savefig(file_name)
    plt.close()

def is_overlapping(rect1, rect2):  
    x1, y1, x2, y2 = rect1  
    x3, y3, x4, y4 = rect2  
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4) 

def draw_entity_box_on_image(image, collect_entity_location):  
    """_summary_

    Args:
        image (_type_): image or image path
        collect_entity_location (_type_): _description_
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        # pdb.set_trace()
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")
    
    if len(collect_entity_location) == 0:
        return image

    new_image = image.copy()
    previous_locations = []
    previous_bboxes = []
    text_offset = 10
    text_offset_original = 4
    text_size = max(0.07 * min(image_h, image_w) / 100, 0.5)
    text_line = int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = int(max(2 * min(image_h, image_w) / 512, 2))
    text_height = text_offset # init
    for (phrase, x1_norm, y1_norm, x2_norm, y2_norm) in collect_entity_location:  
        x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
        # draw bbox
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        new_image = cv2.rectangle(new_image, (x1, y1), (x2, y2), color, box_line)
        
        # add phrase name  
        # decide the text location first  
        for x_prev, y_prev in previous_locations:  
            if abs(x1 - x_prev) < abs(text_offset) and abs(y1 - y_prev) < abs(text_offset):  
                y1 += text_height  
  
        if y1 < 2 * text_offset:  
            y1 += text_offset + text_offset_original  

        # add text background
        (text_width, text_height), _ = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_line)  
        text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - text_height - text_offset_original, x1 + text_width, y1  
        
        for prev_bbox in previous_bboxes:  
            while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):  
                text_bg_y1 += text_offset  
                text_bg_y2 += text_offset  
                y1 += text_offset 
                
                if text_bg_y2 >= image_h:  
                    text_bg_y1 = max(0, image_h - text_height - text_offset_original)  
                    text_bg_y2 = image_h  
                    y1 = max(0, image_h - text_height - text_offset_original + text_offset)  
                    break 
        
        alpha = 0.5  
        for i in range(text_bg_y1, text_bg_y2):  
            for j in range(text_bg_x1, text_bg_x2):  
                if i < image_h and j < image_w: 
                    new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(color)).astype(np.uint8) 
        
        cv2.putText(  
            new_image, phrase, (x1, y1 - text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA  
        )  
        previous_locations.append((x1, y1))  
        previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))
  
    return new_image  


def visualize_results_on_image(img_path, caption, quantized_size=16, save_path=f"show_box_on_image.jpg", show=True):
    # collect_entity_location = decode_phrase_with_bbox_from_caption(caption, quantized_size=quantized_size)
    collect_entity_location = decode_bbox_from_caption(caption, quantized_size=quantized_size)
    image = draw_entity_box_on_image(img_path, collect_entity_location)
    if show:
        imshow(image, file_name=save_path, caption=caption)
    else:
        # return a PIL Image
        image = image[:, :, [2, 1, 0]]
        pil_image = Image.fromarray(image)  
        return pil_image
    
if __name__ == "__main__":

    
    caption = "a wet suit is at <object><patch_index_0003><patch_index_0004></delimiter_of_multi_objects/><patch_index_0005><patch_index_0006></object> in the picture" 
    print(decode_bbox_from_caption(caption))
