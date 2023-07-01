import json  
import os  
import requests  
from urllib.parse import urlparse  
from requests.exceptions import HTTPError  

import sys
from pathlib import Path
import textwrap

import ast
import os
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 20, 12

import cv2
import base64
import io

def download_images_from_jsonl(jsonl_path, output_folder):  
    with open(jsonl_path, 'r') as jsonl_file:  
        for line in jsonl_file:  
            json_obj = json.loads(line)  
            url = json_obj['url']  
            # download_image(url, output_folder)
            vis_image(json_obj, output_folder)  
  
def download_image(url, output_folder):  
    try:  
        response = requests.get(url)  
        response.raise_for_status()  
    except HTTPError as e:  
        print(f"Error while downloading {url}: {e}")  
        return  
  
    file_name = os.path.basename(urlparse(url).path)  
    output_path = os.path.join(output_folder, file_name)  
  
    with open(output_path, 'wb') as file:  
        file.write(response.content)  
        
def imshow(img, file_name = "tmp.jpg", caption='test'):
    # Create figure and axis objects
    fig, ax = plt.subplots()
    # Show image on axis
    ax.imshow(img[:, :, [2, 1, 0]])
    ax.set_axis_off()
    # Set caption text
    # Add caption below image
    ax.text(0.5, -0.2, '\n'.join(textwrap.wrap(caption, 120)), ha='center', transform=ax.transAxes, fontsize=18)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
    
def vis_image(json_obj, output_folder): 
    url = json_obj['url']  
    try:  
        response = requests.get(url)  
        response.raise_for_status()  
        
        file_name = os.path.basename(urlparse(url).path)  
        # output_path = os.path.join(output_folder, file_name) 
        file_key_name = json_obj['key'] + os.path.splitext(file_name)[1]
        output_path = os.path.join(output_folder, file_key_name) 
        
    except Exception as e:  
        print(f"Error while downloading {url}: {e}")  
        return    

    with open(output_path, 'wb') as file:  
        file.write(response.content) 
    
    try:
        pil_img = Image.open(output_path).convert("RGB")
    except:
        return 
    image = np.array(pil_img)[:, :, [2, 1, 0]]
    image_h = pil_img.height
    image_w = pil_img.width
    caption = json_obj['caption']
    
    def is_overlapping(rect1, rect2):  
        x1, y1, x2, y2 = rect1  
        x3, y3, x4, y4 = rect2  
        return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4) 
    
    grounding_list = json_obj['ref_exps']
    new_image = image.copy()
    previous_locations = []
    previous_bboxes = []
    text_offset = 10
    text_offset_original = 4
    text_size = max(0.07 * min(image_h, image_w) / 100, 0.5)
    text_line = int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = int(max(2 * min(image_h, image_w) / 512, 2))
    text_height = text_offset # init
    # pdb.set_trace()
    for (phrase_s, phrase_e, x1_norm, y1_norm, x2_norm, y2_norm, score) in grounding_list:  
        phrase = caption[phrase_s:phrase_e]
        x1, y1, x2, y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
        print(f"Decode results: {phrase} - {[x1, y1, x2, y2]}")
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
    
    try:
        file_key_name = json_obj['key'] + '_exp' + os.path.splitext(file_name)[1]
        output_path = os.path.join(output_folder, file_key_name) 
        
        imshow(new_image, file_name= output_path, caption=caption)
    except:
        # Out of (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp)
        return 

    
if __name__ == '__main__':  
    # you need to download the jsonl before run this file    
    jsonl_path = '/tmp/grit_coyo.jsonl'
    output_folder = './output/vis_grit'
    
    if not os.path.exists(output_folder):  
        os.makedirs(output_folder)  
  
    download_images_from_jsonl(jsonl_path, output_folder)  
