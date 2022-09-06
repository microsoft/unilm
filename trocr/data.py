import glob
import logging
import os
import random

import torch
from fairseq.data import FairseqDataset, data_utils
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def default_collater(target_dict, samples, dataset=None):
    if not samples:
        return None
    if any([sample is None for sample in samples]):
        if not dataset:
            return None
        len_batch = len(samples)        
        while True:
            samples.append(dataset[random.choice(range(len(dataset)))])
            samples =list(filter (lambda x:x is not None, samples))
            if len(samples) == len_batch:
                break        
    indices = []

    imgs = [] # bs, c, h , w
    target_samples = []
    target_ntokens = 0

    for sample in samples:
        index = sample['id']
        indices.append(index)

        
        imgs.append(sample['tfm_img'])
        
        target_samples.append(sample['label_ids'].long())
        target_ntokens += len(sample['label_ids'])

    num_sentences = len(samples)

    target_batch = data_utils.collate_tokens(target_samples,
                                            pad_idx=target_dict.pad(),
                                            eos_idx=target_dict.eos(),
                                            move_eos_to_beginning=False)
    rotate_batch = data_utils.collate_tokens(target_samples,
                                            pad_idx=target_dict.pad(),
                                            eos_idx=target_dict.eos(),
                                            move_eos_to_beginning=True)                                               

    indices = torch.tensor(indices, dtype=torch.long)
    imgs = torch.stack(imgs, dim=0)

    return {
        'id': indices,
        'net_input': {
            'imgs': imgs,
            'prev_output_tokens': rotate_batch
        },
        'ntokens': target_ntokens,
        'nsentences': num_sentences,            
        'target': target_batch
    }

def read_txt_and_tokenize(txt_path: str, bpe, target_dict):
    annotations = []
    with open(txt_path, 'r', encoding='utf8') as fp:
        for line in fp.readlines():
            line = line.rstrip()
            if not line:
                continue
            line_split = line.split(',', maxsplit=8)
            quadrangle = list(map(int, line_split[:8]))
            content = line_split[-1]

            if bpe:
                encoded_str = bpe.encode(content)
            else:
                encoded_str = content

            xs = [quadrangle[i] for i in range(0, 8, 2)]
            ys = [quadrangle[i] for i in range(1, 8, 2)]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            annotations.append({'bbox': bbox, 'encoded_str': encoded_str, 'category_id': 0, 'segmentation': [quadrangle]})  # 0 for text, 1 for background

    return annotations

def SROIETask2(root_dir: str, bpe, target_dict, crop_img_output_dir=None):
    data = []
    img_id = -1

    crop_data = []
    crop_img_id = -1

    image_paths = natsorted(list(glob.glob(os.path.join(root_dir, '*.jpg'))))
    for jpg_path in tqdm(image_paths):
        im = Image.open(jpg_path).convert('RGB')
        
        img_w, img_h = im.size
        img_id += 1

        txt_path = jpg_path.replace('.jpg', '.txt')
        annotations = read_txt_and_tokenize(txt_path, bpe, target_dict) 
        img_dict = {'file_name': jpg_path, 'width': img_w, 'height': img_h, 'image_id':img_id, 'annotations':annotations}
        data.append(img_dict)

        for ann in annotations:
            crop_w = ann['bbox'][2] - ann['bbox'][0]
            crop_h = ann['bbox'][3] - ann['bbox'][1]

            if not (crop_w > 0 and crop_h > 0):
                logger.warning('Error occurs during image cropping: {} has a zero area bbox.'.format(os.path.basename(jpg_path)))
                continue
            crop_img_id += 1
            crop_im = im.crop(ann['bbox'])
            if crop_img_output_dir:
                crop_im.save(os.path.join(crop_img_output_dir, '{:d}.jpg'.format(crop_img_id)))
            crop_img_dict = {'img':crop_im, 'file_name': jpg_path, 'width': crop_w, 'height': crop_h, 'image_id':crop_img_id, 'encoded_str':ann['encoded_str']}
            crop_data.append(crop_img_dict)

    return data, crop_data

class SROIETextRecognitionDataset(FairseqDataset):
    def __init__(self, root_dir, tfm, bpe_parser, target_dict, crop_img_output_dir=None):
        self.root_dir = root_dir
        self.tfm = tfm            
        self.target_dict = target_dict
        # self.bpe_parser = bpe_parser
        self.ori_data, self.data = SROIETask2(root_dir, bpe_parser, target_dict, crop_img_output_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]
        
        image = img_dict['img']
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)   # h, w, c
        return {'id': idx, 'tfm_img': tfm_img, 'label_ids': input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)


    def collater(self, samples):
        return default_collater(self.target_dict, samples)

def STR(gt_path, bpe_parser):
    root_dir = os.path.dirname(gt_path)
    data = []
    img_id = 0
    with open(gt_path, 'r') as fp:
        for line in tqdm(list(fp.readlines()), desc='Loading STR:'):
            line = line.rstrip()
            temp = line.split('\t', 1)
            img_file = temp[0]
            text = temp[1]

            img_path = os.path.join(root_dir, 'image', img_file)  
            if not bpe_parser:
                encoded_str = text
            else:
                encoded_str = bpe_parser.encode(text)      

            data.append({'img_path': img_path, 'image_id':img_id, 'text':text, 'encoded_str':encoded_str})
            img_id += 1

    return data


class SyntheticTextRecognitionDataset(FairseqDataset):
    def __init__(self, gt_path, tfm, bpe_parser, target_dict):
        self.gt_path = gt_path
        self.tfm = tfm
        self.target_dict = target_dict
        self.data = STR(gt_path, bpe_parser)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]

        image = Image.open(img_dict['img_path']).convert('RGB')
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)  # h, w, c
        return {'id': idx, 'tfm_img': tfm_img, 'label_ids': input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples)

def Receipt53K(gt_path):
    root_dir = os.path.dirname(gt_path)
    data = []
    with open(gt_path, 'r', encoding='utf8') as fp:
        for line in tqdm(list(fp.readlines()), desc='Loading Receipt53K:'):
            line = line.rstrip()
            temp = line.split('\t', 1)
            img_file = temp[0]
            text = temp[1]

            img_path = os.path.join(root_dir, img_file)  
            data.append({'img_path': img_path, 'text':text})

    return data

    
class Receipt53KDataset(FairseqDataset):
    def __init__(self, gt_path, tfm, bpe_parser, target_dict):
        self.gt_path = gt_path
        self.tfm = tfm            
        self.target_dict = target_dict
        self.bpe_parser = bpe_parser
        self.data = Receipt53K(gt_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]
        
        try:
            image = Image.open(img_dict['img_path']).convert('RGB')
        except Exception as e:
            logger.warning('Failed to load image: {}, since {}'.format(img_dict['img_path'], str(e)))
            return None    
        encoded_str = self.bpe_parser.encode(img_dict['text'])
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)   # h, w, c
        return {'id': idx, 'tfm_img':tfm_img, 'label_ids':input_ids}

    def size(self, idx):
        img_dict = self.data[idx]
        return len(img_dict['text'])
        # item = self[idx]
        # return len(item['label_ids'])

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples)