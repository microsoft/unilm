import argparse
import glob
import logging
import os
import io
import h5py
import pickle
import sys
import random


import numpy as np
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

def Receipt53K(gt_path, bpe_parser):
    root_dir = os.path.dirname(gt_path)
    data = []
    img_id = 0
    with open(gt_path, 'r', encoding='utf8') as fp:
        for line in tqdm(list(fp.readlines()), desc='Loading Receipt53K:'):
            line = line.rstrip()
            temp = line.split('\t', 1)
            img_file = temp[0]
            text = temp[1]

            img_path = os.path.join(root_dir, img_file)  
            encoded_str = bpe_parser.encode(text)      

            data.append({'img_path': img_path, 'image_id':img_id, 'text':text, 'encoded_str':encoded_str})
            img_id += 1

    return data

    
class Receipt53KDataset(FairseqDataset):
    def __init__(self, gt_path, tfm, bpe_parser, target_dict):
        self.gt_path = gt_path
        self.tfm = tfm            
        self.target_dict = target_dict
        # self.bpe_parser = bpe_parser
        self.data = Receipt53K(gt_path, bpe_parser)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]
        
        image = Image.open(img_dict['img_path']).convert('RGB')
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)   # h, w, c
        return {'id': idx, 'tfm_img':tfm_img, 'label_ids':input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples)


class HDF5Dataset(FairseqDataset):
    def __init__(self, root_dir, tfm, bpe_parser, target_dict):
        self.root_dir = root_dir
        self.tfm = tfm            
        self.target_dict = target_dict
        self.bpe_parser = bpe_parser
        split = os.path.basename(root_dir)
        dirname = os.path.dirname(root_dir)
        logger.info('Loading index file: {}'.format('index_{}.pkl'.format(split)))
        self.indexes = pickle.load(open(os.path.join(dirname, 'index_{}.pkl'.format(split)), 'rb'))
        logger.info('Finished loading index file: {}'.format('index_{}.pkl'.format(split)))
        self.sorted_index_key = sorted(self.indexes.keys())
        self.length = sum(self.indexes.values())

    def __len__(self):
        return self.length

    def filter_indices_by_size(self, indices, max_sizes):
        return indices, []

    def index_file(self, idx):
        t = idx
        for basename in self.sorted_index_key:
            num = self.indexes[basename]
            if t < num:                
                return basename + '.hdf5', t 
            else:
                t -= num
        basename = random.choice(self.sorted_index_key)
        num = self.indexes[basename]
        t = random.choice(range(num))
        return basename, t

    def getitem(self, idx):        
        hdf5_file, idx = self.index_file(idx)
        with h5py.File(os.path.join(self.root_dir, hdf5_file), 'r') as hdf5_fp:
            filename = sorted(hdf5_fp.keys())[idx]
            hdf5_content = hdf5_fp[filename][()].tobytes()
            hdf5_content = pickle.loads(hdf5_content)

            image = Image.open(io.BytesIO(hdf5_content['image_bin'])).convert('RGB')
            text = hdf5_content['label']

            if len(text) > 100:            
                return None
        
        encoded_str = self.bpe_parser.encode(text)            
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)   # h, w, c
        return {'id': idx, 'tfm_img':tfm_img, 'label_ids':input_ids}

    def __getitem__(self, idx):
        try:
            return self.getitem(idx)
        except Exception as e:
            logger.warning('{:d} has error to load: {}'.format(idx, str(e)))
            return None

    def size(self, idx):
        return 100
        hdf5_file, idx = self.index_file(idx)
        with h5py.File(os.path.join(self.root_dir, hdf5_file), 'r') as hdf5_fp:
            filename = sorted(hdf5_fp.keys())[idx]
            hdf5_content = hdf5_fp[filename][()].tobytes()
            hdf5_content = pickle.loads(hdf5_content)
            
            text = hdf5_content['label']
        return len(text)        

    def num_tokens(self, idx):      
        return 100  
        return self.size(idx)


    def collater(self, samples):
        return default_collater(self.target_dict, samples, self)


class MultiHDF5Dataset(FairseqDataset):
    def __init__(self, root_dir, tfm, bpe_parser, target_dict):
        self.root_dir = root_dir
        self.tfm = tfm            
        self.target_dict = target_dict
        self.bpe_parser = bpe_parser
        first_1m = HDF5Dataset(root_dir, tfm, bpe_parser, target_dict)
        second_1m = HDF5Dataset(root_dir.replace('IIT_CDIP_CROP_HDF5', 'IIT_CDIP_CROP_HDF5_2'), tfm, bpe_parser, target_dict)
        logger.info('Join two datasets: IIT_CDIP_CROP_HDF5 and IIT_CDIP_CROP_HDF5_2.')
        self.datasets = [first_1m, second_1m]

    def __len__(self):        
        return sum([len(t) for t in self.datasets])

    def filter_indices_by_size(self, indices, max_sizes):
        return indices, []

    def __getitem__(self, idx):
        ori_idx = idx
        for dataset in self.datasets:
            if idx >= len(dataset):
                idx -= len(dataset)
            else:
                try:
                    return dataset[idx]
                except:
                    logger.warning('Cannot get the {:d} item. Info: {}'.format(ori_idx, str([len(t) for t in self.datasets])))
                    return None

    def size(self, idx):
        return 100

    def num_tokens(self, idx):      
        return 100  

    def collater(self, samples):
        return default_collater(self.target_dict, samples, self)

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

class SyntheticTextRecognitionDataset(Receipt53KDataset):
    def __init__(self, gt_path, tfm, bpe_parser, target_dict):
        self.gt_path = gt_path
        self.tfm = tfm            
        self.target_dict = target_dict 
        self.data = STR(gt_path, bpe_parser)   


class IIITHWSDataset(FairseqDataset):
    
    def load_annotations(self, bpe_parser):
    # <img1-path><space><text1-string><space><dummyInt><space><train/test flag>
        train_set = []
        test_set = []
        with open(self.gt_path, 'r', encoding='utf8') as fp:            
            for line in tqdm(fp.readlines(), desc='Loading GT'):
                line = line.split()
                im_path, text, dummyint, iftest = line
                dummyint = int(dummyint)
                iftest = bool(int(iftest))
                sample = {
                    'im_path': im_path,
                    'text': text,
                    'encoded_str': bpe_parser.encode(text) if bpe_parser else [],
                    'dummyint': dummyint,
                    'iftest': iftest
                }
                if iftest:
                    test_set.append(sample)
                else:
                    train_set.append(sample)            
        return train_set, test_set

    def __init__(self, root_dir, split, tfm, bpe_parser, target_dict):
        self.gt_path = os.path.join(root_dir, 'IIIT-HWS-90K.txt')
        self.im_dir = os.path.join(root_dir, 'Images_90K_Normalized')
        self.tfm = tfm            
        self.target_dict = target_dict
        # self.bpe_parser = bpe_parser
        self.train_data, self.test_data = self.load_annotations(bpe_parser)        
        if split == 'train':
            self.data = self.train_data
        elif split == 'test':
            self.data = self.test_data[:1000]
        elif split == 'valid':
            self.data = self.test_data[1000:2000]
        else:
            raise NotImplementedError            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_dict = self.data[idx]

        im_path = os.path.join(self.im_dir, img_dict['im_path'])
        
        image = Image.open(im_path).convert('RGB')
        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)

        tfm_img = self.tfm(image)   # h, w, c
        return {'id': idx, 'tfm_img':tfm_img, 'label_ids':input_ids}

    def size(self, idx):
        img_dict = self.data[idx]

        encoded_str = img_dict['encoded_str']
        input_ids = self.target_dict.encode_line(encoded_str, add_if_not_exist=False)
        return len(input_ids)

    def num_tokens(self, idx):
        return self.size(idx)

    def collater(self, samples):
        return default_collater(self.target_dict, samples) 

class GeneralDataset(FairseqDataset):
    def __init__(self, desc_path, tfm, bpe_parser, target_dict):
        self.datasets = []
        with open(desc_path, 'r') as fp:
            for line in fp:
                line = line.split()
                data_type = line[0]
                data_path = line[1]

                if data_type == 'SROIE':
                    self.datasets.append(SROIETextRecognitionDataset(data_path, tfm, bpe_parser, target_dict))
                elif data_type == 'STR':
                    self.datasets.append(SyntheticTextRecognitionDataset(data_path, tfm, bpe_parser, target_dict))
                elif data_type == 'Receipt':
                    self.datasets.append(Receipt53KDataset(data_path, tfm, bpe_parser, target_dict))
                else:
                    raise Exception('Not Defined.')
        self.tfm = tfm
        self.bpe_parset = bpe_parser
        self.target_dict = target_dict

    def __len__(self):        
        return sum([len(t) for t in self.datasets])

    def __getitem__(self, idx):
        ori_idx = idx
        for dataset in self.datasets:
            if idx >= len(dataset):
                idx -= len(dataset)
            else:
                try:
                    return dataset[idx]
                except:
                    logger.warning('Cannot get the {:d} item. Info: {}'.format(ori_idx, str([len(t) for t in self.datasets])))
                    return None

    def size(self, idx):
        ori_idx = idx
        for dataset in self.datasets:
            if idx >= len(dataset):
                idx -= len(dataset)
            else:
                try:
                    return dataset.size(idx)
                except:
                    logger.warning('Cannot get the size of {:d}. Info: {}'.format(ori_idx, str([len(t) for t in self.datasets])))
                    return None

    def num_tokens(self, idx):      
        ori_idx = idx
        for dataset in self.datasets:
            if idx >= len(dataset):
                idx -= len(dataset)
            else:
                try:
                    return dataset.num_tokens(idx)
                except:
                    logger.warning('Cannot get the size of {:d}. Info: {}'.format(ori_idx, str([len(t) for t in self.datasets])))
                    return None 

    def collater(self, samples):
        return default_collater(self.target_dict, samples, self)


if __name__ == '__main__':
    import torchvision.transforms as transforms
    from fairseq.data import Dictionary
    from fairseq.data.encoders import build_bpe

    from deit_datasets import build_transform
    from torchvision.transforms.functional import InterpolationMode

    from data_aug import ResizeNormalize, build_data_aug, KeepOriginal
    import matplotlib.pyplot as plt


    parser = argparse.ArgumentParser()
    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                            "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--input-size', type=int, default=384, help='images input size')

    args = parser.parse_args()

    logging.basicConfig()

    gt_path = 'data/SyntheticTextRecognition/train_gt.txt'
    tfm = build_data_aug((384, 384), mode='train')
    # tfm = KeepOriginal()
    bpe_parser = build_bpe('gpt2')
    target_dict = Dictionary.load('data/SROIE_Task2_Original/gpt2.dict.txt')

    dataset = SyntheticTextRecognition(gt_path, tfm, bpe_parser, target_dict)
    # rates = [t['tfm_img'].size[0]/t['tfm_img'].size[1] for t in dataset]
    # plt.hist(rates)
    # plt.savefig('temp.jpg')
    print()
    # for i in tqdm(range(1000)):
    #     dataset.__getitem__(i)['tfm_img'].save('temp/{:d}.jpg'.format(i))