from fairseq.data import Dictionary
from data import SROIETextRecognitionDataset, Receipt53KDataset
import argparse
from fairseq.data.encoders import build_bpe
import os
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-type', type=str,
        help='the data type to create the dictionary (SROIE|Receipt)'
    )
    parser.add_argument(
        '--bpe', type=str,
        help='the bpe type to create the dictionary (SROIE|Receipt)'
    )
    parser.add_argument(
        'data_dir', type=str,
        help='the data dir'
    )
    args = parser.parse_args()

    tfm = None
    bpe = build_bpe(args)

    target_dict = Dictionary()
    datasets = {}

    for split in ['train', 'valid', 'test']:
        if args.data_type == 'SROIE':
            root_dir = os.path.join(args.data_dir, split)
            datasets[split] = SROIETextRecognitionDataset(root_dir, tfm, bpe, target_dict, None).data      
        elif args.data_type == 'Receipt53K':
            gt_path = os.path.join(args.data_dir, 'gt_{}.txt'.format(split))            
            datasets[split] = Receipt53KDataset(gt_path, tfm, bpe, target_dict).data
        for img_dict in tqdm(datasets[split], desc='Encoding:'):
            encoded_str = img_dict['encoded_str']
            input_ids = target_dict.encode_line(encoded_str)
        print('| [label] load dictionary: {} types'.format(len(target_dict)))

    target_dict.save(os.path.join(args.data_dir, 'dict.label.txt'))
    