"""
this script will convert the original data format to tsv format, in which jpg files are converted to base64 strings and
saved in a resolution of 512(shorter side).
each contains 3 columns: input, edit, seed0_0, seed0_1, ..., seedN_0, seedN_1
each tsv file should contain 1000 samples.
"""

import argparse
import base64
import io
import json
import os
from multiprocessing import Process

from PIL import Image
from tqdm import tqdm


def save_tsv(args, i, sub_seeds_list):
    with open(os.path.join(args.output_dir, f'{str(i).zfill(4)}.tsv'), 'w') as f:
        for name, seeds in tqdm(sub_seeds_list, desc=f'processing {i}th tsv file', leave=False):
            # load prompt
            prompt = json.load(open(os.path.join(args.data_dir, name, 'prompt.json')))
            # load images
            images = [Image.open(os.path.join(args.data_dir, name, f'{seed}_{j}.jpg')).convert('RGB') for seed in seeds
                      for j in range(2)]
            # resize the shorter side to 512
            images = [im.resize((512, int(512 / im.size[0] * im.size[1])) if im.size[0] < im.size[1] else
                                (int(512 / im.size[1] * im.size[0]), 512)) for im in images]
            # encode image using base64
            for j, im in enumerate(images):
                buffer = io.BytesIO()
                im.save(buffer, format='PNG')
                images[j] = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # write to tsv file
            f.write('\t'.join([
                prompt['input'].replace('\t', '').replace('\n', '').strip(),
                prompt['edit'].replace('\t', '').replace('\n', '').strip(),
                *images
            ]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/path/to/clip-filtered-dataset/')
    parser.add_argument('--output-dir', type=str, default='/path/to/output-dir/')
    parser.add_argument('--max-image', type=int, default=1000)
    parser.add_argument('--num-process', type=int, default=64)

    args = parser.parse_args()

    # load seeds
    seeds_list = json.load(open(os.path.join(args.data_dir, 'seeds.json')))

    # split seeds into 1000 samples per tsv file
    seeds_list = [seeds_list[i:i + args.max_image] for i in range(0, len(seeds_list), args.max_image)]

    # save tsv files
    processes = []
    for i, sub_seeds_list in enumerate(seeds_list):
        p = Process(target=save_tsv, args=(args, i, sub_seeds_list))
        p.start()
        processes.append(p)
        if len(processes) == args.num_process:
            for p in processes:
                p.join()
            processes = []

    for p in processes:
        p.join()

    print('done.')
