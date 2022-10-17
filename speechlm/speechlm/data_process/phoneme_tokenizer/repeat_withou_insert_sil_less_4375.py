# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import sys, json, tqdm
import numpy as np

input_file = sys.argv[1]
mean_and_std_file = sys.argv[2]
out_file = sys.argv[3]

mean_and_std = json.load(open(mean_and_std_file, 'r'))

with open(input_file, 'r') as f, open(out_file, 'w') as w:
    for line in tqdm.tqdm(f):
        l = line.split()

        new_l = []
        for phn in l:
            if phn not in mean_and_std:
               mean_and_std[phn] = [5, 2.5]
               print(f'unk phone {phn}')
            n = max(1, round(np.random.normal(loc=mean_and_std[phn][0], scale=mean_and_std[phn][1])))
            new_l.extend([phn] * int(n))

        minus = 0
        while len(new_l) >= 4375:
            minus += 1
            new_l = []
            for phn in l:
                n = max(1, round(mean_and_std[phn][0] - minus))
                new_l.extend([phn] * n)
            print(f"too long line try minus {minus}")

        w.write(' '.join(new_l)+'\n')

