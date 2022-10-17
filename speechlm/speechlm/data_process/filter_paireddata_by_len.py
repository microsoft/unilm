# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import os
import argparse
from tqdm import tqdm
import numpy as np


lg_label = "__label__{}"

def writefile(filename, lines):
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    parser.add_argument("--src", "-s", required=True, type=str)
    parser.add_argument("--tgt", "-t", required=True, type=str)
    parser.add_argument("--max-len", "-m", default=2998, type=int)
    args = parser.parse_args()
    
    src_lines, tgt_lines = [], []
    with open(f"{args.input}.{args.src}", 'r') as f1, open(f"{args.input}.{args.tgt}", 'r') as f2: 
        for src_line, tgt_line in tqdm(zip(f1, f2)):
            src_len = len(src_line.strip().split())
            tgt_len = len(tgt_line.strip().split())
            if src_len < args.max_len and src_len > 0 and tgt_len < args.max_len and tgt_len > 0:
                src_lines.append(src_line)
                tgt_lines.append(tgt_line)

    writefile(f"{args.output}.{args.src}", src_lines)
    writefile(f"{args.output}.{args.tgt}", tgt_lines)

if __name__ == "__main__":
    main()



