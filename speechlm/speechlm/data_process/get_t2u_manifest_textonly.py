# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
import numpy as np
from examples.speech_to_text.data_utils import save_df_to_tsv


log = logging.getLogger(__name__)

def get_duration(fa_phone):
    """fa_phone: force-aligned phone, 1-D numpy"""
    same = np.concatenate(([True], fa_phone[:-1] != fa_phone[1:], [True]))
    index = np.where(same)[0]
    count = np.diff(index)
    return count

def process(args):
    # assert "train" in args.splits
    out_root = Path(args.output_root).absolute()
    out_root.mkdir(exist_ok=True)

    print("Fetching data...")
    audio_manifest_root = Path(args.audio_manifest_root).absolute()
    for s in args.splits:
        manifest = defaultdict(list)
        with open(audio_manifest_root / f"{s}.phn") as f1:
            for i, reduced_phone in tqdm(enumerate(f1)):
                reduced_phone = reduced_phone.strip()
                uttid = f"librilm-{i}"
                speaker = uttid.split("-")[0]
                
                manifest["id"].append(uttid)
                manifest["speaker"].append(speaker)
                manifest["n_frames"].append(len(reduced_phone))
                manifest["tgt_text"].append(reduced_phone)
                manifest["unit"].append(0)
        save_df_to_tsv(
            pd.DataFrame.from_dict(manifest),
            out_root / f"{s}.tsv"
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-manifest-root", "-m", type=str)
    parser.add_argument("--output-root", "-o", required=True, type=str)
    parser.add_argument("--splits", "-s", type=str, nargs="+",
                        default=["train", "dev", "test"])
    parser.add_argument("--add-fastspeech-targets", action="store_true")
    args = parser.parse_args()

    process(args)

if __name__ == "__main__":
    main()
