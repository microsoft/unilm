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
import torchaudio
from tqdm import tqdm
import numpy as np
import torch

from fairseq.data.audio.audio_utils import convert_waveform
from examples.speech_to_text.data_utils import save_df_to_tsv
from examples.speech_synthesis.data_utils import extract_pitch


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
        if args.add_pitch:
            pitch_root = out_root / "pitch" / s
            pitch_root.mkdir(exist_ok=True)
        manifest = defaultdict(list)
        with open(audio_manifest_root / f"{s}.audio.tsv") as f1, \
            open(audio_manifest_root / f"{s}.phn") as f2, \
            open(audio_manifest_root / f"{s}.km") as f3:
            audio_root = f1.readline().strip()
            audio_root = Path(audio_root)
            for audio_path, fa_phone, fa_unit in tqdm(zip(f1, f2, f3)):
                record = True
                audio_path, n_frames = audio_path.strip().split("\t")
                fa_phone = fa_phone.strip().split()
                fa_unit = fa_unit.strip()
                uttid = audio_path.split("/")[-1].split(".")[0]
                speaker = uttid.split("-")[0]
                
                if args.add_duration:
                    assert len(fa_phone) == len(fa_unit.split())
                    fa_phone = np.array(list(map(int, fa_phone)))
                    duration = get_duration(fa_phone)
                    reduced_phone = torch.LongTensor(fa_phone).unique_consecutive().numpy()
                    if args.add_pitch:
                        pitch_path = pitch_root / f"{uttid}.npy"
                        if not pitch_path.is_file():
                            waveform, sample_rate = torchaudio.load(audio_root / audio_path)
                            waveform, sample_rate = convert_waveform(
                                waveform, sample_rate, normalize_volume=args.normalize_volume,
                            )
                            pitch = extract_pitch(
                                waveform, sample_rate, None,
                                hop_length=args.hop_length, log_scale=True,
                                phoneme_durations=duration
                            )
                            if pitch is not None:
                                np.save(pitch_path.as_posix(), pitch)
                            else:
                                record = False
                else:
                    reduced_phone = fa_phone

                if record:
                    manifest["id"].append(uttid)
                    manifest["speaker"].append(speaker)
                    manifest["n_frames"].append(len(fa_unit.split()))
                    manifest["tgt_text"].append(" ".join(map(str, reduced_phone)))
                    manifest["unit"].append(fa_unit)
                    if args.add_duration:
                        manifest["duration"].append(" ".join(map(str, duration)))
                        if args.add_pitch:
                            manifest["pitch"].append(f"pitch/{s}/{uttid}.npy")
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
    parser.add_argument("--normalize-volume", "-n", action="store_true")
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--add-duration", action="store_true")
    parser.add_argument("--add-pitch", action="store_true")
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
