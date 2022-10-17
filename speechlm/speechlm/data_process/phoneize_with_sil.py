# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

"""
Modified from https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py
"""

import argparse
import numpy as np
import sys
from g2p_en import G2p
from tqdm import tqdm
import logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(
        description="converts words to phones adding optional silences around in between words"
    )
    parser.add_argument(
        "--sil-prob",
        "-s",
        type=float,
        default=0,
        help="probability of inserting silence between each word",
    )
    parser.add_argument(
        "--surround",
        action="store_true",
        help="if set, surrounds each example with silence",
    )
    parser.add_argument(
        "--lexicon",
        help="lexicon to convert to phones",
        required=True,
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="if set, OOV words will raise a error (for train/valid set)",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="input text file",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="input text file",
        required=True,
    )


    return parser


def normalize_phn(phons):
    """
    convert g2p style phone to 39-phone set
    """
    return [p.rstrip('0123456789') for p in phons]


def main():
    parser = get_parser()
    args = parser.parse_args()

    sil_prob = args.sil_prob
    surround = args.surround
    sil = "<SIL>"

    wrd_to_phn = {}
    g2p = G2p()

    with open(args.lexicon, "r") as lf:
        for line in lf:
            items = line.rstrip().split()
            assert len(items) > 1, line
            assert items[0] not in wrd_to_phn, items
            wrd_to_phn[items[0]] = items[1:]

    with open(args.input, "r") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in tqdm(fin):
            words = line.strip().upper().split()

            if not all(w in wrd_to_phn for w in words):
                if args.strict:
                    # logger.warning(f"| Warning: OOV words found: {line}")
                    pass
                else:
                    continue

            phones = []
            if surround:
                phones.append(sil)

            sample_sil_probs = None
            if sil_prob > 0 and len(words) > 1:
                sample_sil_probs = np.random.random(len(words) - 1)

            for i, w in enumerate(words):
                if w in wrd_to_phn:
                    phones.extend(wrd_to_phn[w])
                else:
                    phones.extend(normalize_phn(g2p(w)))
                if (
                    sample_sil_probs is not None
                    and i < len(sample_sil_probs)
                    and sample_sil_probs[i] < sil_prob
                ):
                    phones.append(sil)

            if surround:
                phones.append(sil)
            print(" ".join(phones), file=fout)


if __name__ == "__main__":
    main()
