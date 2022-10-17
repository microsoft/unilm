# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------
"""
Modified from: https://github.com/facebookresearch/fairseq/blob/272c4c5197250997148fb12c0db6306035f166a4/examples/speech_to_text/prep_covost_data.py
1. normalize the punctuation
2. instead of extract fbank features, we direcly use 16k-Hz waveform
"""
import argparse
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf
import sacremoses

log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text"]


def mp3_convert_wav(mp3_file, wav_file):
    sound = AudioSegment.from_mp3(mp3_file)
    sound=sound.set_frame_rate(16000)
    sound=sound.set_channels(1)
    sound=sound.set_sample_width(2)
    sound.export(wav_file, format="wav")

class CoVoST(Dataset):
    """Create a Dataset for CoVoST (https://github.com/facebookresearch/covost).

    Args:
        root (str): root path to the dataset and generated manifests/features
        source_language (str): source (audio) language
        target_language (str, optional): target (text) language,
        None for no translation (default: None)
        version (int, optional): CoVoST version. (default: 2)
        download (bool, optional): Whether to download the dataset if it is not
        found at root path. (default: ``False``).
    """

    COVOST_URL_TEMPLATE = (
        "https://dl.fbaipublicfiles.com/covost/"
        "covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
    )

    VERSIONS = {2}
    SPLITS = ["train", "dev", "test"]

    XX_EN_LANGUAGES = {
        1: ["fr", "de", "nl", "ru", "es", "it", "tr", "fa", "sv-SE", "mn", "zh-CN"],
        2: [
            "fr",
            "de",
            "es",
            "ca",
            "it",
            "ru",
            "zh-CN",
            "pt",
            "fa",
            "et",
            "mn",
            "nl",
            "tr",
            "ar",
            "sv-SE",
            "lv",
            "sl",
            "ta",
            "ja",
            "id",
            "cy",
        ],
    }
    EN_XX_LANGUAGES = {
        1: [],
        2: [
            "de",
            "tr",
            "fa",
            "sv-SE",
            "mn",
            "zh-CN",
            "cy",
            "ca",
            "sl",
            "et",
            "id",
            "ar",
            "ta",
            "lv",
            "ja",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        source_language: str,
        target_language: Optional[str] = None,
        version: int = 2,
    ) -> None:
        assert version in self.VERSIONS and split in self.SPLITS
        assert source_language is not None
        self.no_translation = target_language is None
        if not self.no_translation:
            assert "en" in {source_language, target_language}
            if source_language == "en":
                assert target_language in self.EN_XX_LANGUAGES[version]
            else:
                assert source_language in self.XX_EN_LANGUAGES[version]
        else:
            # Hack here so that we can get "split" column from CoVoST TSV.
            # Note that we use CoVoST train split for ASR which is an extension
            # to Common Voice train split.
            target_language = "de" if source_language == "en" else "en"

        self.root: Path = Path(root)

        cv_tsv_path = self.root / "validated.tsv"
        assert cv_tsv_path.is_file()

        covost_url = self.COVOST_URL_TEMPLATE.format(
            src_lang=source_language, tgt_lang=target_language
        )
        covost_archive = self.root / Path(covost_url).name
        if not covost_archive.is_file():
            download_url(covost_url, self.root.as_posix(), hash_value=None)
        extract_archive(covost_archive.as_posix())

        cv_tsv = load_df_from_tsv(cv_tsv_path)
        covost_tsv = load_df_from_tsv(
            self.root / Path(covost_url).name.replace(".tar.gz", "")
        )
        df = pd.merge(
            left=cv_tsv[["path", "sentence", "client_id"]],
            right=covost_tsv[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        if split == "train":
            df = df[(df["split"] == split) | (df["split"] == f"{split}_covost")]
        else:
            df = df[df["split"] == split]
        data = df.to_dict(orient="index").items()
        data = [v for k, v in sorted(data, key=lambda x: x[0])]
        self.data = []
        for e in data:
            try:
                path = self.root / "clips" / e["path"]
                _ = torchaudio.info(path.as_posix())
                self.data.append(e)
            except RuntimeError:
                pass
        
        self.normalizer = sacremoses.MosesPunctNormalizer(
            lang=target_language,
            pre_replace_unicode_punct=True,
            post_remove_control_chars=True,
        )

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, sentence, translation, speaker_id,
            sample_id)``
        """
        data = self.data[n]
        path = self.root / "clips" / data["path"]
        # waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation = None if self.no_translation else data["translation"]
        translation = self.normalizer.normalize(translation)
        speaker_id = data["client_id"]
        _id = data["path"].replace(".mp3", "")
        return path, -1, sentence, translation, speaker_id, _id

    def __len__(self) -> int:
        return len(self.data)


def process(args):
    root = Path(args.data_root).absolute() / args.src_lang
    outroot = root / f"{args.src_lang}-{args.tgt_lang}"
    if args.vocab_type != "char":
        outroot = root / f"{args.src_lang}-{args.tgt_lang}-{args.vocab_type}"
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    #1.  Extract featuress
    # mp3-to-wav can take long long time, better run it externally with multi threads.
    feature_root = root / "wav"
    # feature_root.mkdir(exist_ok=True)
    # for split in CoVoST.SPLITS:
    #     print(f"Fetching split {split}...")
    #     dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        # print("Converting mp3 to wav...")
        # handle = open(root / f"{split}.id", "w")
        # for waveform, _, _, _, _, utt_id in tqdm(dataset):
        #     wav_file = feature_root / f"{utt_id}.wav"
            # print(waveform, file=handle)
            # mp3_convert_wav(waveform, wav_file)

    #2. Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in CoVoST.SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = CoVoST(root, split, args.src_lang, args.tgt_lang)
        for waveform, _, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            wav_file = feature_root / f"{utt_id}.wav"
            manifest["id"].append(utt_id)
            manifest["audio"].append(wav_file.as_posix().replace("/data/", "/mnt/default/"))
            manifest["n_frames"].append(sf.info(wav_file).frames)
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split, min_n_frames=320, max_n_frames=480000)
        save_df_to_tsv(df, outroot / f"{split}_{task}.tsv")
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            outroot / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size
        )
    # Generate config YAML
    # gen_config_yaml(
    #     outroot,
    #     spm_filename=spm_filename_prefix + ".model",
    #     yaml_filename=f"config_{task}.yaml",
    #     specaugment_policy="lb",
    # )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", required=True, type=str,
        help="data root with sub-folders for each language <root>/<src_lang>"
    )
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=1000, type=int)
    parser.add_argument("--src-lang", "-s", required=True, type=str)
    parser.add_argument("--tgt-lang", "-t", type=str)
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
