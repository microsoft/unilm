# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import itertools
import logging
import os
from typing import Any, List, Optional

import numpy as np

import torch
import torch.nn.functional as F
import librosa
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset


logger = logging.getLogger(__name__)

def _collate_frames(
    frames: List[torch.Tensor], is_audio_input: bool = False
):
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    max_len = max(frame.size(0) for frame in frames)
    if is_audio_input:
        out = frames[0].new_zeros((len(frames), max_len))
    else:
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out

def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes, spk_embeds = [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 3, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                spk_embeds.append(items[2])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes, spk_embeds


def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def logmelfilterbank(
    audio,
    sampling_rate,
    fft_size=1024,
    hop_size=256,
    win_length=None,
    window="hann",
    num_mels=80,
    fmin=80,
    fmax=7600,
    eps=1e-10,
):
    """Compute log-Mel filterbank feature. 
    (https://github.com/kan-bayashi/ParallelWaveGAN/blob/master/parallel_wavegan/bin/preprocess.py)

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=fft_size, n_mels=num_mels, fmin=fmin, fmax=fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))



class TextToSpeechDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        store_labels: bool = True,
        src_dict: Optional[Dictionary] = None,
        tokenizer = None,
        reduction_factor: int = 1,
    ):
        self.audio_root, self.audio_names, inds, tot, self.wav_sizes, self.spk_embeds = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.src_dict = src_dict
        self.tokenizer = tokenizer

        self.num_labels = len(label_paths)
        self.label_processors = label_processors
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels

        self.normalize = normalize
        self.reduction_factor = reduction_factor
        logger.info(
            f"reduction_factor={reduction_factor}, normalize={normalize}"
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        fbank = logmelfilterbank(
            wav.view(-1).cpu().numpy(), 16000
        )
        fbank = torch.from_numpy(fbank).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav, fbank

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.tokenizer is not None:
            label = self.tokenizer.encode(label)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav, fbank = self.get_audio(index)
        labels = self.get_labels(index)
        spkembs = get_features_or_waveform(
            os.path.join(self.audio_root, self.spk_embeds[index])
        )
        spkembs = torch.from_numpy(spkembs).float()
        return {"id": index, "source": labels, "target": fbank, "spkembs": spkembs, "audio_name": self.audio_names[index]}
    
    def __len__(self):
        return len(self.wav_sizes)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        fbanks = [s["target"] for s in samples]
        fbank_sizes = [len(s) for s in fbanks]

        collated_fbanks = _collate_frames(fbanks)
        collated_fbanks_size = torch.tensor(fbank_sizes, dtype=torch.long)

        # thin out frames for reduction factor (B, Lmax, odim) ->  (B, Lmax//r, odim)
        if self.reduction_factor > 1:
            collated_fbanks_in = collated_fbanks[:, self.reduction_factor - 1 :: self.reduction_factor]
            collated_fbanks_size_in = collated_fbanks_size.new([torch.div(olen, self.reduction_factor, rounding_mode='floor') for olen in collated_fbanks_size])
        else:
            collated_fbanks_in, collated_fbanks_size_in = collated_fbanks, collated_fbanks_size

        prev_output_tokens = torch.cat(
            [collated_fbanks_in.new_zeros((collated_fbanks_in.shape[0], 1, collated_fbanks_in.shape[2])), collated_fbanks_in[:, :-1]], dim=1
        )

        # make labels for stop prediction
        labels = collated_fbanks.new_zeros(collated_fbanks.size(0), collated_fbanks.size(1))
        for i, l in enumerate(fbank_sizes):
            labels[i, l - 1 :] = 1.0

        spkembs = _collate_frames([s["spkembs"] for s in samples], is_audio_input=True)

        sources_by_label = [
            [s["source"][i] for s in samples] for i in range(self.num_labels)
        ]
        sources_list, lengths_list, ntokens_list = self.collater_label(sources_by_label)

        net_input = {
            "src_tokens": sources_list[0], 
            "src_lengths": lengths_list[0],
            "prev_output_tokens": prev_output_tokens,
            "tgt_lengths": collated_fbanks_size_in,
            "spkembs": spkembs,
            "task_name": "t2s",
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "name": [s["audio_name"] for s in samples],
            "net_input": net_input,
            "labels": labels,
            "dec_target": collated_fbanks,
            "dec_target_lengths": collated_fbanks_size,
            "src_lengths": lengths_list[0],
            "task_name": "t2s",
            "ntokens": ntokens_list[0],
            "target": collated_fbanks,
        }

        return batch

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, [self.src_dict.pad()])
        for targets, pad in itr:
            targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.wav_sizes[index]

    @property
    def sizes(self):
        return np.array(self.wav_sizes)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.wav_sizes)
        return np.lexsort(order)[::-1]

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
