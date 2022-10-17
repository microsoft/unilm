# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import logging
import os
from typing import Any, List, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
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
    """manifest tsv: src_wav, src_nframe, tgt_wav, tgt_nframe, tgt_spkemb"""
    n_long, n_short = 0, 0
    src_names, tgt_names, inds, sizes, tgt_sizes, spk_embeds = [], [], [], [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) >= 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                src_names.append(items[0])
                tgt_names.append(items[2])
                tgt_sizes.append(items[3])
                spk_embeds.append(items[4])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(src_names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, src_names, inds, tot, sizes, tgt_names, tgt_sizes, spk_embeds


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


class SpeechToSpeechDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        shuffle: bool = True,
        normalize: bool = False,
        reduction_factor: int = 1,
    ):
        self.audio_root, self.audio_names, inds, tot, self.wav_sizes, self.tgt_audios, self.tgt_sizes, self.tgt_spkembs = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle

        self.normalize = normalize
        self.reduction_factor = reduction_factor
        logger.info(
            f"reduction_factor={reduction_factor}, normalize={normalize}"
        )

    def get_audio(self, index):
        import soundfile as sf

        wav_fbank = []
        for name in [self.audio_names[index], self.tgt_audios[index]]:
            wav_path = os.path.join(self.audio_root, name)
            wav, cur_sample_rate = sf.read(wav_path)
            wav = torch.from_numpy(wav).float()
            fbank = logmelfilterbank(
                wav.view(-1).cpu().numpy(), 16000
            )
            fbank = torch.from_numpy(fbank).float()
            wav = self.postprocess(wav, cur_sample_rate)
            wav_fbank.append(wav)
            wav_fbank.append(fbank)
        src_wav, src_fbank, tgt_wav, tgt_fbank = wav_fbank
        return src_wav, src_fbank, tgt_wav, tgt_fbank

    def __getitem__(self, index):
        src_wav, src_fbank, tgt_wav, tgt_fbank = self.get_audio(index)
        spkembs = np.load(os.path.join(self.audio_root, self.tgt_spkembs[index]))
        spkembs = torch.from_numpy(spkembs).float()
        name = self.audio_names[index].replace("/", ".").replace(".wav", "") + "-" + self.tgt_audios[index].replace("/", ".").replace(".wav", "") + ".wav"
        return {"id": index, "source": src_wav, "target": tgt_fbank, "spkembs": spkembs, "audio_name": name, "tgt_name": self.tgt_audios[index]}
    
    def __len__(self):
        return len(self.wav_sizes)

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]

        audio_size = max(audio_sizes)
        collated_audios, padding_mask = self.collater_audio(
            audios, audio_size
        )

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

        net_input = {
            "source": collated_audios,
            "padding_mask": padding_mask,
            "prev_output_tokens": prev_output_tokens,
            "tgt_lengths": collated_fbanks_size_in,
            "spkembs": spkembs,
            "task_name": "s2s",
        }
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "name": [s["audio_name"] for s in samples],
            "tgt_name": [s["tgt_name"] for s in samples],
            "net_input": net_input,
            "labels": labels,
            "dec_target": collated_fbanks,
            "dec_target_lengths": collated_fbanks_size,
            "src_lengths": torch.LongTensor(audio_sizes),
            "task_name": "s2s",
            "ntokens": sum(audio_sizes),
            "target": collated_fbanks,
        }

        return batch

    def collater_audio(self, audios, audio_size):
        collated_audios = audios[0].new_zeros(len(audios), audio_size)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
        )
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                raise Exception("Diff should not be larger than 0")
        return collated_audios, padding_mask


    def num_tokens(self, index):
        return self.wav_sizes[index]

    def size(self, index):
        return self.wav_sizes[index], self.tgt_sizes[index]

    @property
    def sizes(self):
        return np.array(self.wav_sizes)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """No cache dataset if dataset is large-scale. Cache dataset for small dataset."""
        return True

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
