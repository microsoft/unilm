# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import itertools
import logging
import io
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Union, Tuple

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils, Dictionary
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    read_from_stored_zip,
    is_sf_audio_data,
)

FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS = {".npy", ".wav", ".flac", ".ogg"}

logger = logging.getLogger(__name__)

def parse_path(path: str) -> Tuple[str, List[int]]:
    """Parse data path which is either a path to
    1. a .npy/.wav/.flac/.ogg file
    2. a stored ZIP file with slicing info: "[zip_path]:[offset]:[length]"

      Args:
          path (str): the data path to parse

      Returns:
          file_path (str): the file path
          slice_ptr (list of int): empty in case 1;
            byte offset and length for the slice in case 2
    """

    if Path(path).suffix in FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS:
        _path, slice_ptr = path, []
    else:
        _path, *slice_ptr = path.split(":")
        if not Path(_path).is_file():
            raise FileNotFoundError(f"File not found: {_path}")
    assert len(slice_ptr) in {0, 1, 2}, f"Invalid path: {path}"
    slice_ptr = [int(i) for i in slice_ptr]
    return _path, slice_ptr

def load_audio(manifest_path, max_keep, min_keep, retry_times=5):
    n_long, n_short = 0, 0
    names, inds, sizes, chunk_names, chunk_indices = [], [], [], [], []
    for i in range(retry_times):
        with open(manifest_path) as f:
            root = f.readline().strip()
            for ind, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_keep is not None and sz < min_keep:
                    n_short += 1
                elif max_keep is not None and sz > max_keep:
                    n_long += 1
                else:
                    fname = items[0].split(":")
                    if len(fname) > 2:
                        if len(chunk_names) == 0 or fname[0] != chunk_names[-1]:
                            chunk_names.append(fname[0])
                            chunk_indices.append(len(names))
                    names.append(items[0])
                    inds.append(ind)
                    sizes.append(sz)
        if len(names) == 0:
            logger.warn(f"Fail to load manifest for the {i} time")
            time.sleep(1)
            continue
        else:
            break
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes, chunk_names, chunk_indices


def load_label(label_path, inds, tot, retry_times=5):
    for i in range(retry_times):
        with open(label_path) as f:
            labels = [line.rstrip() for line in f]
            if len(labels) == 0:
                logger.warn(f"Fail to load label for the {i} time")
                time.sleep(1)
                continue
            else:
                break
    assert (
        len(labels) == tot
    ), f"number of labels does not match ({len(labels)} != {tot})"
    labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot, retry_times=5):
    for i in range(retry_times):
        with open(label_path) as f:
            code_lengths = [len(line.encode("utf-8")) for line in f]
            if len(code_lengths) == 0:
                logger.warn(f"Fail to load label for the {i} time")
                time.sleep(1)
                continue
            else:
                break
    assert (
        len(code_lengths) == tot
    ), f"number of labels does not match ({len(code_lengths)} != {tot})"
    offsets = list(itertools.accumulate([0] + code_lengths))
    offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class HubertDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        tgt_dict: Optional[Dictionary] = None,
        add_decoder_target: bool = False,
        fine_tuning: bool = False,
        tgt_lang_idx: int = None,
        tokenizer = None,
        mbart_style_lang_id: bool = False,
        retry_times: int = 5,
        reduce_label_for_dec: bool = True,
    ):
        self.audio_root, self.audio_names, inds, tot, self.wav_sizes, self.chunk_names, self.chunk_indices = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size, retry_times
        )
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.tgt_dict = tgt_dict
        self.add_decoder_target = add_decoder_target
        self.fine_tuning = fine_tuning

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.epoch = 0

        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot, retry_times) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot, retry_times) for p in label_paths
            ]
        assert label_processors is None or len(label_processors) == self.num_labels
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.wav_sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        self.tgt_lang_idx = tgt_lang_idx
        self.tokenizer = tokenizer
        self.mbart_style_lang_id = mbart_style_lang_id
        self.retry_times = retry_times
        self.reduce_label_for_dec = reduce_label_for_dec
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, tgt_lang_idx={self.tgt_lang_idx}, reduce_label_for_dec={reduce_label_for_dec}, "
            f"mbart_style_lang_id={mbart_style_lang_id}, normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.required_batch_size_multiple = required_batch_size_multiple
        if isinstance(indices[0], np.ndarray):
            batch_list = []
            for indice in indices:
                batch = super(HubertDataset, self).batch_by_size(indice, max_tokens, max_sentences, required_batch_size_multiple)
                batch_list.append(batch)
            return batch_list
        else:
            return super(HubertDataset, self).batch_by_size(indices, max_tokens, max_sentences, required_batch_size_multiple)
    def shuffle_batches(self, batches, seed):
        if isinstance(batches[0], list):
            new_batches = []
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
                for batch in batches:
                    np.random.shuffle(batch)
                    new_batches.extend(batch)
            return new_batches
        else:
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
        return batches
    
    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        _path, slice_ptr = parse_path(wav_path)
        if len(slice_ptr) == 1:
            import kaldiio
            feat = kaldiio.load_mat(wav_path)
            feat = torch.from_numpy(feat).float()
            if self.normalize:
                with torch.no_grad():
                    feat = F.layer_norm(feat, feat.shape[-1])
            return feat
        else:
            if len(slice_ptr) == 2:
                byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
                assert is_sf_audio_data(byte_data)
                wav_path = io.BytesIO(byte_data)
            for i in range(self.retry_times):
                if i < self.retry_times - 1:
                    try:
                        wav, cur_sample_rate = sf.read(wav_path)
                        break
                    except Exception as e:
                        logger.warn(f"Fail to load wav for the {i} time")
                        logger.warn(e)
                        time.sleep(1)
                        continue
                else:
                    wav, cur_sample_rate = sf.read(wav_path)

            wav = torch.from_numpy(wav).float()
            wav = self.postprocess(wav, cur_sample_rate)
            return wav

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.tokenizer is not None and self.fine_tuning:
            label = self.tokenizer.encode(label)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        wav = self.get_audio(index)
        labels = self.get_labels(index)
        return {"id": index, "source": wav, "label_list": labels}

    def __len__(self):
        return len(self.wav_sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        feat_dim = audios[0].size(-1) if audios[0].dim() > 1 else 1
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size, feat_dim,
        )

        targets_by_label = [
            [s["label_list"][i] for s in samples] for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        if self.add_decoder_target:
            if self.fine_tuning:
                    decoder_label = [
                        torch.cat((targets_list[0][i, :lengths_list[0][i]], torch.tensor([self.tgt_dict.eos()])), 0).long()
                        for i in range(targets_list[0].size(0))
                    ]
            else:
                if self.tokenizer is not None:
                    decoder_label = [
                        # Set 48 for translate int to char and avoid \n
                        torch.cat(
                            (
                                torch.tensor(
                                    self.tokenizer.sp.Encode(
                                        "".join(
                                            [chr(j + 48) for j in (
                                                targets_list[0][i, :lengths_list[0][i]].unique_consecutive() if self.reduce_label_for_dec else targets_list[0][i, :lengths_list[0][i]]
                                            ).tolist()]
                                        ), out_type=int
                                    )
                                ), 
                                torch.tensor([self.tgt_dict.eos()])
                            ), dim=0
                        ).long()
                        for i in range(targets_list[0].size(0))
                    ]
                else:
                    decoder_label = [
                        torch.cat((targets_list[0][i, :lengths_list[0][i]].unique_consecutive() if self.reduce_label_for_dec else targets_list[0][i, :lengths_list[0][i]], torch.tensor([self.tgt_dict.eos()])), 0).long()
                        for i in range(targets_list[0].size(0))
                    ]

            if self.mbart_style_lang_id:
                decoder_label = [
                    torch.cat((decoder_label[i], torch.tensor([self.tgt_lang_idx])), 0).long()
                    for i in range(targets_list[0].size(0))
                ]

            dec_ntokens = sum(x.size(0) for x in decoder_label)
            decoder_target = data_utils.collate_tokens(
                decoder_label,
                self.tgt_dict.pad(),
                self.tgt_dict.eos() if not self.mbart_style_lang_id else self.tgt_lang_idx,
                left_pad=False,
                move_eos_to_beginning=False,
            )
            decoder_target_lengths = torch.tensor(
                [x.size(0) for x in decoder_label], dtype=torch.long
            )
            prev_output_tokens = data_utils.collate_tokens(
                decoder_label,
                self.tgt_dict.pad(),
                self.tgt_dict.eos() if not self.mbart_style_lang_id else self.tgt_lang_idx,
                left_pad=False,
                move_eos_to_beginning=True,
            )
            
            if self.tgt_lang_idx is not None and not self.mbart_style_lang_id:
                assert (prev_output_tokens[:, 0] != self.tgt_dict.eos()).sum() == 0
                prev_output_tokens[:, 0] = self.tgt_lang_idx

            net_input = {
                "source": collated_audios, 
                "padding_mask": padding_mask,
                "prev_output_tokens": prev_output_tokens,
            }
            batch = {
                "id": torch.LongTensor([s["id"] for s in samples]),
                "net_input": net_input,
                "decoder_target": decoder_target,
                "decoder_target_lengths": decoder_target_lengths,
                "dec_ntokens": dec_ntokens,
                "lang_idx": self.tgt_lang_idx,
            }
        else:
            net_input = {"source": collated_audios, "padding_mask": padding_mask}
            batch = {
                "id": torch.LongTensor([s["id"] for s in samples]),
                "net_input": net_input,
            }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list
        return batch

    def collater_audio(self, audios, audio_size, feat_dim=1):
        collated_audios = audios[0].new_zeros(len(audios), audio_size, feat_dim)
        padding_mask = (
            torch.BoolTensor(collated_audios.shape[0:2]).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            audio = audio.view(-1, feat_dim)
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff, feat_dim), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios.squeeze(-1), padding_mask, audio_starts

    def collater_frm_label(self, targets, audio_size, audio_starts, label_rate, pad):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s : s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(targets, pad_idx=pad, left_pad=False)
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                targets, lengths, ntokens = self.collater_seq_label(targets, pad)
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.wav_sizes[index]
        return min(self.wav_sizes[index], self.max_sample_size)

    @property
    def sizes(self):
        return np.array(self.wav_sizes)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            if len(self.chunk_names) > 0:
                logger.info(f"ordered indices for epoch {self.epoch}")
                with data_utils.numpy_seed(self.epoch):
                    self.chunk_order = np.random.permutation(len(self.chunk_names))
                chunk_count = 0
                tmp_sizes = []
                tmp_indices = []
                indice = []
                for i in self.chunk_order:
                    chunk_count += 1
                    start = self.chunk_indices[i]
                    end = self.chunk_indices[i+1] if i < len(self.chunk_names) - 1 else len(self)
                    size = list(self.sizes[start:end])
                    tmp_indices.extend(list(np.arange(start, end)))
                    tmp_sizes.extend(size)
                    if chunk_count % 10 == 0 or i == self.chunk_order[0]:
                        order = [np.random.permutation(len(tmp_indices))]
                        order.append(
                            np.minimum(
                                np.array(tmp_sizes),
                                self.max_sample_size,
                            )
                        )
                        sort_idx = np.lexsort(order)[::-1]
                        indice.append(np.array([tmp_indices[k] for k in sort_idx]))
                        tmp_indices = []
                        tmp_sizes =[]
                return indice
            else:
                order = [np.random.permutation(len(self))]
                order.append(
                    np.minimum(
                        np.array(self.sizes),
                        self.max_sample_size,
                    )
                )
                return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

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
