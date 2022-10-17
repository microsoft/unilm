# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

"""
Modified form: https://github.com/facebookresearch/fairseq/blob/272c4c5197250997148fb12c0db6306035f166a4/fairseq/sequence_generator.py
"""

import torch
import numpy as np

from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.speech_generator import SpeechGenerator

class NonAutoregressiveUnitGenerator(SpeechGenerator):
    @torch.no_grad()
    def generate(self, model, sample, has_targ=False, **kwargs):
        model.eval()

        bsz, max_src_len = sample["net_input"]["src_tokens"].size()
        n_frames_per_step = model.encoder.n_frames_per_step
        out_dim = model.encoder.out_dim
        raw_dim = out_dim // n_frames_per_step

        logit, logit_post, out_lens, log_dur_out, _, _ = model(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            speaker=sample["speaker"],
            durations=sample["durations"],
            pitches=sample["pitches"],
            energies=sample["energies"],
        )
        if logit_post is not None:
            logit = logit_post

        logit = logit.view(bsz, -1, raw_dim)
        pred = logit.argmax(dim=-1)

        ## get duration prediction
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        padding_mask = src_tokens.eq(model.encoder.padding_idx)
        d_factor = 1.0 ## set by model
        dur_out = torch.clamp(
            torch.round((torch.exp(log_dur_out) - 1) * d_factor).long(), min=0
        )
        dur_out.masked_fill_(padding_mask, 0)
        x = src_tokens.unsqueeze(-1)
        x, src_out_lens = model.encoder.var_adaptor.length_regulator(x, dur_out)
        fa_src_tokens = x.view(bsz, -1)
        
        finalized = [
            {
                "unit": pred[b, :l],
                "fa_src": fa_src_tokens[b, :l],
                "duration": dur_out[b, :L],
            }
            for b, l, L in zip(range(bsz), out_lens, src_lengths)
        ]

        return finalized
