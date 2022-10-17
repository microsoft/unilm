# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

from typing import List, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import lengths_to_mask
from fairseq.models.fairseq_model import FairseqEncoderModel

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        ntokens = (~pad_mask).sum()
        nll_loss = nll_loss.sum() / ntokens
        smooth_loss = smooth_loss.sum() / ntokens
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@dataclass
class FastText2UnitCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    dur_loss_weight: float = field(
        default=1.0,
        metadata={"help": "scale of duration loss"},
    )
    report_accuracy: bool = field(
        default=True,
        metadata={"help": "report decoder accuracy metric"},
    )

@register_criterion("fasttext2unit_criterion", dataclass=FastText2UnitCriterionConfig)
class FastText2UnitLoss(FairseqCriterion):
    def __init__(self,
        task,
        label_smoothing=0,
        dur_loss_weight=1.0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.eps = label_smoothing
        self.dur_loss_weight = dur_loss_weight
        self.pad_idx = task.tgt_dict.pad()
        self.report_accuracy = report_accuracy

    def forward(self, model: FairseqEncoderModel, sample, reduction="mean"):
        src_tokens = sample["net_input"]["src_tokens"]
        src_lens = sample["net_input"]["src_lengths"]
        tgt_lens = sample["target_lengths"]
        
        _feat_out, _feat_out_post, out_lens, log_dur_out, pitch_out, energy_out = model(
            src_tokens=src_tokens,
            src_lengths=src_lens,
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            incremental_state=None,
            target_lengths=tgt_lens,
            speaker=sample["speaker"],
            durations=sample["durations"],
            pitches=sample["pitches"],
            energies=sample["energies"],
        )

        src_mask = lengths_to_mask(sample["net_input"]["src_lengths"])
        tgt_mask = lengths_to_mask(sample["target_lengths"])

        lprobs = model.get_normalized_probs((_feat_out,), log_probs=True)
        target = sample["target"].long()
        ce_loss, nll_loss = label_smoothed_nll_loss(lprobs, target, self.eps, self.padding_idx, reduce=True)

        pitches, energies = sample["pitches"], sample["energies"]
        if pitches is not None:
            pitch_out, pitches = pitch_out[src_mask], pitches[src_mask]
            pitch_loss = F.mse_loss(pitch_out, pitches, reduction=reduction)
        else:
            pitch_loss = 0
        if energies is not None:
            energy_out, energies = energy_out[src_mask], energies[src_mask]
            energy_loss = F.mse_loss(energy_out, energies, reduction=reduction)
        else:
            energy_loss = 0

        log_dur_out = log_dur_out[src_mask]
        dur = sample["durations"].float()
        dur = dur.half() if log_dur_out.type().endswith(".HalfTensor") else dur
        log_dur = torch.log(dur + 1)[src_mask]
        dur_loss = F.mse_loss(log_dur_out, log_dur, reduction=reduction)
        dur_loss = self.dur_loss_weight * dur_loss

        loss = ce_loss + dur_loss + pitch_loss + energy_loss

        sample_size = sample["nsentences"]
        logging_output = {
            "loss": utils.item(loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "ce_loss": utils.item(ce_loss.data),
            "dur_loss": utils.item(dur_loss.data),
            "pitch_loss": utils.item(pitch_loss),
            "energy_loss": utils.item(energy_loss),
        }
        if self.report_accuracy:
            n_correct = lprobs.argmax(-1).masked_select(tgt_mask).eq(target.masked_select(tgt_mask)).sum()
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = tgt_mask.sum()
        return loss, 1, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        ns = [log.get("sample_size", 0) for log in logging_outputs]
        ntot = sum(ns)
        ws = [n / (ntot + 1e-8) for n in ns]
        for key in [
            "loss",
            "ce_loss",
            "dur_loss",
            "pitch_loss",
            "energy_loss",
        ]:
            vals = [log.get(key, 0) for log in logging_outputs]
            val = sum(val * w for val, w in zip(vals, ws))
            metrics.log_scalar(key, val, ntot, round=3)
        metrics.log_scalar("sample_size", ntot, len(logging_outputs))

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                    sum(log.get("n_correct", 0) for log in logging_outputs)
                )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        # inference metrics
        if "targ_frames" not in logging_outputs[0]:
            return
        n = sum(log.get("targ_frames", 0) for log in logging_outputs)
        for key, new_key in [
            ("mcd_loss", "mcd_loss"),
            ("pred_frames", "pred_ratio"),
            ("nins", "ins_rate"),
            ("ndel", "del_rate"),
        ]:
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(new_key, val / n, n, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
