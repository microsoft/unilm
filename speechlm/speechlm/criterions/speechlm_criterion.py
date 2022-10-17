# ----------------------------------------------------------------------------
# SpeechLM: Enhanced Speech Pre-Training with Unpaired Textual Data (https://arxiv.org/abs/2209.15329)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechLM
# Code based on fairseq: https://github.com/facebookresearch/fairseq/tree/272c4c5197250997148fb12c0db6306035f166a4
# 
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# ----------------------------------------------------------------------------

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)

@dataclass
class HSTCriterionConfig(FairseqDataclass):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )
    text_ctc_weight: float = field(
        default=0.1,
        metadata={"help": "weights for text CTC Loss, loss will be (hubert_loss + dec_weight * CE_Loss + text_weight * (CE_Loss + CTC_loss))"},
    )
    text_mum_weight: float = field(
        default=0.0,
        metadata={"help": "masked unit modeling weight from the text end"},
    )
    report_accuracy: bool = field(
        default=True,
        metadata={"help": "report decoder accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    no_ctc_blank: bool = field(
        default=False,
        metadata={"help": "mask out the blank of ctc, only when dec_loss_type=ctc"},
    )

@register_criterion("speechlm_criterion", dataclass=HSTCriterionConfig)
class SpeechLMCriterion(FairseqCriterion):
    def __init__(
        self, 
        task, 
        pred_masked_weight, 
        pred_nomask_weight, 
        loss_weights=None, 
        log_keys=None, 
        text_ctc_weight=0.1,
        text_mum_weight=0,
        report_accuracy=False, 
        ignore_prefix_size=0, 
        no_ctc_blank=False,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.text_ctc_weight = text_ctc_weight
        self.text_mum_weight = text_mum_weight
        self.report_accuracy = report_accuracy
        self.ignore_prefix_size = ignore_prefix_size
        self.no_ctc_blank = no_ctc_blank
        self.padding_idx = task.dictionaries[0].pad()
        self.eos_idx = task.dictionaries[0].eos()
        self.blank_idx = task.dictionaries[0].bos()

    def compute_hubert_loss(self, model, net_output, reduction, suffix=''):
        loss = 0
        sample_size = []
        logging_output = {}
        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True)
        targ_m_list = model.get_targets(net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}{suffix}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size.append(targ_m_list[0].numel())

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}{suffix}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size.append(targ_u_list[0].numel())
        
        sample_size = np.mean(sample_size)

        def compute_correct(logits, targets):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == targets
                min = logits.argmin(-1) == targets
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        with torch.no_grad():
            for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
                corr_m, count_m = compute_correct(logp_m, targ_m)
                logging_output[f"correct_m_{i}{suffix}"] = corr_m
                logging_output[f"count_m_{i}{suffix}"] = count_m

            for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
                corr_u, count_u = compute_correct(logp_u, targ_u)
                logging_output[f"correct_u_{i}{suffix}"] = corr_u
                logging_output[f"count_u_{i}{suffix}"] = count_u

        return loss, sample_size, logging_output


    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        reduction = "sum" if reduce else "none"

        if "net_input" in sample:
            text_sample = None
        else:
            text_sample = sample.get("text_paired")
            sample = sample.get("speech")

        ### 1. L_UMLM: do hubert forward and loss computation
        sample["modality"] = "speech"
        net_output = model(target_list=sample["target_list"], **sample["net_input"])
        loss, sample_size, logging_output = self.compute_hubert_loss(
            model,
            net_output,
            reduction,
        )
        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()
        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk]))
        
        ### 2. do text forward and loss computation
        if text_sample is not None:
            text_sample["modality"] = "text"
            ## 2.1 re-loading "target_list", in default case, target_list = [src_tokens],
            ## while in case of using "unit-phone-char" structure, target_list will be [ref_tokens]
            text_sample["net_input"]["target_list"] = [
                text_sample.get("ref_tokens", text_sample["net_input"]["src_tokens"].clone()),
            ]
            text_net_output = model(**text_sample["net_input"])

            ### 2.2 L_UMLM (text-end, not applied by default)
            if self.text_mum_weight > 0:
                loss_u2t, sample_size_u2t, logging_output_u2t = self.compute_hubert_loss(
                    model,
                    text_net_output,
                    reduction,
                    suffix="_u2t",
                )
                loss += self.text_mum_weight * loss_u2t * sample_size / sample_size_u2t
                logging_output.update(logging_output_u2t)

            ### 2.3 L_UCTC
            text_sample_size = text_sample["ntokens"]
            if self.text_ctc_weight > 0:
                text_ctc_loss = self.compute_ctc_loss(model, text_net_output, text_sample["target"], reduction=reduction)
                loss += self.text_ctc_weight * text_ctc_loss * sample_size / text_sample_size
                logging_output["text_ctc_loss"] = utils.item(text_ctc_loss)
                logging_output["text_sample_size"] = text_sample_size

        logging_output = {
            "loss": utils.item(loss) if reduce else loss,
            "ntokens": sample_size,
            "nsentences": sample["id"].numel() + (text_sample["id"].numel() if text_sample is not None else 0),
            "sample_size": sample_size,
            **logging_output,
        }

        return loss, sample_size, logging_output

    def compute_ctc_loss(self, model, net_output, target, reduction):
        logits = net_output["encoder_out_ctc"][0]  # (T, B, C) from the code-encoder
        if self.no_ctc_blank:
            ## set prob of <blank> to -inf
            logits = logits.float()
            logits[:, :, self.blank_idx] = -1000000.0
        
        lprobs = F.log_softmax(logits.float(), dim=-1)

        encoder_padding_mask = net_output["encoder_padding_mask"][0]
        non_padding_mask = ~encoder_padding_mask
        input_lengths = non_padding_mask.long().sum(-1)
        pad_mask = (target != self.padding_idx) & (target != self.eos_idx)
        targets_flat = target.masked_select(pad_mask)
        target_lengths = pad_mask.sum(-1)

        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                lprobs,
                targets_flat,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction=reduction,
                zero_infinity=True,
            )
        return loss

    def compute_ce_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if sample["modality"] == "speech":
            target = sample["decoder_target"]
            if self.ignore_prefix_size > 0:
                if getattr(lprobs, "batch_first", False):
                    lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                    target = target[:, self.ignore_prefix_size :].contiguous()
                else:
                    lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                    target = target[self.ignore_prefix_size :, :].contiguous()
        else:
            target = sample["target"]

        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log.get(lk, 0) for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log.get(lk, 0) for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log.get(lk, 0) for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

        if "text_sample_size" in logging_outputs[0]:
            text_sample_size = sum(log.get("text_sample_size", 0) for log in logging_outputs)
            for lk in logging_outputs[0].keys():
                if lk.startswith("text_") and lk.endswith("_loss"):
                    val = sum(log.get(lk, 0) for log in logging_outputs)
                    metrics.log_scalar(lk, val / text_sample_size / math.log(2), round=3)

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
