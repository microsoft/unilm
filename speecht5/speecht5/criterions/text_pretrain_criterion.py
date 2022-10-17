# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class TextPretrainCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    loss_weights: Optional[List[float]] = field(
        default_factory=lambda: [0.1,],
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    bart_weight: float = field(
        default=1.0,
        metadata={"help": "loss weight for cross entropy"},
    )


class TextPretrainCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, bart_weight, loss_weights=None):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.loss_weights = loss_weights
        self.bart_weight = bart_weight

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, codebook_out, encoder_output = model(**sample["net_input"])
        bart_loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        
        loss = self.bart_weight * bart_loss
        logging_output = {
            "loss": loss.item(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "bart_loss": bart_loss.item(),
            "sample_size": sample_size,
        }

        if "prob_perplexity" in codebook_out:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(codebook_out)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            if len(self.loss_weights) > len(extra_losses):
                modified_loss_weight = self.loss_weights[len(extra_losses):]
            else:
                modified_loss_weight = self.loss_weights

            # assert len(extra_losses) == len(self.loss_weights), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, modified_loss_weight):
                # print(n + str(coef))
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()

        if 'loss_prob_perplexity' in logging_output:
            logging_output['code_perplexity'] = codebook_out['code_perplexity'].item()

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        bart_loss_sum = sum(log.get("bart_loss", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "bart_loss", bart_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", bart_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["bart_loss"].avg)
            )

        if "loss_prob_perplexity" in logging_outputs[0].keys():
            val = sum(log["loss_prob_perplexity"] for log in logging_outputs)
            metrics.log_scalar("loss_prob_perplexity", val / sample_size / math.log(2), round=3)
        if "code_perplexity" in logging_outputs[0].keys():
            val = sum(log["code_perplexity"] for log in logging_outputs)
            metrics.log_scalar("code_perplexity", val / len(logging_outputs), round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
