# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion
from speecht5.criterions.text_to_speech_loss import TexttoSpeechLoss, TexttoSpeechLossConfig


@dataclass
class SpeechPretrainCriterionConfig(TexttoSpeechLossConfig):
    pred_masked_weight: float = field(
        default=1.0,
        metadata={"help": "weight for predictive loss for masked frames"},
    )
    pred_nomask_weight: float = field(
        default=0.0,
        metadata={"help": "weight for predictive loss for unmasked frames"},
    )
    loss_weights: Optional[List[float]] = field(
        default_factory=lambda: [10,],
        metadata={"help": "weights for additional loss terms (not first one)"},
    )
    log_keys: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "output keys to log"},
    )
    hubert_weight: float = field(
        default=1.0,
        metadata={"help": "weight of hubert loss"},
    )
    dec_weight: float = field(
        default=1.0,
        metadata={"help": "weight of decoder loss"},
    )


class SpeechPretrainCriterion(FairseqCriterion):
    def __init__(
        self, 
        task, 
        sentence_avg, 
        pred_masked_weight, 
        pred_nomask_weight, 
        loss_weights=None, 
        log_keys=None,
        use_masking=True,
        use_weighted_masking=False,
        loss_type="L1",
        bce_pos_weight=5.0,
        hubert_weight=1.0,
        dec_weight=1.0,
    ):
        super().__init__(task)
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys
        self.hubert_weight = hubert_weight
        self.dec_weight = dec_weight

        self.speech_criterion = TexttoSpeechLoss(
            task,
            sentence_avg,
            use_masking,
            use_weighted_masking,
            loss_type,
            bce_pos_weight,
        )

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.dec_weight == 0:
            sample["net_input"]["only_hubert"] = True
        net_output, net_output_dec = model(target_list=sample["target_list"], **sample["net_input"])
        loss = 0.
        sample_size = 0
        logging_output = {}
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = model.get_logits(net_output, True)
        targ_m_list = model.get_targets(None, net_output, True)
        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        for i, (logp_m, targ_m) in enumerate(zip(logp_m_list, targ_m_list)):
            loss_m = F.cross_entropy(logp_m, targ_m, reduction=reduction)
            loss_m_list.append(loss_m)
            logging_output[f"loss_m_{i}"] = loss_m.detach().item()
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += targ_m_list[0].numel()

        loss_u_list = []
        logp_u_list = model.get_logits(net_output, False)
        targ_u_list = model.get_targets(None, net_output, False)
        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        for i, (logp_u, targ_u) in enumerate(zip(logp_u_list, targ_u_list)):
            loss_u = F.cross_entropy(logp_u, targ_u, reduction=reduction)
            loss_u_list.append(loss_u)
            logging_output[f"loss_u_{i}"] = loss_u.detach().item()
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += targ_u_list[0].numel()

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses, names = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            if len(self.loss_weights) > len(extra_losses):
                modified_loss_weight = self.loss_weights[:len(extra_losses)]
            else:
                modified_loss_weight = self.loss_weights

            # assert len(extra_losses) == len(self.loss_weights), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, modified_loss_weight):
                # print(n + str(coef))
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.detach().item()

        logging_output = {
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            "ngpu": 1,
            **logging_output,
        }

        if 'loss_prob_perplexity' in logging_output:
            logging_output['code_perplexity'] = net_output['code_perplexity'].detach().item()     

        for lk in self.log_keys:
            if lk in net_output:
                logging_output[lk] = float((net_output[lk].item()))

        def compute_correct(logits):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == 0
                min = logits.argmin(-1) == 0
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        with torch.no_grad():
            for i, logp_m in enumerate(logp_m_list):
                corr_m, count_m = compute_correct(logp_m)
                logging_output[f"correct_m_{i}"] = corr_m
                logging_output[f"count_m_{i}"] = count_m

            for i, logp_u in enumerate(logp_u_list):
                corr_u, count_u = compute_correct(logp_u)
                logging_output[f"correct_u_{i}"] = corr_u
                logging_output[f"count_u_{i}"] = count_u

        if self.dec_weight == 0.0:
            logging_output["loss"] = loss.item() if reduce else loss
            return loss, sample_size, logging_output

#       ## dec loss
        dec_loss, l1_loss, l2_loss, bce_loss, enc_dec_attn_loss = self.speech_criterion.compute_loss(model, net_output_dec, sample)
        
        # Log tts loss
        logging_output['dec_loss'] = dec_loss.item()
        logging_output['l1_loss'] = l1_loss.item()
        logging_output['l2_loss'] = l2_loss.item()
        logging_output['bce_loss'] = bce_loss.item()
        if enc_dec_attn_loss is not None:
            logging_output['enc_dec_attn_loss'] = enc_dec_attn_loss.item()

        loss = self.hubert_weight * loss + self.dec_weight * sample_size * dec_loss
        logging_output["loss"] = loss.item() if reduce else loss
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        dec_loss_sum = sum(log.get("dec_loss", 0) for log in logging_outputs)
        l1_loss_sum = sum(log.get("l1_loss", 0) for log in logging_outputs)
        l2_loss_sum = sum(log.get("l2_loss", 0) for log in logging_outputs)
        bce_loss_sum = sum(log.get("bce_loss", 0) for log in logging_outputs)
        ngpu = sum(log.get("ngpu", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar("nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))
        else:
            metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])
            elif lk == 'code_perplexity':
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / len(logging_outputs), round=3)

        metrics.log_scalar(
            "dec_loss", dec_loss_sum / ngpu, sample_size, 2, round=5
        )
        metrics.log_scalar(
            "l1_loss", l1_loss_sum / ngpu, sample_size, 2, round=5
        )
        metrics.log_scalar(
            "l2_loss", l2_loss_sum / ngpu, sample_size, 2, round=5
        )
        metrics.log_scalar(
            "bce_loss", bce_loss_sum / ngpu, sample_size, 2, round=5
        )
        if "enc_dec_attn_loss" in logging_outputs[0]:
            enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "enc_dec_attn_loss", enc_dec_attn_loss_sum / ngpu, sample_size, round=8
            )

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
