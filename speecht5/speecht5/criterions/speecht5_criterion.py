# --------------------------------------------------------
# SpeechT5: Unified-Modal Encoder-Decoder Pre-Training for Spoken Language Processing (https://arxiv.org/abs/2110.07205)
# Github source: https://github.com/microsoft/SpeechT5/tree/main/SpeechT5
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq and espnet code bases
# https://github.com/pytorch/fairseq; https://github.com/espnet/espnet
# --------------------------------------------------------

import re
from dataclasses import dataclass

import math
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from speecht5.criterions.text_to_speech_loss import TexttoSpeechLoss
from speecht5.criterions.text_pretrain_criterion import TextPretrainCriterion, TextPretrainCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
from speecht5.criterions.speech_pretrain_criterion import SpeechPretrainCriterion, SpeechPretrainCriterionConfig
from speecht5.criterions.speech_to_text_loss import SpeechtoTextLoss, SpeechtoTextLossConfig                                                  
from fairseq.logging.meters import safe_round

@dataclass
class SpeechT5CriterionConfig(
    LabelSmoothedCrossEntropyCriterionConfig, 
    TextPretrainCriterionConfig,
    SpeechPretrainCriterionConfig,
    SpeechtoTextLossConfig
    ):
    pass

@register_criterion(
    "speecht5", dataclass=SpeechT5CriterionConfig
)
class SpeechT5Criterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        pred_masked_weight, 
        pred_nomask_weight, 
        loss_weights=None, 
        log_keys=None,
        ignore_prefix_size=0,
        report_accuracy=False,
        use_masking=True,
        use_weighted_masking=False,
        loss_type="L1",
        bce_pos_weight=5.0,
        bce_loss_lambda=1.0,
        use_guided_attn_loss=False,
        num_heads_applied_guided_attn=2,
        ce_weight=1.0,
        ctc_weight=0.0,
        hubert_weight=1.0,
        dec_weight=1.0,
        bart_weight=1.0,
    ):
        super().__init__(task)
        self.speech_criterion = TexttoSpeechLoss(
            task,
            sentence_avg,
            use_masking,
            use_weighted_masking,
            loss_type,
            bce_pos_weight,
            bce_loss_lambda,
            use_guided_attn_loss,
            num_heads_applied_guided_attn=num_heads_applied_guided_attn,
        )
        self.text_criterion = SpeechtoTextLoss(
            SpeechtoTextLossConfig,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            ce_weight,
            ctc_weight
        )
        self.text_pretrain_criterion = TextPretrainCriterion(
            task,
            sentence_avg,
            bart_weight,
            loss_weights,
        )
        self.speech_pretrain_criterion = SpeechPretrainCriterion(
            task,
            sentence_avg,
            pred_masked_weight,
            pred_nomask_weight,
            loss_weights,
            log_keys,
            use_masking,
            use_weighted_masking,
            loss_type,
            bce_pos_weight,
            hubert_weight,
            dec_weight
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        task_name = sample['task_name']
        if task_name == 's2t' or task_name == 's2c':
            return self.text_criterion(model, sample, reduce)
        elif task_name == 't2s' or task_name == 's2s':
            return self.speech_criterion(model, sample)
        elif task_name == 'text_pretrain':
            return self.text_pretrain_criterion(model, sample, reduce)
        elif task_name == 'speech_pretrain':
            return self.speech_pretrain_criterion(model, sample, reduce)

    @classmethod
    def reduce_metrics(cls, logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        logging_outputs_dict = {}
        for logging_output in logging_outputs:
            for task_name in logging_output:
                if task_name not in ['s2t', 't2s', 's2c', 's2s', 'text_pretrain', 'speech_pretrain']:
                    continue

                if task_name not in logging_outputs_dict:
                    logging_outputs_dict[task_name] = []
                logging_outputs_dict[task_name].append(logging_output[task_name])

        for task_name in logging_outputs_dict:
            if task_name == 's2t':
                # LabelSmoothedCrossEntropyCriterion.reduce_metrics([logging_output['s2t'] for logging_output in logging_outputs])
                s2t_logging_output = logging_outputs_dict[task_name]
                # s2t_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
                loss_sum = sum(log.get("loss", 0) for log in s2t_logging_output)
                nll_loss_sum = sum(log.get("nll_loss", 0) for log in s2t_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in s2t_logging_output)
                ce_loss_sum = sum(log.get("ce_loss", 0) for log in s2t_logging_output)
                ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in s2t_logging_output)

                sample_size = max(1, sum(log.get("sample_size", 0) for log in s2t_logging_output))
                metrics.log_scalar(
                    "s2t_loss", loss_sum / sample_size / math.log(2), sample_size, 1, round=3
                )

                metrics.log_scalar(
                    "s2t_nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, 2, round=3
                )
                metrics.log_derived(
                    "s2t_ppl", lambda meters: utils.get_perplexity(meters["s2t_nll_loss"].avg, 2)
                )
                metrics.log_scalar(
                    "ctc_loss", ctc_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                metrics.log_scalar(
                    "ce_loss", ce_loss_sum / ntokens, ntokens, 2, round=3
                )

                total = utils.item(sum(log.get("total", 0) for log in s2t_logging_output))
                if total > 0:
                    metrics.log_scalar("s2t_total", total)
                    n_correct = utils.item(
                        sum(log.get("n_correct", 0) for log in s2t_logging_output)
                    )
                    metrics.log_scalar("s2t_n_correct", n_correct)
                    metrics.log_derived(
                        "s2t_accuracy",
                        lambda meters: round(
                            meters["s2t_n_correct"].sum * 100.0 / meters["s2t_total"].sum, 3
                        )
                        if meters["s2t_total"].sum > 0
                        else float("nan"),
                        2
                    )
                c_errors = sum(log.get("c_errors", 0) for log in s2t_logging_output)
                metrics.log_scalar("_c_errors", c_errors)
                c_total = sum(log.get("c_total", 0) for log in s2t_logging_output)
                metrics.log_scalar("_c_total", c_total)
                w_errors = sum(log.get("w_errors", 0) for log in s2t_logging_output)
                metrics.log_scalar("_w_errors", w_errors)
                wv_errors = sum(log.get("wv_errors", 0) for log in s2t_logging_output)
                metrics.log_scalar("_wv_errors", wv_errors)
                w_total = sum(log.get("w_total", 0) for log in s2t_logging_output)
                metrics.log_scalar("_w_total", w_total)
                if c_total > 0:
                    metrics.log_derived(
                        "uer",
                        lambda meters: safe_round(
                            meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                        )
                        if meters["_c_total"].sum > 0
                        else float("nan"),
                    )
                if w_total > 0:
                    metrics.log_derived(
                        "wer",
                        lambda meters: safe_round(
                            meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                        )
                        if meters["_w_total"].sum > 0
                        else float("nan"),
                    )
                    metrics.log_derived(
                        "raw_wer",
                        lambda meters: safe_round(
                            meters["_wv_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                        )
                        if meters["_w_total"].sum > 0
                        else float("nan"),
                    )

            if task_name == 't2s':
                # TTSLossCriterion.reduce_metrics([logging_output['t2s'] for logging_output in logging_outputs])
                # t2s_sum = sum(log.get("speech_loss", 0) for log in logging_outputs)
                t2s_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in t2s_logging_output)
                l1_loss_sum = sum(log.get("l1_loss", 0) for log in t2s_logging_output)
                l2_loss_sum = sum(log.get("l2_loss", 0) for log in t2s_logging_output)
                bce_loss_sum = sum(log.get("bce_loss", 0) for log in t2s_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in t2s_logging_output))
                metrics.log_scalar(
                    "t2s_loss", loss_sum / sample_size, sample_size, 1, round=5
                )
                encoder_alpha_sum = sum(log.get("encoder_alpha", 0) for log in t2s_logging_output)
                decoder_alpha_sum = sum(log.get("decoder_alpha", 0) for log in t2s_logging_output)
                ngpu = sum(log.get("ngpu", 0) for log in t2s_logging_output)

                metrics.log_scalar(
                    "t2s_l1_loss", l1_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "t2s_l2_loss", l2_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "t2s_bce_loss", bce_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "t2s_encoder_alpha", encoder_alpha_sum / sample_size, sample_size, round=5
                )
                metrics.log_scalar(
                    "t2s_decoder_alpha", decoder_alpha_sum / sample_size, sample_size, round=5
                )

                if "enc_dec_attn_loss" in t2s_logging_output[0]:
                    enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in t2s_logging_output)
                    metrics.log_scalar(
                        "t2s_enc_dec_attn_loss", enc_dec_attn_loss_sum / sample_size, sample_size, round=8
                    )

            if task_name == 's2c':
                s2c_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in s2c_logging_output)
                nll_loss_sum = sum(log.get("nll_loss", 0) for log in s2c_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in s2c_logging_output)

                sample_size = max(1, sum(log.get("sample_size", 0) for log in s2c_logging_output))
                metrics.log_scalar(
                    "s2c_loss", loss_sum / sample_size / math.log(2), sample_size, 1, round=3
                )

                metrics.log_scalar(
                    "s2c_nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, 2, round=3
                )

                total = utils.item(sum(log.get("total", 0) for log in s2c_logging_output)) 
                if total > 0:
                    metrics.log_scalar("s2c_total", total)
                    n_correct = utils.item(sum(log.get("n_correct", 0) for log in s2c_logging_output))
                    metrics.log_scalar("s2c_n_correct", n_correct)
                    metrics.log_derived(
                        "s2c_accuracy",
                        lambda meters: round(
                            meters["s2c_n_correct"].sum * 100.0 / meters["s2c_total"].sum, 3
                        )
                        if meters["s2c_total"].sum > 0
                        else float("nan"),
                        2
                    )
            
            if task_name == 's2s':
                s2s_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in s2s_logging_output)
                l1_loss_sum = sum(log.get("l1_loss", 0) for log in s2s_logging_output)
                l2_loss_sum = sum(log.get("l2_loss", 0) for log in s2s_logging_output)
                bce_loss_sum = sum(log.get("bce_loss", 0) for log in s2s_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in s2s_logging_output))
                metrics.log_scalar(
                    "s2s_loss", loss_sum / sample_size, sample_size, 1, round=5
                )
                encoder_alpha_sum = sum(log.get("encoder_alpha", 0) for log in s2s_logging_output)
                decoder_alpha_sum = sum(log.get("decoder_alpha", 0) for log in s2s_logging_output)
                ngpu = sum(log.get("ngpu", 0) for log in s2s_logging_output)

                metrics.log_scalar(
                    "s2s_l1_loss", l1_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "s2s_l2_loss", l2_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "s2s_bce_loss", bce_loss_sum / sample_size, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "s2s_decoder_alpha", decoder_alpha_sum / sample_size, sample_size, round=5
                )

                if "enc_dec_attn_loss" in s2s_logging_output[0]:
                    enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in s2s_logging_output)
                    metrics.log_scalar(
                        "s2s_enc_dec_attn_loss", enc_dec_attn_loss_sum / sample_size, sample_size, round=8
                    )

            if task_name == 'text_pretrain':
                bart_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in bart_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in bart_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in bart_logging_output))
                bart_loss_sum = sum(log.get("bart_loss", 0) for log in bart_logging_output)

                # we divide by log(2) to convert the loss from base e to base 2
                metrics.log_scalar(
                    "text_loss", loss_sum / sample_size / math.log(2), sample_size, round=3
                )
                metrics.log_scalar(
                    "bart_loss", bart_loss_sum / sample_size / math.log(2), ntokens, 2, round=3
                )
                if sample_size != ntokens:
                    metrics.log_scalar(
                        "bart_nll_loss", bart_loss_sum / ntokens / math.log(2), ntokens, round=3
                    )
                    metrics.log_derived(
                        "bart_ppl", lambda meters: utils.get_perplexity(meters["bart_nll_loss"].avg)
                    )
                else:
                    metrics.log_derived(
                        "bart_ppl", lambda meters: utils.get_perplexity(meters["bart_loss"].avg)
                    )
                metrics.log_scalar("bart_wpb", ntokens, priority=180, round=1)

                val_prob_perplexity = 0
                val_code_perplexity = 0
                sample_size_pp = 0
                count_log_cp = 0
                for log in bart_logging_output:
                    if "loss_prob_perplexity" in log:
                        val_prob_perplexity = val_prob_perplexity + log["loss_prob_perplexity"]
                        sample_size_pp = sample_size_pp + log["sample_size"]
                    if "code_perplexity" in log:
                        val_code_perplexity = val_code_perplexity + log["code_perplexity"]
                        count_log_cp = count_log_cp + 1
                if val_prob_perplexity > 0:
                    metrics.log_scalar("text_loss_prob_perplexity", val_prob_perplexity / sample_size_pp / math.log(2), round=3)
                if val_code_perplexity > 0:
                    metrics.log_scalar("text_code_perplexity", val_code_perplexity / count_log_cp, round=3)

            if task_name == 'speech_pretrain':
                hubert_logging_output = logging_outputs_dict[task_name]
                loss_sum = sum(log.get("loss", 0) for log in hubert_logging_output)
                ntokens = sum(log.get("ntokens", 0) for log in hubert_logging_output)
                sample_size = max(1, sum(log.get("sample_size", 0) for log in hubert_logging_output))
                dec_loss_sum = sum(log.get("dec_loss", 0) for log in hubert_logging_output)
                l1_loss_sum = sum(log.get("l1_loss", 0) for log in hubert_logging_output)
                l2_loss_sum = sum(log.get("l2_loss", 0) for log in hubert_logging_output)
                bce_loss_sum = sum(log.get("bce_loss", 0) for log in hubert_logging_output)
                ngpu = sum(log.get("ngpu", 0) for log in hubert_logging_output)

                metrics.log_scalar("hubert_loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
                if sample_size != ntokens:
                    metrics.log_scalar("hubert_nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3)
                    metrics.log_derived("hubert_ppl", lambda meters: utils.get_perplexity(meters["hubert_nll_loss"].avg))
                else:
                    metrics.log_derived("hubert_ppl", lambda meters: utils.get_perplexity(meters["hubert_loss"].avg))

                counts = {}
                for lk in hubert_logging_output[0].keys():
                    if lk.startswith("count_"):
                        val = sum(log[lk] for log in hubert_logging_output)
                        metrics.log_scalar("hubert_" + lk, val)
                        counts[lk] = val

                for lk in hubert_logging_output[0].keys():
                    if lk.startswith("loss_") and lk != 'loss_prob_perplexity':
                        val = sum(log[lk] for log in hubert_logging_output)
                        metrics.log_scalar("hubert_" + lk, val / sample_size / math.log(2), round=3)
                    elif lk.startswith("correct_"):
                        val = sum(log[lk] for log in hubert_logging_output)
                        metrics.log_scalar("hubert_" + lk, val / counts[re.sub("correct", "count", lk)])
                    # elif lk == 'code_perplexity':
                    #     val = sum(log[lk] for log in hubert_logging_output)
                    #     metrics.log_scalar("hubert_" + lk, val / len(hubert_logging_output), round=3)

                val_prob_perplexity = 0
                val_code_perplexity = 0
                sample_size_pp = 0
                count_log_cp = 0
                for log in hubert_logging_output:
                    if "loss_prob_perplexity" in log:
                        val_prob_perplexity = val_prob_perplexity + log["loss_prob_perplexity"]
                        sample_size_pp = sample_size_pp + log["sample_size"]
                    if "code_perplexity" in log:
                        val_code_perplexity = val_code_perplexity + log["code_perplexity"]
                        count_log_cp = count_log_cp + 1
                if val_prob_perplexity > 0:
                    metrics.log_scalar("hubert_loss_prob_perplexity", val_prob_perplexity / sample_size_pp / math.log(2), round=3)
                if val_code_perplexity > 0:
                    metrics.log_scalar("hubert_code_perplexity", val_code_perplexity / count_log_cp, round=3)

                metrics.log_scalar(
                    "hubert_dec_loss", dec_loss_sum / ngpu, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "hubert_l1_loss", l1_loss_sum / ngpu, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "hubert_l2_loss", l2_loss_sum / ngpu, sample_size, 2, round=5
                )
                metrics.log_scalar(
                    "hubert_bce_loss", bce_loss_sum / ngpu, sample_size, 2, round=5
                )
                if "enc_dec_attn_loss" in hubert_logging_output[0]:
                    enc_dec_attn_loss_sum = sum(log.get("enc_dec_attn_loss", 0) for log in hubert_logging_output)
                    metrics.log_scalar(
                        "hubert_enc_dec_attn_loss", enc_dec_attn_loss_sum / ngpu, sample_size, round=8
                    )
                metrics.log_scalar("hubert_wpb", ntokens, priority=180, round=1)

        loss = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = max(1, sum(log.get("sample_size", 0) for log in logging_outputs))
        metrics.log_scalar(
            "loss", loss / sample_size, sample_size, 1, round=5
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
