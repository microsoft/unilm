from dataclasses import dataclass, field
import math
from omegaconf import II

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

LOSS_NAMES = ["gpt", "image_interleaved", "image_laion", "image_tune", "gpt_tune"]

@dataclass
class UniGPTLossConfig(FairseqDataclass):
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "ignore mlm output at eos token."},
    )
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion(
    "unigpt", dataclass=UniGPTLossConfig
)
class UniGPTLoss(FairseqCriterion):
    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg

    def forward(self, model, sample, reduce=True, loss_name="gpt"):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (sample["target"][sample["net_input"]['gpt_loss_mask']] != self.padding_idx).sum().int()
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        # for loss_name_item in LOSS_NAMES:
        #     logging_output[loss_name_item] = 0.01
        #     logging_output[loss_name_item + "sample_size"] = 1
        logging_output[loss_name] = loss.data
        logging_output[loss_name + "sample_size"] = sample["ntokens"]
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        if hasattr(model, "gpt_model"):
            lprobs = model.gpt_model.get_normalized_probs(net_output, log_probs=True)
        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
        loss_mask = sample["net_input"]['gpt_loss_mask']

        lprobs = lprobs[loss_mask]
        target = model.get_targets(sample, net_output)[loss_mask].view(-1)
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
        
        loss_items = []
        # log individual losses
        for loss_name in LOSS_NAMES:
            loss_sum = sum(log.get(loss_name, 0) for log in logging_outputs)
            single_sample_size = sum(log.get(loss_name + "sample_size", 0) for log in logging_outputs)

            if loss_sum != 0:
                metrics.log_scalar(
                    loss_name, loss_sum / single_sample_size / math.log(2), single_sample_size, round=3
                )
                metrics.log_scalar(
                    loss_name + "_sample_size", single_sample_size, round=3
                )
                loss_items.append(loss_sum / single_sample_size / math.log(2))
            else:
                metrics.log_scalar(
                    loss_name + "_sample_size", 0, round=3
                )
        metrics.log_scalar(
            "loss", sum(loss_items) / len(loss_items), sample_size, round=3
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
