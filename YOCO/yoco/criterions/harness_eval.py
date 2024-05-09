import torch
import torch.nn.functional as F

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@register_criterion("harness_eval", dataclass=FairseqDataclass)
class HarnessEvalCriterion(FairseqCriterion):
    def __init__(self, cfg, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model.eval()
        net_output, _ = model(sample["net_input"]["src_tokens"])
        net_output = net_output[:, :-1, :]
        targets = sample["net_input"]["src_tokens"][:, 1:]
        loss_mask = sample["net_input"]["gpt_loss_mask"][:, 1:]
        label_length = sample["net_input"]["label_length"]
        loss = F.cross_entropy(
            net_output.float().reshape(-1, net_output.size(-1)),
            targets.reshape(-1),
            reduction="none",
            ignore_index=self.padding_idx,
        ).reshape(targets.size(0), -1)
        loss = loss * loss_mask.int()
        loss_norm = loss.sum(-1) / label_length.float()
        loss = loss.sum(-1)

        option_num = self.task.harness_task.class_num
        labels = sample["targets"].view(-1)
        
        assert sample["targets"].size(0) % option_num == 0
        sample_size = sample["ntokens"]

        pred_label = torch.argmin(loss.view(-1, option_num), dim=1)
        pred_norm_label = torch.argmin(loss_norm.view(-1, option_num), dim=1)
        target_label = labels.view(-1, option_num)[:, 0]

        logging_output = {}

        logging_output.update(
            {
                "loss": 0,
                "nsentences": pred_label.size(0),
                "sample_size": pred_label.size(0),
                "ncorrect": (pred_label == target_label).sum().item(),
                "ncorrect_norm": (pred_norm_label == target_label).sum().item(),
            }
        )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
        ncorrect_norm = sum(log.get("ncorrect_norm", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss / nsentences, nsentences, round=3
        )
        metrics.log_scalar(
            "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=2
        )
        metrics.log_scalar(
            "accuracy_norm", 100.0 * ncorrect_norm / nsentences, nsentences, round=2
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True