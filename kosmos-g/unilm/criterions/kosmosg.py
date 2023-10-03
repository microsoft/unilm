from dataclasses import dataclass, field

from omegaconf import II

from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class KosmosGDiffConfig(FairseqDataclass):
    ignore_eos: bool = field(
        default=False,
        metadata={"help": "ignore mlm output at eos token."},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    data_weights: str = field(
        default="",
        metadata={"help": "dataset weights"}
    )
    align: bool = field(
        default=False,
        metadata={"help": "use clip supervision"},
    )


@register_criterion(
    "kosmosg", dataclass=KosmosGDiffConfig
)
class KosmosGLoss(FairseqCriterion):
    def __init__(self, cfg, task):
        super().__init__(task)
        self.cfg = cfg
        global LOSS_NAMES
        global SEPERATED_LOSS
        LOSS_NAMES = ["image_laion", "image_instructpix2pix", "image_openimage"]
        LOSS_NAMES = [loss_name for i, loss_name in enumerate(LOSS_NAMES) if float(cfg.data_weights.split(',')[i]) > 0]
        SEPERATED_LOSS = ["diff_loss", "mse_loss", "rec_loss"]

    def forward(self, model, sample, reduce=True, loss_name="image_laion"):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        raw_loss = net_output[1]['loss']

        ntokens = sample["ntokens"]
        nsentences = sample["nsentences"]

        logging_output = {
            "ntokens": ntokens,
            "nsentences": nsentences
        }

        if 'diff_loss' in raw_loss:
            loss = raw_loss['diff_loss']
            sample_size = nsentences
            logging_output['diff_loss'] = loss.data

        elif 'clip_loss' in raw_loss:
            loss = raw_loss['clip_loss']['mse_loss'] + raw_loss['clip_loss']['rec_loss']
            sample_size = 1
            for k, v in raw_loss['clip_loss'].items():
                logging_output[k] = v.data

        else:
            raise NotImplementedError

        logging_output['loss'] = loss.data
        logging_output[loss_name] = loss.data
        logging_output["sample_size"] = sample_size
        logging_output[loss_name + "_sample_size"] = logging_output["sample_size"]
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_items = []
        seperated_loss_items = {k: [] for k in SEPERATED_LOSS}
        # log individual losses
        for loss_name in LOSS_NAMES:
            num_log = sum(1 for log in logging_outputs if loss_name in log)
            loss_sum = sum(log.get(loss_name, 0) for log in logging_outputs)
            single_sample_size = sum(log.get(loss_name + "_sample_size", 0) for log in logging_outputs)
            seperated_loss_sum = {
                k: sum(log.get(k, 0) for log in logging_outputs if loss_name in log) for k in SEPERATED_LOSS
            }

            if loss_sum != 0:
                metrics.log_scalar(
                    loss_name, loss_sum / num_log, num_log, round=3
                )
                metrics.log_scalar(
                    loss_name + "_sample_size", single_sample_size, round=3
                )

                loss_items.append(loss_sum / num_log)

                for k in SEPERATED_LOSS:
                    if seperated_loss_sum[k] != 0:
                        metrics.log_scalar(
                            loss_name + "_" + k, seperated_loss_sum[k] / single_sample_size, single_sample_size,
                            round=3
                        )
                        seperated_loss_items[k].append(seperated_loss_sum[k] / single_sample_size)

            else:
                metrics.log_scalar(
                    loss_name, 0, round=3
                )
                metrics.log_scalar(
                    loss_name + "_sample_size", 0, round=3
                )
                for k in SEPERATED_LOSS:
                    is_k_exist = [k in log for log in logging_outputs]
                    if any(is_k_exist):
                        metrics.log_scalar(
                            loss_name + "_" + k, 0, round=3
                        )

        metrics.log_scalar(
            "loss", sum(loss_items) / len(loss_items), num_log, round=3
        )

        for k in SEPERATED_LOSS:
            if len(seperated_loss_items[k]) > 0:
                metrics.log_scalar(
                    k, sum(seperated_loss_items[k]) / len(seperated_loss_items[k]), num_log, round=3
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
