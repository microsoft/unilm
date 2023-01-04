import os
import torch

from typing import Optional, Dict, Tuple
from transformers.trainer import Trainer

from logger_config import logger
from metrics import accuracy, batch_mrr
from models import BiencoderOutput, BiencoderModel
from utils import AverageMeter


def _unpack_qp(inputs: Dict[str, torch.Tensor]) -> Tuple:
    q_prefix, d_prefix, kd_labels_key = 'q_', 'd_', 'kd_labels'
    query_batch_dict = {k[len(q_prefix):]: v for k, v in inputs.items() if k.startswith(q_prefix)}
    doc_batch_dict = {k[len(d_prefix):]: v for k, v in inputs.items() if k.startswith(d_prefix)}

    if kd_labels_key in inputs:
        assert len(query_batch_dict) > 0
        query_batch_dict[kd_labels_key] = inputs[kd_labels_key]

    if not query_batch_dict:
        query_batch_dict = None
    if not doc_batch_dict:
        doc_batch_dict = None

    return query_batch_dict, doc_batch_dict


class BiencoderTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(BiencoderTrainer, self).__init__(*pargs, **kwargs)
        self.model: BiencoderModel

        self.acc1_meter = AverageMeter('Acc@1', round_digits=2)
        self.acc3_meter = AverageMeter('Acc@3', round_digits=2)
        self.mrr_meter = AverageMeter('mrr', round_digits=2)
        self.last_epoch = 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False):
        query, passage = _unpack_qp(inputs)
        outputs: BiencoderOutput = model(query=query, passage=passage)
        loss = outputs.loss

        if self.model.training:
            step_acc1, step_acc3 = accuracy(output=outputs.scores.detach(), target=outputs.labels, topk=(1, 3))
            step_mrr = batch_mrr(output=outputs.scores.detach(), target=outputs.labels)

            self.acc1_meter.update(step_acc1)
            self.acc3_meter.update(step_acc3)
            self.mrr_meter.update(step_mrr)

            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                log_info = ', '.join(map(str, [self.mrr_meter, self.acc1_meter, self.acc3_meter]))
                logger.info('step: {}, {}'.format(self.state.global_step, log_info))

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.acc1_meter.reset()
            self.acc3_meter.reset()
            self.mrr_meter.reset()
