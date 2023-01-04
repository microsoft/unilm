import os

from typing import Optional
from transformers.trainer import Trainer

from logger_config import logger
from models import ReplaceLM, ReplaceLMOutput
from utils import AverageMeter


class ReplaceLMTrainer(Trainer):
    def __init__(self, *pargs, **kwargs):
        super(ReplaceLMTrainer, self).__init__(*pargs, **kwargs)
        self.model: ReplaceLM

        self.enc_mlm_loss = AverageMeter('enc_mlm_loss', round_digits=3)
        self.dec_mlm_loss = AverageMeter('dec_mlm_loss', round_digits=3)
        self.g_mlm_loss = AverageMeter('g_mlm_loss', round_digits=3)
        self.replace_ratio = AverageMeter('replace_ratio', round_digits=3)
        self.last_epoch = 0

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to {}".format(output_dir))
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs: ReplaceLMOutput = model(model_input=inputs)
        loss = outputs.loss

        if self.model.training:
            self.enc_mlm_loss.update(outputs.encoder_mlm_loss.item())
            self.dec_mlm_loss.update(outputs.decoder_mlm_loss.item())
            self.g_mlm_loss.update(outputs.g_mlm_loss.item())
            self.replace_ratio.update(outputs.replace_ratio.item())
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
                log_info = ', '.join(map(str, [self.enc_mlm_loss, self.dec_mlm_loss, self.g_mlm_loss, self.replace_ratio]))
                logger.info('step: {}, {}'.format(self.state.global_step, log_info))

            self._reset_meters_if_needed()

        return (loss, outputs) if return_outputs else loss

    def _reset_meters_if_needed(self):
        if int(self.state.epoch) != self.last_epoch:
            self.last_epoch = int(self.state.epoch)
            self.enc_mlm_loss.reset()
            self.dec_mlm_loss.reset()
            self.g_mlm_loss.reset()
            self.replace_ratio.reset()
