import logging
import numpy as np

from typing import Dict
from transformers.utils.logging import enable_explicit_format
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    PreTrainedTokenizerFast,
    EvalPrediction,
)

from logger_config import logger, LoggerCallback
from config import Arguments
from loaders import ReplaceLMDataloader
from collators import DataCollatorForReplaceLM
from trainers import ReplaceLMTrainer
from models import ReplaceLM


def _common_setup(args: Arguments):
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
    enable_explicit_format()
    set_seed(args.seed)


def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds = eval_pred.predictions

    avg_enc_mlm_loss = float(np.mean(preds[0]))
    avg_dec_mlm_loss = float(np.mean(preds[1]))
    avg_g_mlm_loss = float(np.mean(preds[2]))
    avg_replace_ratio = float(np.mean(preds[3]))

    return {'avg_enc_mlm_loss': round(avg_enc_mlm_loss, 4),
            'avg_dec_mlm_loss': round(avg_dec_mlm_loss, 4),
            'avg_g_mlm_loss': round(avg_g_mlm_loss, 4),
            'avg_replace_ratio': round(avg_replace_ratio, 4)}


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: ReplaceLM = ReplaceLM.from_pretrained(
        all_args=args, model_name_or_path=args.model_name_or_path)
    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))

    dataloader = ReplaceLMDataloader(args=args, tokenizer=tokenizer)
    train_dataset, eval_dataset = dataloader.train_dataset, dataloader.eval_dataset

    data_collator = DataCollatorForReplaceLM(
        tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
        args=args,
    )

    trainer: ReplaceLMTrainer = ReplaceLMTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)

    model.trainer = trainer

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return


if __name__ == "__main__":
    main()
