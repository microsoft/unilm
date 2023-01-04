import logging

import torch
from typing import Dict
from transformers.utils.logging import enable_explicit_format
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    EvalPrediction,
    Trainer,
    set_seed,
    PreTrainedTokenizerFast
)

from logger_config import logger, LoggerCallback
from config import Arguments
from trainers.reranker_trainer import RerankerTrainer
from loaders import CrossEncoderDataLoader
from collators import CrossEncoderCollator
from metrics import accuracy
from models import Reranker


def _common_setup(args: Arguments):
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
    enable_explicit_format()
    set_seed(args.seed)


def _compute_metrics(eval_pred: EvalPrediction) -> Dict:
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[-1]
    logits = torch.tensor(preds).float()
    labels = torch.tensor(eval_pred.label_ids).long()
    acc = accuracy(output=logits, target=labels)[0]

    return {'acc': acc}


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)

    model: Reranker = Reranker.from_pretrained(
        all_args=args,
        pretrained_model_name_or_path=args.model_name_or_path,
        num_labels=1)

    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))

    data_collator = CrossEncoderCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None)

    rerank_data_loader = CrossEncoderDataLoader(args=args, tokenizer=tokenizer)
    train_dataset = rerank_data_loader.train_dataset
    eval_dataset = rerank_data_loader.eval_dataset

    trainer: Trainer = RerankerTrainer(
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
    rerank_data_loader.trainer = trainer

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return


if __name__ == "__main__":
    main()
