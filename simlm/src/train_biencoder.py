import logging

import torch
from typing import Dict
from functools import partial
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
from trainers import BiencoderTrainer
from loaders import RetrievalDataLoader
from collators import BiencoderCollator
from metrics import accuracy, batch_mrr
from models import BiencoderModel


def _common_setup(args: Arguments):
    if args.process_index > 0:
        logger.setLevel(logging.WARNING)
    enable_explicit_format()
    set_seed(args.seed)


def _compute_metrics(args: Arguments, eval_pred: EvalPrediction) -> Dict[str, float]:
    # field consistent with BiencoderOutput
    preds = eval_pred.predictions
    scores = torch.tensor(preds[-1]).float()
    labels = torch.arange(0, scores.shape[0], dtype=torch.long) * args.train_n_passages
    labels = labels % scores.shape[1]

    topk_metrics = accuracy(output=scores, target=labels, topk=(1, 3))
    mrr = batch_mrr(output=scores, target=labels)

    return {'mrr': mrr, 'acc1': topk_metrics[0], 'acc3': topk_metrics[1]}


def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    logger.info('Args={}'.format(str(args)))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: BiencoderModel = BiencoderModel.build(args=args)
    logger.info(model)
    logger.info('Vocab size: {}'.format(len(tokenizer)))

    data_collator = BiencoderCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None)

    retrieval_data_loader = RetrievalDataLoader(args=args, tokenizer=tokenizer)
    train_dataset = retrieval_data_loader.train_dataset
    eval_dataset = retrieval_data_loader.eval_dataset

    trainer: Trainer = BiencoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        data_collator=data_collator,
        compute_metrics=partial(_compute_metrics, args),
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    retrieval_data_loader.trainer = trainer
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
        metrics = trainer.evaluate(metric_key_prefix="eval")
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return


if __name__ == "__main__":
    main()
