import random

from typing import Tuple
from transformers import PreTrainedTokenizerFast
from datasets import Dataset, load_dataset

from config import Arguments
from logger_config import logger


def split_dataset(dataset: Dataset,
                  num_eval_examples: int,
                  max_train_samples: int = None) -> Tuple[Dataset, Dataset]:
    indices = list(range(len(dataset)))
    random.Random(123).shuffle(indices)
    eval_dataset = dataset.select(indices[:num_eval_examples])
    train_dataset = dataset.select(indices[num_eval_examples:])

    if max_train_samples is not None:
        train_dataset = train_dataset.select(range(max_train_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    return train_dataset, eval_dataset


class ReplaceLMDataloader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.tokenizer = tokenizer
        data_files = args.train_file.strip().split(',')
        self.corpus: Dataset = load_dataset('json', data_files=data_files)['train']
        self.train_dataset, self.eval_dataset = split_dataset(
            self.corpus,
            num_eval_examples=args.rlm_num_eval_samples,
            max_train_samples=args.max_train_samples)
