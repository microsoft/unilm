import os.path
import random

from typing import Tuple, Dict, List, Optional
from datasets import load_dataset, DatasetDict, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from .loader_utils import group_doc_ids


class CrossEncoderDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.negative_size = args.train_n_passages - 1
        assert self.negative_size > 0
        self.tokenizer = tokenizer
        corpus_path = os.path.join(args.data_dir, 'passages.jsonl.gz')
        self.corpus: Dataset = load_dataset('json', data_files=corpus_path)['train']
        self.train_dataset, self.eval_dataset = self._get_transformed_datasets()

        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def _transform_func(self, examples: Dict[str, List]) -> Dict[str, List]:
        current_epoch = int(self.trainer.state.epoch or 0)

        input_doc_ids = group_doc_ids(
            examples=examples,
            negative_size=self.negative_size,
            offset=current_epoch + self.args.seed,
            use_first_positive=self.args.use_first_positive
        )
        assert len(input_doc_ids) == len(examples['query']) * self.args.train_n_passages

        input_queries, input_docs = [], []
        for idx, doc_id in enumerate(input_doc_ids):
            prefix = ''
            if self.corpus[doc_id].get('title', ''):
                prefix = self.corpus[doc_id]['title'] + ': '

            input_docs.append(prefix + self.corpus[doc_id]['contents'])
            input_queries.append(examples['query'][idx // self.args.train_n_passages])

        batch_dict = self.tokenizer(input_queries,
                                    text_pair=input_docs,
                                    max_length=self.args.rerank_max_length,
                                    padding=PaddingStrategy.DO_NOT_PAD,
                                    truncation=True)

        packed_batch_dict = {}
        for k in batch_dict:
            packed_batch_dict[k] = []
            assert len(examples['query']) * self.args.train_n_passages == len(batch_dict[k])
            for idx in range(len(examples['query'])):
                start = idx * self.args.train_n_passages
                packed_batch_dict[k].append(batch_dict[k][start:(start + self.args.train_n_passages)])

        return packed_batch_dict

    def _get_transformed_datasets(self) -> Tuple:
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file.split(',')
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        raw_datasets: DatasetDict = load_dataset('json', data_files=data_files)

        train_dataset, eval_dataset = None, None

        if self.args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]
            if self.args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.args.max_train_samples))
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            train_dataset.set_transform(self._transform_func)

        if self.args.do_eval:
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            eval_dataset.set_transform(self._transform_func)

        return train_dataset, eval_dataset
