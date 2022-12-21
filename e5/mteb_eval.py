import os
import torch
import torch.nn.functional as F
import tqdm
import json
import numpy as np
import argparse

from functools import partial
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerFast, BatchEncoding
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Dict
from mteb import MTEB

from utils import logger, pool, move_to_cuda

parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
parser.add_argument('--task-types', nargs='+', default=['sts'], help='task types to evaluate')
parser.add_argument('--output-dir', default='',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--model-name-or-path', default='tmp-outputs/',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--l2-normalize', action='store_true', help='whether to l2 normalize embeddings')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prompt', default='query: ', help='prompt')

args = parser.parse_args()
logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.prompt in ['', 'query: ', 'passage: ']
assert args.output_dir, 'output_dir should be specified'
os.makedirs(args.output_dir, exist_ok=True)


def _transform_func(tokenizer: PreTrainedTokenizerFast,
                    examples: Dict[str, List]) -> BatchEncoding:
    if args.prompt:
        examples['input_texts'] = [args.prompt + t for t in examples['input_texts']]
    batch_dict = tokenizer(examples['input_texts'],
                           max_length=512,
                           padding=True,
                           truncation=True)

    return batch_dict


class DenseEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.gpu_count = torch.cuda.device_count()

        self.encoder.eval()
        self.encoder.cuda()

        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

    @torch.no_grad()
    def encode(self, sentences, **kwargs) -> np.ndarray:
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """

        dataset: Dataset = Dataset.from_dict({'input_texts': sentences})
        dataset.set_transform(partial(_transform_func, self.tokenizer))

        data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        data_loader = DataLoader(
            dataset,
            batch_size=128 * self.gpu_count,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            collate_fn=data_collator,
            pin_memory=True)

        encoded_embeds = []
        for batch_dict in tqdm.tqdm(data_loader, desc='encoding', mininterval=10, disable=len(sentences) < 128):
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                if args.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)


def main():
    model = DenseEncoder()
    args.task_types = [t for t in args.task_types if t.strip()]
    evaluation = MTEB(
        task_types=args.task_types or None,
        task_langs=['en'])
    evaluation.run(model, eval_splits=["test"],
                   output_folder=args.output_dir,
                   overwrite_results=False)


if __name__ == '__main__':
    main()
