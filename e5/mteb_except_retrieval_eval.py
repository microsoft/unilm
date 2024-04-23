import os
import torch
import torch.nn.functional as F
import tqdm
import json
import numpy as np
import argparse

from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from typing import List
from mteb import MTEB

from utils import logger, pool, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, create_batch_dict
from model_config import MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE

parser = argparse.ArgumentParser(description='evaluation for MTEB benchmark except its Retrieval category')
parser.add_argument('--task-types', nargs='+', default=[], help='task types to evaluate')
parser.add_argument('--output-dir', default='',
                    type=str, metavar='N', help='output directory')
parser.add_argument('--model-name-or-path', default='tmp-outputs/',
                    type=str, metavar='N', help='which model to use')
parser.add_argument('--pool-type', default='avg', help='pool type')
parser.add_argument('--prefix-type', default='query_or_passage', help='prefix type')
parser.add_argument('--multilingual', action='store_true', help='whether to use multilingual model')
parser.add_argument('--dry-run', action='store_true', help='whether to run the script in dry run mode')

args = parser.parse_args()
base_name: str = args.model_name_or_path.split('/')[-1]
args.pool_type = MODEL_NAME_TO_POOL_TYPE.get(base_name, args.pool_type)
args.prefix_type = MODEL_NAME_TO_PREFIX_TYPE.get(base_name, args.prefix_type)

logger.info('Args: {}'.format(json.dumps(args.__dict__, ensure_ascii=False, indent=4)))
assert args.pool_type in ['cls', 'avg', 'last', 'weightedavg'], 'pool_type should be cls / avg / last'
assert args.prefix_type in ['query_or_passage', 'instruction'], 'prefix_type should be query_or_passage / instruction'
os.makedirs(args.output_dir, exist_ok=True)


class DenseEncoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.l2_normalize = True
        self.prompt = None
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

        input_texts: List[str] = [self.prompt + s for s in sentences]

        encoded_embeds = []
        batch_size = 64 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], args.pool_type)
                if self.l2_normalize:
                    embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt


def main():
    model = DenseEncoder()
    args.task_types = [t for t in args.task_types if t.strip()]
    evaluation = MTEB(
        task_types=args.task_types or None,
        task_langs=['en'] if not args.multilingual else None
    )

    for task_cls in evaluation.tasks:
        task_name: str = task_cls.description['name']
        task_type: str = task_cls.description['type']
        if args.dry_run and task_name not in ['Banking77Classification', 'ImdbClassification', 'STS12']:
            continue

        if args.prefix_type == 'query_or_passage':
            prompt: str = 'query: '
        else:
            task_def: str = get_task_def_by_task_name_and_type(task_name=task_name, task_type=task_type)
            prompt: str = get_detailed_instruct(task_def)
        model.set_prompt(prompt=prompt)
        logger.info('Set prompt: {}'.format(prompt))

        # disable l2 normalize for classification tasks, as it achieves slightly better results
        if task_type == 'Classification':
            logger.info('Set l2_normalize to False for classification task')
            model.l2_normalize = False
        else:
            model.l2_normalize = True
            logger.info('Set l2_normalize to {}'.format(model.l2_normalize))

        sub_eval = MTEB(tasks=[task_name], task_langs=['en'] if not args.multilingual else None)
        logger.info('Running evaluation for task: {}, type: {}'.format(task_name, task_type))
        eval_splits = ["test"] if "test" in task_cls.description["eval_splits"] else task_cls.description["eval_splits"]
        sub_eval.run(
            model, eval_splits=eval_splits,
            output_folder=args.output_dir
        )


if __name__ == '__main__':
    main()
