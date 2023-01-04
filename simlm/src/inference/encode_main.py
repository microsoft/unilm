import os
import tqdm
import torch

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from datasets import load_dataset
from typing import Dict, List
from transformers.file_utils import PaddingStrategy
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    HfArgumentParser,
    BatchEncoding
)

from config import Arguments
from logger_config import logger
from utils import move_to_cuda
from models import BiencoderModelForInference, BiencoderOutput

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]


def _psg_transform_func(tokenizer: PreTrainedTokenizerFast,
                        examples: Dict[str, List]) -> BatchEncoding:
    batch_dict = tokenizer(examples['title'],
                           text_pair=examples['contents'],
                           max_length=args.p_max_len,
                           padding=PaddingStrategy.DO_NOT_PAD,
                           truncation=True)
    # for co-Condenser reproduction purpose only
    if args.model_name_or_path.startswith('Luyu/'):
        del batch_dict['token_type_ids']

    return batch_dict


@torch.no_grad()
def _worker_encode_passages(gpu_idx: int):
    def _get_out_path(shard_idx: int = 0) -> str:
        return '{}/shard_{}_{}'.format(args.encode_save_dir, gpu_idx, shard_idx)

    if os.path.exists(_get_out_path(0)):
        logger.error('{} already exists, will skip encoding'.format(_get_out_path(0)))
        return

    dataset = load_dataset('json', data_files=args.encode_in_path)['train']
    if args.dry_run:
        dataset = dataset.select(range(4096))
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: BiencoderModelForInference = BiencoderModelForInference.build(args)
    model.eval()
    model.cuda()

    dataset.set_transform(partial(_psg_transform_func, tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    data_loader = DataLoader(
        dataset,
        batch_size=args.encode_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    num_encoded_docs, encoded_embeds, cur_shard_idx = 0, [], 0
    for batch_dict in tqdm.tqdm(data_loader, desc='passage encoding', mininterval=8):
        batch_dict = move_to_cuda(batch_dict)

        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            outputs: BiencoderOutput = model(query=None, passage=batch_dict)
        encoded_embeds.append(outputs.p_reps.cpu())
        num_encoded_docs += outputs.p_reps.shape[0]

        if num_encoded_docs >= args.encode_shard_size:
            out_path = _get_out_path(cur_shard_idx)
            concat_embeds = torch.cat(encoded_embeds, dim=0)
            logger.info('GPU {} save {} embeds to {}'.format(gpu_idx, concat_embeds.shape[0], out_path))
            torch.save(concat_embeds, out_path)

            cur_shard_idx += 1
            num_encoded_docs = 0
            encoded_embeds.clear()

    if num_encoded_docs > 0:
        out_path = _get_out_path(cur_shard_idx)
        concat_embeds = torch.cat(encoded_embeds, dim=0)
        logger.info('GPU {} save {} embeds to {}'.format(gpu_idx, concat_embeds.shape[0], out_path))
        torch.save(concat_embeds, out_path)

    logger.info('Done computing score for worker {}'.format(gpu_idx))


def _batch_encode_passages():
    logger.info('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error('No gpu available')
        return

    logger.info('Use {} gpus'.format(gpu_count))
    torch.multiprocessing.spawn(_worker_encode_passages, args=(), nprocs=gpu_count)
    logger.info('Done batch encode passages')


if __name__ == '__main__':
    _batch_encode_passages()
