import json
import os
import glob
import tqdm
import torch

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from collections import defaultdict
from datasets import Dataset
from typing import Dict, List, Tuple
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
from utils import move_to_cuda, save_json_to_file
from metrics import compute_mrr, trec_eval, ScoredDoc
from data_utils import load_queries, load_qrels, load_msmarco_predictions, save_preds_to_msmarco_format
from models import BiencoderModelForInference, BiencoderOutput

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]
assert os.path.exists(args.encode_save_dir)


def _get_all_shards_path() -> List[str]:
    path_list = glob.glob('{}/shard_*_*'.format(args.encode_save_dir))
    assert len(path_list) > 0

    def _parse_worker_idx_shard_idx(p: str) -> Tuple:
        worker_idx, shard_idx = [int(f) for f in os.path.basename(p).split('_')[-2:]]
        return worker_idx, shard_idx

    path_list = sorted(path_list, key=lambda path: _parse_worker_idx_shard_idx(path))
    logger.info('Embeddings path list: {}'.format(path_list))
    return path_list


def _get_topk_result_save_path(worker_idx: int) -> str:
    return '{}/top{}_{}_{}.txt'.format(args.search_out_dir, args.search_topk, args.search_split, worker_idx)


def _query_transform_func(tokenizer: PreTrainedTokenizerFast,
                          examples: Dict[str, List]) -> BatchEncoding:
    batch_dict = tokenizer(examples['query'],
                           max_length=args.q_max_len,
                           padding=PaddingStrategy.DO_NOT_PAD,
                           truncation=True)

    return batch_dict


@torch.no_grad()
def _worker_encode_queries(gpu_idx: int) -> Tuple:
    # fail fast if shard does not exist
    _get_all_shards_path()

    query_id_to_text = load_queries(path=os.path.join(args.data_dir, '{}_queries.tsv'.format(args.search_split)),
                                    task_type=args.task_type)
    query_ids = sorted(list(query_id_to_text.keys()))
    queries = [query_id_to_text[query_id] for query_id in query_ids]
    dataset = Dataset.from_dict({'query_id': query_ids,
                                 'query': queries})
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    # only keep data for current shard
    query_ids = dataset['query_id']
    query_id_to_text = {qid: query_id_to_text[qid] for qid in query_ids}

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: BiencoderModelForInference = BiencoderModelForInference.build(args)
    model.eval()
    model.cuda()

    dataset.set_transform(partial(_query_transform_func, tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    data_loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    encoded_embeds = []
    for batch_dict in tqdm.tqdm(data_loader, desc='query encoding', mininterval=5):
        batch_dict = move_to_cuda(batch_dict)

        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            outputs: BiencoderOutput = model(query=batch_dict, passage=None)
        encoded_embeds.append(outputs.q_reps)

    query_embeds = torch.cat(encoded_embeds, dim=0)
    logger.info('Done query encoding for worker {}'.format(gpu_idx))

    return query_embeds, query_ids, query_id_to_text


@torch.no_grad()
def _worker_batch_search(gpu_idx: int):
    embeds_path_list = _get_all_shards_path()

    query_embeds, query_ids, query_id_to_text = _worker_encode_queries(gpu_idx)
    assert query_embeds.shape[0] == len(query_ids), '{} != {}'.format(query_embeds.shape[0], len(query_ids))

    query_id_to_topk = defaultdict(list)
    psg_idx_offset = 0
    for shard_idx, shard_path in enumerate(embeds_path_list):
        shard_psg_embed = torch.load(shard_path, map_location=lambda storage, loc: storage).to(query_embeds.device)
        logger.info('Load {} passage embeddings from {}'.format(shard_psg_embed.shape[0], shard_path))

        for start in tqdm.tqdm(range(0, len(query_ids), args.search_batch_size),
                               desc="search shard {}".format(shard_idx),
                               mininterval=5):
            batch_query_embed = query_embeds[start:(start + args.search_batch_size)]
            batch_query_ids = query_ids[start:(start + args.search_batch_size)]
            batch_score = torch.mm(batch_query_embed, shard_psg_embed.t())
            batch_sorted_score, batch_sorted_indices = torch.topk(batch_score, k=args.search_topk, dim=-1, largest=True)
            for batch_idx, query_id in enumerate(batch_query_ids):
                cur_scores = batch_sorted_score[batch_idx].cpu().tolist()
                cur_indices = [idx + psg_idx_offset for idx in batch_sorted_indices[batch_idx].cpu().tolist()]
                query_id_to_topk[query_id] += list(zip(cur_scores, cur_indices))
                query_id_to_topk[query_id] = sorted(query_id_to_topk[query_id], key=lambda t: (-t[0], t[1]))
                query_id_to_topk[query_id] = query_id_to_topk[query_id][:args.search_topk]

        psg_idx_offset += shard_psg_embed.shape[0]

    out_path = _get_topk_result_save_path(worker_idx=gpu_idx)
    with open(out_path, 'w', encoding='utf-8') as writer:
        for query_id in query_id_to_text:
            for rank, (score, doc_id) in enumerate(query_id_to_topk[query_id]):
                writer.write('{}\t{}\t{}\t{}\n'.format(query_id, doc_id, rank + 1, round(score, 4)))

    logger.info('Write scores to {} done'.format(out_path))


def _compute_and_save_metrics(worker_cnt: int):
    preds: Dict[str, List[ScoredDoc]] = {}
    for worker_idx in range(worker_cnt):
        path = _get_topk_result_save_path(worker_idx)
        preds.update(load_msmarco_predictions(path))
    out_path = os.path.join(args.search_out_dir, '{}.msmarco.txt'.format(args.search_split))
    save_preds_to_msmarco_format(preds, out_path)
    logger.info('Merge done: save {} predictions to {}'.format(len(preds), out_path))

    path_qrels = os.path.join(args.data_dir, '{}_qrels.txt'.format(args.search_split))
    if os.path.exists(path_qrels):
        qrels = load_qrels(path=path_qrels)
        all_metrics = trec_eval(qrels=qrels, predictions=preds)
        all_metrics['mrr'] = compute_mrr(qrels=qrels, predictions=preds)

        logger.info('{} trec metrics = {}'.format(args.search_split, json.dumps(all_metrics, ensure_ascii=False, indent=4)))
        save_json_to_file(all_metrics, os.path.join(args.search_out_dir, 'metrics_{}.json'.format(args.search_split)))
    else:
        logger.warning('No qrels found for {}'.format(args.search_split))

    # do some cleanup
    for worker_idx in range(worker_cnt):
        path = _get_topk_result_save_path(worker_idx)
        os.remove(path)


def _batch_search_queries():
    logger.info('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error('No gpu available')
        return

    logger.info('Use {} gpus'.format(gpu_count))
    torch.multiprocessing.spawn(_worker_batch_search, args=(), nprocs=gpu_count)
    logger.info('Done batch search queries')

    _compute_and_save_metrics(gpu_count)


if __name__ == '__main__':
    _batch_search_queries()
