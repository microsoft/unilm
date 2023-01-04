import os
import tqdm
import torch

from contextlib import nullcontext
from torch.utils.data import DataLoader
from functools import partial
from datasets import Dataset, load_dataset
from typing import Dict, List
from transformers.file_utils import PaddingStrategy
from transformers.modeling_outputs import SequenceClassifierOutput
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
from models import RerankerForInference
from data_utils import load_corpus, load_queries, save_to_readable_format

parser = HfArgumentParser((Arguments,))
args: Arguments = parser.parse_args_into_dataclasses()[0]
kd_gen_score_in_path = os.path.join(args.data_dir, '{}.jsonl'.format(args.kd_gen_score_split))
kd_gen_score_out_path = os.path.join(args.data_dir, 'kd_{}.jsonl'.format(args.kd_gen_score_split))


def _kd_gen_score_transform_func(tokenizer: PreTrainedTokenizerFast,
                                 corpus: Dataset,
                                 queries: Dict[str, str],
                                 examples: Dict[str, List]) -> BatchEncoding:
    input_docs: List[str] = []

    # ATTENTION: this code should be consistent with CrossEncoderDataLoader
    for doc_id in examples['doc_id']:
        doc_id = int(doc_id)
        prefix = ''
        if corpus[doc_id].get('title', ''):
            prefix = corpus[doc_id]['title'] + ': '
        input_docs.append(prefix + corpus[doc_id]['contents'])

    input_queries = [queries[query_id] for query_id in examples['query_id']]
    batch_dict = tokenizer(input_queries,
                           text_pair=input_docs,
                           max_length=args.rerank_max_length,
                           padding=PaddingStrategy.DO_NOT_PAD,
                           truncation=True)

    return batch_dict


def _get_shard_path(worker_idx: int) -> str:
    return '{}_shard_{}'.format(kd_gen_score_in_path, worker_idx)


@torch.no_grad()
def _worker_gen_teacher_score(gpu_idx: int):
    dataset = load_dataset('json', data_files=kd_gen_score_in_path)['train']
    if args.dry_run:
        dataset = dataset.select(range(100))
    dataset = dataset.shard(num_shards=torch.cuda.device_count(),
                            index=gpu_idx,
                            contiguous=True)

    qid_pids = []
    for ex in tqdm.tqdm(dataset, desc='get qid-pid pairs', mininterval=3):
        for pos_doc_id in ex['positives']['doc_id']:
            qid_pids.append((ex['query_id'], pos_doc_id))
        for neg_doc_id in ex['negatives']['doc_id'][:args.kd_gen_score_n_neg]:
            qid_pids.append((ex['query_id'], neg_doc_id))

    dataset = Dataset.from_dict({'query_id': [t[0] for t in qid_pids],
                                 'doc_id': [t[1] for t in qid_pids]})

    query_ids, doc_ids = dataset['query_id'], dataset['doc_id']

    logger.info('GPU {} needs to process {} examples'.format(gpu_idx, len(dataset)))
    torch.cuda.set_device(gpu_idx)

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model: RerankerForInference = RerankerForInference.from_pretrained(args.model_name_or_path)
    model.eval()
    model.cuda()

    corpus: Dataset = load_corpus(path=os.path.join(args.data_dir, 'passages.jsonl.gz'))
    queries = load_queries(path='{}/{}_queries.tsv'.format(args.data_dir, args.kd_gen_score_split),
                           task_type=args.task_type)
    dataset.set_transform(partial(_kd_gen_score_transform_func, tokenizer, corpus, queries))

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    data_loader = DataLoader(
        dataset,
        batch_size=args.kd_gen_score_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=True)

    scores = []
    for batch_dict in tqdm.tqdm(data_loader, desc='generate teacher score', mininterval=5):
        batch_dict = move_to_cuda(batch_dict)

        with torch.cuda.amp.autocast() if args.fp16 else nullcontext():
            outputs: SequenceClassifierOutput = model(batch_dict)
        scores.append(outputs.logits.squeeze(dim=-1).cpu())
        assert len(scores[-1].shape) == 1

    all_scores = torch.cat(scores, dim=-1)
    assert all_scores.shape[0] == len(dataset), '{} != {}'
    all_scores = all_scores.tolist()

    with open(_get_shard_path(gpu_idx), 'w', encoding='utf-8') as writer:
        for idx in range(len(query_ids)):
            writer.write('{}\t{}\t{}\n'.format(query_ids[idx], doc_ids[idx], round(all_scores[idx], 5)))

    logger.info('Done computing teacher score for worker {}'.format(gpu_idx))


def _merge_teacher_scores(worker_cnt: int):
    qid_to_pid_to_score = {}
    for worker_idx in range(worker_cnt):
        shard_path = _get_shard_path(worker_idx)
        for line in tqdm.tqdm(open(shard_path, 'r', encoding='utf-8'),
                              desc='Load shard {} score'.format(worker_idx), mininterval=3):
            fs = line.strip().split('\t')
            assert len(fs) == 3
            qid, pid, score = fs
            if qid not in qid_to_pid_to_score:
                qid_to_pid_to_score[qid] = {}
            qid_to_pid_to_score[qid][pid] = float(score)
        os.remove(shard_path)

    dataset = load_dataset('json', data_files=kd_gen_score_in_path)['train']
    if args.dry_run:
        dataset = dataset.select(range(100))

    def _update_score(ex: Dict) -> Dict:
        query_id = ex['query_id']
        pid_to_score = qid_to_pid_to_score[query_id]
        ex['negatives']['doc_id'] = [neg_doc_id for neg_doc_id in ex['negatives']['doc_id'] if neg_doc_id in pid_to_score]
        ex['positives']['score'] = [pid_to_score[pos_doc_id] for pos_doc_id in ex['positives']['doc_id']]
        ex['negatives']['score'] = [pid_to_score[neg_doc_id] for neg_doc_id in ex['negatives']['doc_id']]
        return ex

    dataset = dataset.map(_update_score, num_proc=4)
    logger.info('Writing teacher score to {}'.format(kd_gen_score_out_path))
    dataset.to_json(kd_gen_score_out_path, force_ascii=False, lines=True)


def _batch_compute_teacher_score():
    logger.info('Args={}'.format(str(args)))
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.error('No gpu available')
        return

    logger.info('Use {} gpus'.format(gpu_count))
    torch.multiprocessing.spawn(_worker_gen_teacher_score, args=(), nprocs=gpu_count)
    logger.info('Done batch generate teacher score')

    _merge_teacher_scores(gpu_count)
    logger.info('Done merge results')

    corpus = load_corpus(path=os.path.join(args.data_dir, 'passages.jsonl.gz'))
    save_to_readable_format(in_path=kd_gen_score_out_path, corpus=corpus)


if __name__ == '__main__':
    _batch_compute_teacher_score()
